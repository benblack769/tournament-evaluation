# !/usr/local/bin/python
import time
import numpy as np
from tqdm import tqdm, trange
from open_spiel.python.egt import alpharank
from open_spiel.python.egt import heuristic_payoff_table, utils
import extract_payoffs
from run_experiment import compute_score_nash
from utils import spearman
np.set_printoptions(linewidth=200)


def compute_alpha(payoff, keys):
    # TODO: To debug this, print out the matrix game from utils and see what it looks like
    payoff = np.asarray(payoff) / 750.0
    payoff_matrix = np.asarray([payoff, np.copy(payoff)])
    payoff_tables = [heuristic_payoff_table.from_matrix_game(payoff_matrix[0]),
                     heuristic_payoff_table.from_matrix_game(payoff_matrix[1])]

    # Check if the game is symmetric (i.e., players have identical strategy sets
    # and payoff tables) and return only a single-player’s payoff table if so.
    # This ensures Alpha-Rank automatically computes rankings based on the
    # single-population dynamics.
    _, payoff_tables = utils.is_symmetric_matrix_game(payoff_tables)

    (_, _, pi, _, _) = alpharank.compute(payoff_tables, alpha=300)
    return pi


def compute_nash(payoff_matrix, keys):
    def eval_payoff(score_fn, matrix, keys):
        return score_fn(np.triu(matrix), np.ones_like(matrix))
    return eval_payoff(compute_score_nash, payoff_matrix, keys)


def get_ranking(score_list, agent_list):
    _, agents_s = zip(*sorted(zip(score_list, agent_list), reverse=True))
    ranking = []
    for agent in agents_s:
        ranking.append(agent_to_index[agent])
    return ranking


def iterative_ranking(payoff_matrix, all_agents, score_fn):
    current_agents = all_agents.copy()
    payoff_matrix = np.asarray(payoff_matrix)
    final_ranking = []
    while len(final_ranking) < len(all_agents):
        # Compute scores and rank
        scores = score_fn(payoff_matrix, current_agents)
        ranking = get_ranking(scores, current_agents)

        # Add anyone in the metagame to the ranking, in order
        indices_to_remove = []
        agents_to_remove = []
        for index, agent_id in enumerate(ranking):
            if scores[index] > 1e-7:
                indices_to_remove.append(index)
                agent_name = all_agents[agent_id]
                agents_to_remove.append(agent_name)
                final_ranking.append(all_agents.index(agent_name))
        # Remove ranked agents from payoff matrix and agent list
        for agent in agents_to_remove:
            current_agents.remove(agent)
        payoff_matrix = np.delete(payoff_matrix, indices_to_remove, axis=0)
        payoff_matrix = np.delete(payoff_matrix, indices_to_remove, axis=1)

    return final_ranking


# TODO: Run experiment 10 times per sample size
agents, payoffs = extract_payoffs.generate_payoffs(included_ratio=1.0)
agent_to_index = {}
index_to_agent = []
for i, agent in enumerate(agents):
    agent_to_index[agent] = i
    index_to_agent.append(agent)

time.sleep(100)
total_winrate_ranking = get_ranking(range(len(agents), 0, -1), agents)
print(total_winrate_ranking)

total_alpha_ranking = iterative_ranking(payoffs, agents, compute_alpha)
print(total_alpha_ranking)

total_nash_ranking = iterative_ranking(payoffs, agents, compute_nash)
print(total_nash_ranking)


winrate_spears = []
nash_spears = []
alpha_spears = []
scales = [i/10.0 for i in range(1, 11, 2)]
scales = [0.01] + [0.05] + scales
scales_loop = tqdm(scales, leave=False)
for data_ratio in scales_loop:
    scales_loop.set_description("Testing {}% of data".format(data_ratio*100))
    total_winrate_spear = 0
    total_nash_spear = 0
    total_alpha_spear = 0
    range_loop = tqdm(trange(5), leave=False)
    for i in range_loop:
        range_loop.set_description("Trial {}/5".format(i))
        agents, payoffs = extract_payoffs.generate_payoffs(included_ratio=data_ratio)
        alpha_payoffs = payoffs.copy()
        alpha_agents = agents.copy()

        # Agents are ranked according to winrate by default
        winrate_ranking = get_ranking(range(len(agents), 0, -1), agents)
        total_winrate_spear += spearman(total_winrate_ranking, winrate_ranking)

        # Compute Nash Rank
        nash_ranking = iterative_ranking(payoffs, agents, compute_nash)
        total_nash_spear += spearman(total_nash_ranking, nash_ranking)

        # Compute Alpharank
        alpha_ranking = iterative_ranking(payoffs, agents, compute_alpha)
        total_alpha_spear += spearman(total_alpha_ranking, alpha_ranking)

    average_winrate_spear = total_winrate_spear / 5
    winrate_spears.append(average_winrate_spear)

    average_nash_spear = total_nash_spear / 5
    nash_spears.append(average_nash_spear)

    average_alpha_spear = total_alpha_spear / 5
    alpha_spears.append(average_alpha_spear)

print(winrate_spears)
print(nash_spears)
print(alpha_spears)
