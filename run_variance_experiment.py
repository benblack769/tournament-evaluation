# !/usr/local/bin/python
import time
import multiprocessing
import random
from statistics import stdev
import numpy as np
from tqdm import tqdm, trange
from open_spiel.python.egt import alpharank
from open_spiel.python.egt import heuristic_payoff_table, utils
import extract_payoffs
import extract_data
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
    def eval_payoff(score_func, matrix, keys):
        return score_func(np.triu(matrix), np.ones_like(matrix))
    return eval_payoff(compute_score_nash, payoff_matrix, keys)


def compute_winrate(payoff_matrix, keys):
    payoff_matrix = np.asarray(payoff_matrix)
    row_sums = np.sum(payoff_matrix, axis=1)
    nonzero_payoffs = np.count_nonzero(payoff_matrix != 0, axis=0)
    return (row_sums / nonzero_payoffs) + 750


def get_ranking(score_list, agent_list):
    scores_s, agents_s = zip(*sorted(zip(score_list, agent_list), reverse=True))
    ranking = []
    for agent in agents_s:
        ranking.append(agent_to_index[agent])
    return ranking, scores_s


def iterative_ranking(payoff_matrix, all_agents, score_fn):
    current_agents = all_agents.copy()
    payoff_matrix = np.asarray(payoff_matrix)
    final_ranking = []
    final_scores = []
    while len(final_ranking) < len(all_agents):
        # Compute scores and rank
        scores = score_fn(payoff_matrix, current_agents)
        ranking, scores = get_ranking(scores, current_agents)
        #print("Scores: {}".format(scores))
        #print("Ranking: {}".format(ranking))

        # Add anyone in the metagame to the ranking, in order
        indices_to_remove = []
        agents_to_remove = []
        for index, agent_id in enumerate(ranking):
            if scores[index] > 1e-7:
                agent_name = index_to_agent[agent_id]
                indices_to_remove.append(current_agents.index(agent_name))
                agents_to_remove.append(agent_name)
                final_ranking.append(all_agents.index(agent_name))
                final_scores.append(scores[index])
        # Remove ranked agents from payoff matrix and agent list
        for agent in agents_to_remove:
            current_agents.remove(agent)
        payoff_matrix = np.delete(payoff_matrix, indices_to_remove, axis=0)
        payoff_matrix = np.delete(payoff_matrix, indices_to_remove, axis=1)

    return final_ranking, final_scores


# TODO: Run experiment 10 times per sample size
experiment_id = random.randint(0, 10000)
file1 = open("experiment_stratified_{}.txt".format(experiment_id), "w")
file1.write("Experiment Data \n")

true_agents, true_matchup_payoffs = extract_data.generate_data()
true_payoff_matrix = extract_data.payoffs_from_matchups(true_agents, true_matchup_payoffs, sample_ratio=1.0)
file1.write("Agents: " + str(true_agents) + "\n")
print(true_agents)
print(np.asarray(true_payoff_matrix))
agent_to_index = {}
index_to_agent = []
for i, agent in enumerate(true_agents):
    agent_to_index[agent] = i
    index_to_agent.append(agent)

#scores = compute_winrate(payoffs, agents)
#total_winrate_ranking = get_ranking(scores, agents)

total_winrate_ranking, wscore = iterative_ranking(true_payoff_matrix, true_agents, compute_winrate)
print(total_winrate_ranking)

total_alpha_ranking, ascore = iterative_ranking(true_payoff_matrix, true_agents, compute_alpha)
print(total_alpha_ranking)

total_nash_ranking, nscore = iterative_ranking(true_payoff_matrix, true_agents, compute_nash)
print(total_nash_ranking)
file1.writelines([
    "Winrate Ranking: " + str(total_winrate_ranking) + "\n",
    "Meta Nash Ranking: " + str(total_nash_ranking) + "\n",
    "Alpharank Ranking: " + str(total_alpha_ranking) + "\n",
])


def test_sample(data_ratio):
    payoffs = extract_data.payoffs_from_matchups(true_agents, true_matchup_payoffs, sample_ratio=data_ratio)
    print("Payoffs")
    print(true_agents)
    print(np.asarray(payoffs))
    errors = np.absolute(true_payoff_matrix - payoffs) / 1500
    print(errors)
    print(np.max(errors))
    print(np.mean(errors))
    print(np.min(errors))

    # Agents are ranked according to winrate by default
    winrate_ranking, _ = iterative_ranking(payoffs, true_agents, compute_winrate)
    winrate_spear = spearman(total_winrate_ranking, winrate_ranking)
    print("Winrate: ", winrate_ranking)
    print("Winrate: ", winrate_spear)

    # Compute Nash Rank
    nash_ranking, _ = iterative_ranking(payoffs, true_agents, compute_nash)
    nash_spear = spearman(total_nash_ranking, nash_ranking)
    print("Nash: ", nash_ranking)
    print("Winrate: ", nash_spear)

    # Compute Alpharank
    alpha_ranking, _ = iterative_ranking(payoffs, true_agents, compute_alpha)
    alpha_spear = spearman(total_alpha_ranking, alpha_ranking)
    print("Alpha: ", alpha_ranking)
    print("Winrate: ", alpha_spear)

    file1.write("\t" + str(i) + ": " + str(winrate_spear) + ", " + str(nash_spear) + ", " + str(alpha_spear) + "\n")
    return [winrate_spear, nash_spear, alpha_spear]


N = 200
winrate_spears = []
nash_spears = []
alpha_spears = []
scales = [i/10.0 for i in range(1, 10)]
scales = [0.0001] + [0.001] + [0.01] + scales + [0.99] + [0.999] + [0.9999]
file1.write("Scales: " + str(scales) + "\n")
file1.flush()
scales_loop = tqdm(scales, leave=False)
for data_ratio in scales_loop:
    file1.write("Testing samples of {}% of data...\n".format(data_ratio * 100))
    scales_loop.set_description("Testing {}% of data".format(data_ratio*100))

    with multiprocessing.Pool(processes=12) as a_pool:
        result = list(tqdm(a_pool.imap(test_sample, [data_ratio] * N), total=N))
        result = np.asarray(result)

    spear_sums = [sum(col) for col in zip(*result)]
    spear_stdevs = [stdev(col) for col in zip(*result)]
    average_winrate_spear = spear_sums[0] / N
    winrate_spears.append(average_winrate_spear)

    average_nash_spear = spear_sums[1] / N
    nash_spears.append(average_nash_spear)

    average_alpha_spear = spear_sums[2] / N
    alpha_spears.append(average_alpha_spear)
    file1.writelines([
        "\tWinrate Spearman Mean: " + str(average_winrate_spear) + "\n",
        "\tMeta Nash Spearman Mean: " + str(average_nash_spear) + "\n",
        "\tAlpharank Spearman Mean: " + str(average_alpha_spear) + "\n",
        "\tWinrate Spearman Standard Deviation: " + str(spear_stdevs[0]) + "\n",
        "\tMeta Nash Spearman Standard Deviation: " + str(spear_stdevs[1]) + "\n",
        "\tAlpharank Spearman Standard Deviation: " + str(spear_stdevs[2]) + "\n",
    ])
    file1.flush()

print(winrate_spears)
print(nash_spears)
print(alpha_spears)
file1.writelines([
    "Winrate Spearmans: " + str(winrate_spears) + "\n",
    "Meta Nash Spearmans: " + str(nash_spears) + "\n",
    "Alpharank Spearmans: " + str(alpha_spears) + "\n",
])
file1.close()
