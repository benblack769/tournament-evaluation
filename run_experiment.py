import numpy as np
import random
import sys
import argparse
import math
import time
from nash_solver import max_entropy_nash, entropy_regularized_nash
from open_spiel.python.egt import alpharank
from open_spiel.python.egt import heuristic_payoff_table, utils
from extract_data import generate_data, payoffs_from_matchups
import multiprocessing


def compute_alpha(payoff):
    # TODO: To debug this, print out the matrix game from utils and see what it looks like
    payoff = np.asarray(payoff)
    payoff_matrix = np.asarray([np.copy(payoff), np.copy(payoff)])
    payoff_tables = [heuristic_payoff_table.from_matrix_game(payoff_matrix[0]),
                     heuristic_payoff_table.from_matrix_game(payoff_matrix[1])]

    # Check if the game is symmetric (i.e., players have identical strategy sets
    # and payoff tables) and return only a single-player’s payoff table if so.
    # This ensures Alpha-Rank automatically computes rankings based on the
    # single-population dynamics.
    _, payoff_tables = utils.is_symmetric_matrix_game(payoff_tables)
#
    (_, _, pi, _, _) = alpharank.compute(payoff_tables, alpha=30)
    return pi

def compute_score(result, game_count):
    assert np.equal(np.triu(result), result).all()
    return (result - result.T).sum(axis=1) / (len(result) - 1)

def compute_normalized_score(result, game_count):
    assert np.equal(np.triu(result), result).all()
    assert np.equal(np.triu(game_count), game_count).all()
    weighted_result = result * game_count
    return (weighted_result - weighted_result.T) / (game_count + game_count.T).sum(axis=1)

def compute_win_rate(result, game_count):
    return compute_score(np.sign(result), game_count)

def compute_score_nash(result, game_count):
    assert np.equal(np.triu(result), result).all()
    return max_entropy_nash(result - result.T)

def compute_score_regularized_nash(result, game_count):
    assert np.equal(np.triu(result), result).all()
    return entropy_regularized_nash(result - result.T)

def compute_score_alpharank(result, game_count):
    assert np.equal(np.triu(result), result).all()
    result = result / np.abs(result).max()
    return compute_alpha(result - result.T)


# def variance_experiment(event_df, score_fn, selection_size):
#     event_size = len(event_df)
#     event_df = event_df.reset_index()
#
#     rankings = []
#     for i in range(10):
#         selected_df = event_df.loc[np.random.choice(event_size,size=selection_size,replace=False)]
#
#         event_matrix, player_ids, game_count_matrix = csv_data_to_matrix(selected_df)
#
#         scores = score_fn(event_matrix, game_count_matrix)


# def ranking_similiarity(scores1, scores2):



def ranking_similarity(event_df, score_fn1, score_fn2):
    event_matrix, player_ids, game_count_matrix = csv_data_to_matrix(event_df)
    scores1 = score_fn1(event_matrix, game_count_matrix)
    scores2 = score_fn2(event_matrix, game_count_matrix)
    return ranking_similiarity(scores1, scores2)


def modify_game_matrix(cur_matrix, orig_matrix, colluding_agents, min_score, allow_matchthrows):
    '''
    randomly modify game matrix with restrictions

    Makes exactly one modification, appropriate for simulated annealing
    '''
    p1 = 0
    p2 = 0
    while p1 == p2:
        p1 = random.randrange(0, len(cur_matrix)) if allow_matchthrows else colluding_agents[random.randrange(0, len(colluding_agents))]
        p2 = colluding_agents[random.randrange(0, len(colluding_agents))]

    p1e = min(p1, p2)
    p2e = max(p1, p2)
    sign = 1 if p2 < p1 else -1
    # print(p1e,p2e)

    cur_value = cur_matrix[p1e,p2e]
    orig_value = orig_matrix[p1e,p2e]

    if cur_value != orig_value:
        # if alredy altered, then reset
        cur_matrix[p1e,p2e] = orig_matrix[p1e,p2e]
    else:
        # if unaltered, then minimize score
        cur_matrix[p1e,p2e] = min_score * sign


def get_rank(scores):
    order = np.argsort(-scores)
    result = np.zeros_like(order)
    result[order] = np.arange(len(order))
    return result


def sigmoid(logit):
    return 1./(1+math.exp(-logit)) if logit > -10 else 1.

def competition_fn(score1, score2, value1, value2, temp):
    energy_gained = score1 - score2
    rejection_prob = sigmoid(energy_gained*temp)
    rejected = random.random() < rejection_prob
    return value1 if rejected else value2

def genetic_algorithm(score_fn, mutate_fn, initial_value_fn, crossover_fn, initial_value):
    pop = 10
    n_crossovers = 5
    n_mutates = 5
    n_reinits = 3
    n_initials = 2

    n_iters = 100
    population = [initial_value_fn() for i in range(pop)]
    best_score = score_fn(initial_value)
    best_result = initial_value
    start = time.time()
    tot_time = 0
    for i in range(n_iters):
        s = time.time()
        population += [crossover_fn(random.choice(population),random.choice(population)) for i in range(n_crossovers)]
        population += [mutate_fn(random.choice(population)) for i in range(n_mutates)]
        population += [initial_value_fn() for i in range(n_reinits)]
        population += [initial_value for i in range(n_initials)]
        # find best agent in whole run of optimization
        scores = [score_fn(agent) for agent in population]
        e = time.time()
        cur_best_score = min(scores)
        cur_best_result = population[np.argmin(scores)]

        if cur_best_score < best_score:
            best_score = cur_best_score
            best_result = cur_best_result

        # probablistically remove agents from population
        temp = n_iters/(n_iters-i)
        while len(population) > pop:
            i1, i2 = np.random.choice(len(population), size=2, replace=False)
            removed_i = competition_fn(scores[i1], scores[i2], i1, i2, temp)
            del population[removed_i]
        tot_time += e - s
    end = time.time()
    # print(tot_time, "\t",end-start)
    return best_score, best_result



def simulated_annealing(score_fn, modify_fn, initial_value):
    value = initial_value
    cur_score = score_fn(value)
    best_value = initial_value
    best_score = cur_score
    n_iters = 300
    for i in range(n_iters):
        # linear temperature schedule
        temp = n_iters/(n_iters-i)

        new_value = modify_fn(value)
        new_score = score_fn(new_value)

        energy_gained = cur_score - new_score
        acceptance_prob = sigmoid(energy_gained*temp)
        accepted = random.random() < acceptance_prob
        if new_score < best_score:
            # print(new_score, cur_score, acceptance_prob, temp)
            best_score = new_score
            best_value = new_value
        if accepted:
            value = new_value
            cur_score = new_score
    return best_score, best_value

def single_exploit_experiment(event_matrix, score_fn, n_colluding, min_score, allow_matchthrows):
    np.random.seed(random.randint(0,1<<32))
    game_count_matrix = np.ones_like(event_matrix)
    colluding_agents = [int(x) for x in np.random.choice(len(event_matrix), size=n_colluding, replace=False)]

    def initial_value():
        matrix_copy = event_matrix.copy()
        for i in range(100):
            modify_game_matrix(matrix_copy, event_matrix, colluding_agents, min_score, allow_matchthrows)
        return matrix_copy

    def crossover(m1, m2):
        m1f = m1.flatten()
        m2f = m2.flatten()
        assert len(m1f) == len(m2f)
        mask = np.random.randint(0, 1, size=len(m1f))
        return (m1f*mask + m2f*(1-mask)).reshape(m1.shape)

    def sim_score_fn(matrix):
        all_scores = score_fn(matrix, game_count_matrix)
        full_ranking = get_rank(all_scores)
        min_rank_colluding = min(full_ranking[colluding_agents])
        return min_rank_colluding

    def mutate(src_matrix):
        dest_matrix = src_matrix.copy()
        modify_game_matrix(dest_matrix, event_matrix, colluding_agents, min_score, allow_matchthrows)
        return dest_matrix

    orig_score = sim_score_fn(event_matrix)
    best_score, best_matrix = genetic_algorithm(sim_score_fn, mutate, initial_value, crossover, event_matrix)
    best_score_ann, best_matrix_ann = simulated_annealing(sim_score_fn, mutate, event_matrix)

    genetic_rank_increase = orig_score - best_score
    anneal_rank_increase = orig_score - best_score_ann
    if False and orig_score >= orig_score + 2:
        print(colluding_agents)
        print(best_matrix - event_matrix)
    return genetic_rank_increase, anneal_rank_increase

def run_exploit_arg(arg):
    return single_exploit_experiment(*arg)

def exploitability_experiment(event_matrix, score_fn, n_colluding=2, min_score=None, allow_matchthrows=True):
    assert len(event_matrix) > 2
    n_players = len(event_matrix)

    if min_score is None:
        min_score = min(np.min(event_matrix), -np.max(event_matrix))

    num_tests = 100
    total_rank_increase = 0
    total_annealing_rank_increase = 0
    args = (event_matrix, score_fn, n_colluding, min_score, allow_matchthrows)
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    results = pool.map(run_exploit_arg, [args]*num_tests)
    for i in range(num_tests):
        genetic_rank_increase, anneal_rank_increase = results[i]
        total_rank_increase += genetic_rank_increase
        total_annealing_rank_increase += anneal_rank_increase
    # for i in range(num_tests):
    #     genetic_rank_increase, anneal_rank_increase = single_exploit_experiment(*args)
    #     total_rank_increase += genetic_rank_increase
    #     total_annealing_rank_increase += anneal_rank_increase

    print(total_rank_increase/num_tests, total_annealing_rank_increase/num_tests)


def run_experiment(payoffs):
    for event_payoff in payoffs:
        # event_matrix, player_ids, game_count_matrix = csv_data_to_matrix(event_data)
        print("match throws allowed")
        exploitability_experiment(event_payoff, compute_score, n_colluding=2, min_score=None, allow_matchthrows=True)
        print("match throws disallowed")
        exploitability_experiment(event_payoff, compute_score, n_colluding=2, min_score=None, allow_matchthrows=False)
        # print(event_matrix)
        # print(game_count_matrix)
        # scores = score_fn(event_matrix, game_count_matrix)
        #
        # print_ranking(scores,player_ids)

def print_ranking(score, player_names):
    assert len(player_names) == len(score)
    ranking = list(reversed(sorted([(s,i,n) for i, (s, n) in enumerate(zip(score, player_names))])))
    for r in ranking:
        print(*r)

def print_mat(x):
    print(x*5)

def print_all_rankings(df, score_fn):
    events = set(df['event'])
    for event in list(events):
        print(event)
        event_data = df[df['event'] == event]
        event_matrix, player_ids, game_count_matrix = csv_data_to_matrix(event_data)
        print(event_matrix)
        print(game_count_matrix)
        scores = score_fn(event_matrix, game_count_matrix)
        ranking = list(reversed(sorted([(s,i,n) for i, (s, n) in enumerate(zip(scores, player_ids))])))
        for r in ranking:
            print(*r)
        indicies = [i for s,i,n in ranking]
        event_matrix = event_matrix - event_matrix.T
        ordered_mat = np.zeros_like(event_matrix)
        ordered_mat[indicies] = event_matrix
        # print(ordered_mat)
        # print(event_matrix - event_matrix.T)

def load_payoffs(csv_fnames):
    all_payouts = []

    for fname in csv_fnames:
        names, matchup_results = generate_data(fname)
        payoff_matrix = payoffs_from_matchups(names, matchup_results, clip=False)
        payoff_matrix = np.triu(payoff_matrix)
        all_payouts.append(payoff_matrix)

    return all_payouts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute rank similarities')
    parser.add_argument('--inputs', required=True, nargs='*', help='Tournament datas to construct inputs for')

    args = parser.parse_args()

    # df = pd.read_csv(fname)
    payoffs = load_payoffs(args.inputs)
    run_experiment(payoffs)
