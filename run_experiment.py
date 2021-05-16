import pandas
import numpy as np
import random
import sys
from nash_solver import max_entropy_nash, entropy_regularized_nash
from open_spiel.python.egt import alpharank
from open_spiel.python.egt import heuristic_payoff_table, utils

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

def csv_data_to_matrix(event_data):
    event_data = event_data.reset_index()
    assert all(event_data['event'] == event_data['event'][0])
    del event_data['event']
    grouped = event_data.groupby(['p1','p2'])['result'].aggregate(sum_result=sum,count=len)
    player_ids = sorted(set(event_data['p1']) | set(event_data['p2']))
    num_players = len(player_ids)
    # print(grouped)
    event_matrix = np.zeros((num_players,num_players))
    game_count_matrix = np.zeros((num_players,num_players),dtype=np.int64)
    for i in range(num_players):
        for j in range(num_players):
            try:
                row = grouped.loc[player_ids[i],player_ids[j]]
                sum_result = row['sum_result']
                count = row['count']
            except Exception as e:
                sum_result = 0
                count = 0

            sign = -1
            x,y = j,i
            if j > i:
                x,y = i,j
                sign = 1
            event_matrix[x][y] += sum_result * sign / count if count > 0 else 0
            game_count_matrix[x][y] += count

    return event_matrix, player_ids, game_count_matrix

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

def exploitability_experiment(event_df, score_fn, n_colluding=2, min_score=None, allow_matchthrows=True):
    event_matrix, player_ids, game_count_matrix = csv_data_to_matrix(event_df)
    assert len(player_ids) > 2
    n_players = len(player_ids)

    if min_score is None:
        min_score = min(np.min(event_matrix), -np.max(event_matrix))

    num_tests = 100
    total_rank_increase = 0
    for i in range(num_tests):
        colluding_agents = [int(x) for x in np.random.choice(len(player_ids), size=n_colluding, replace=False)]

        event_matrix_copy = event_matrix.copy()

        orig_scores = score_fn(event_matrix_copy, game_count_matrix)

        orig_ranking = get_rank(orig_scores)
        orig_min_rank_colluding = min(orig_ranking[colluding_agents])

        cur_min_rank_colluding = orig_min_rank_colluding
        best_matrix = event_matrix_copy.copy()
        best_ranking = orig_ranking.copy()
        best_scores = orig_scores.copy()
        for i in range(1000):
            modify_game_matrix(event_matrix_copy, event_matrix, colluding_agents, min_score, allow_matchthrows)

            scores = score_fn(event_matrix_copy, game_count_matrix)

            ranking = get_rank(scores)

            cur_rank_colluding = min(ranking[colluding_agents])
            if cur_rank_colluding < cur_min_rank_colluding:
                cur_min_rank_colluding = cur_rank_colluding
                best_matrix = event_matrix_copy.copy()
                best_ranking = ranking.copy()
                best_scores = scores.copy()

        total_rank_increase += orig_min_rank_colluding - cur_min_rank_colluding
        if False and orig_min_rank_colluding >= cur_min_rank_colluding + 2:
            print(colluding_agents)
            print(orig_min_rank_colluding - cur_min_rank_colluding)
            print(orig_ranking)
            print(orig_scores)
            print(best_ranking)
            print(best_scores)
            print(best_matrix - event_matrix)

    print(total_rank_increase/num_tests)


def run_experiment(df, score_fn):
    events = set(df['event'])
    for event in list(events)[1:]:
        print(event)
        event_data = df[df['event'] == event]
        # event_matrix, player_ids, game_count_matrix = csv_data_to_matrix(event_data)
        print("match throws allowed")
        exploitability_experiment(event_data, compute_score, n_colluding=2, min_score=None, allow_matchthrows=True)
        print("match throws disallowed")
        exploitability_experiment(event_data, compute_score, n_colluding=2, min_score=None, allow_matchthrows=False)
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

if __name__ == "__main__":
    fname = sys.argv[1]
    df = pandas.read_csv(fname)
    print_all_rankings(df,compute_score_nash)
