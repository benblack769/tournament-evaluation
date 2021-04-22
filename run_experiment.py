import pandas
import numpy as np
import random
import math

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
    for i in range(n_iters):
        population += [crossover_fn(random.choice(population),random.choice(population)) for i in range(n_crossovers)]
        population += [mutate_fn(random.choice(population)) for i in range(n_mutates)]
        population += [initial_value_fn() for i in range(n_reinits)]
        population += [initial_value for i in range(n_initials)]

        # find best agent in whole run of optimization
        scores = [score_fn(agent) for agent in population]
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


def exploitability_experiment(event_df, score_fn, n_colluding=2, min_score=None, allow_matchthrows=True):
    event_matrix, player_ids, game_count_matrix = csv_data_to_matrix(event_df)
    assert len(player_ids) > 2
    n_players = len(player_ids)

    if min_score is None:
        min_score = min(np.min(event_matrix), -np.max(event_matrix))

    num_tests = 100
    total_rank_increase = 0
    total_annealing_rank_increase = 0
    for i in range(num_tests):
        colluding_agents = [int(x) for x in np.random.choice(len(player_ids), size=n_colluding, replace=False)]

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

        total_rank_increase += orig_score - best_score
        total_annealing_rank_increase += orig_score - best_score_ann
        if False and orig_score >= orig_score + 2:
            print(colluding_agents)
            print(best_matrix - event_matrix)

    print(total_rank_increase/num_tests, total_annealing_rank_increase/num_tests)


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

if __name__ == "__main__":
    df = pandas.read_csv("clean_chess_data.csv")
    run_experiment(df,compute_score)
