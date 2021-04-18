import pandas
import numpy as np

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
            event_matrix[i][j] -= sum_result
            event_matrix[j][i] += sum_result
            game_count_matrix[i][j] += count
            game_count_matrix[j][i] += count

    return event_matrix, player_ids, game_count_matrix

def compute_score(result, game_count):
    return (np.triu(result) - np.tril(result)).sum(axis=1) / (len(result) - 1)

def compute_normalized_score(result, game_count):
    return ((np.triu(result) - np.tril(result)) * game_count).sum(axis=1) / game_count.sum(axis=1)

def compute_win_rate(result, game_count):
    return compute_score(np.sign(result), game_count)

def variance_experiment(event_df, score_fn, selection_size):
    event_size = len(event_df)
    event_df = event_df.reset_index()

    rankings = []
    for i in range(10):
        selected_df = event_df.loc[np.random.choice(event_size,size=selection_size,replace=False)]

        event_matrix, player_ids, game_count_matrix = csv_data_to_matrix(selected_df)

        scores = score_fn(event_matrix, game_count_matrix)


# def ranking_similiarity(scores1, scores2):



def ranking_similarity(event_df, score_fn1, score_fn2):
    event_matrix, player_ids, game_count_matrix = csv_data_to_matrix(event_df)
    scores1 = score_fn1(event_matrix, game_count_matrix)
    scores2 = score_fn2(event_matrix, game_count_matrix)
    return ranking_similiarity(scores1, scores2)


def exploitability_experiment(event_df, score_fn):
    event_matrix, player_ids, game_count_matrix = csv_data_to_matrix(event_df)
    assert len(player_ids) > 2

    for i in range(30):
        


    pass


def run_experiment(df, score_fn):
    events = set(df['event'])
    for event in events:
        print(event)
        event_data = df[df['event'] == event]
        # event_matrix, player_ids, game_count_matrix = csv_data_to_matrix(event_data)

        variance_experiment(event_data)
        # scores = score_fn(event_matrix, game_count_matrix)
        #
        # print_ranking(scores,player_ids)


def print_ranking(score, player_names):
    assert len(player_names) == len(score)
    ranking = list(reversed(sorted([(s,i,n) for i, (s, n) in enumerate(zip(score, player_names))])))
    for r in ranking:
        print(*r)

if __name__ == "__main__":
    df = pandas.read_csv("clean_data.csv")
    run_experiment(df,compute_win_rate)
