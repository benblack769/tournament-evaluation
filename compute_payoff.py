from run_experiment import compute_score, compute_normalized_score, compute_win_rate, compute_score_nash
from run_experiment import print_ranking
import numpy as np
import pandas

def eval_payoff(score_fn):
    df = pandas.read_csv("payoffs.csv", index_col=0)

    matrix = df.reset_index(drop=True).values
    print(matrix)
    scores = score_fn(np.triu(matrix), np.ones_like(matrix))
    print_ranking(scores, df.index)

eval_payoff(compute_score_nash)
