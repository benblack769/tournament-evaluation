# !/usr/local/bin/python
import extract_payoffs
from run_experiment import compute_score, compute_normalized_score, compute_win_rate, compute_score_nash
from run_experiment import print_ranking
import numpy as np
import pandas

np.set_printoptions(linewidth=200)

def compute_nash(payoffs, keys):
    def eval_payoff(score_fn, payoffs, keys):
        #df = pandas.read_csv("payoffs2.csv", index_col=0)
        #print()
        #print(df.index)
        #print()
    
        #matrix = df.reset_index(drop=True)
        #print(matrix.columns.values)
        #matrix = matrix.values
        #print(matrix)
        matrix = payoffs
        scores = score_fn(np.triu(matrix), np.ones_like(matrix))
        print(scores, keys)
        print(len(scores), len(keys))
        print_ranking(scores, keys)
    
    eval_payoff(compute_score_nash, payoffs, keys)
agents, payoffs = extract_payoffs.generate_payoffs()
print(agents)
print(payoffs)
compute_nash(payoffs, agents)
