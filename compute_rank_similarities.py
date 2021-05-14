from run_experiment import compute_score_nash, compute_score, compute_win_rate, compute_score_regularized_nash, compute_score_alpharank
from run_experiment import csv_data_to_matrix
import pandas as pd
import sys
import numpy as np
from utils import spearman
import argparse
import matplotlib
import matplotlib.pyplot as plt
from extract_data import generate_data, payoffs_from_matchups


def compute_cross_rankings(payouts, score_fns):
    event_matrix = payouts
    game_count_matrix = np.ones_like(payouts)
    rank_similarities = np.zeros((len(score_fns), len(score_fns)))

    score_results = [fn(event_matrix, game_count_matrix) for fn in score_fns]
    rankings = [np.argsort(res) for res in score_results]
    for i,_ in enumerate(score_results):
        for j,_ in enumerate(score_results):
            similarity = spearman(rankings[i],rankings[j])
            rank_similarities[i,j] += similarity

    return rank_similarities

def compute_capped_score(result, game_count):
    print(result)
    result = np.minimum(750, np.maximum(result*5, -750))
    print(result)
    return compute_score(result, game_count)

def plot_similarities(csv_fnames, out_plotname, include_cap=False):
    all_payouts = []

    for fname in csv_fnames:
        names, matchup_results = generate_data(fname)
        payoff_matrix = payoffs_from_matchups(names, matchup_results, clip=False)
        payoff_matrix = np.triu(payoff_matrix)
        all_payouts.append(payoff_matrix)

    score_fn_names = [
        "compute_score_nash",
        "compute_score",
        "compute_win_rate",
        "compute_score_regularized_nash",
        'compute_score_alpharank',
    ]
    if include_cap:
        score_fn_names += ['compute_capped_score']
    score_fns = [
        compute_score_nash,
        compute_score,
        compute_win_rate,
        compute_score_regularized_nash,
        compute_score_alpharank,
    ]
    if include_cap:
        score_fns += [compute_capped_score]
    print(all_payouts[0])
    rank_similarities = np.mean([compute_cross_rankings(payouts, score_fns) for payouts in all_payouts],axis=0)
    rank_similarities = (100*rank_similarities).round()/100
    print(all_payouts[0])

    fig, ax = plt.subplots()
    im = ax.imshow(rank_similarities)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(score_fn_names)))
    ax.set_yticks(np.arange(len(score_fn_names)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(score_fn_names)
    ax.set_yticklabels(score_fn_names)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    for i in range(len(score_fn_names)):
        for j in range(len(score_fn_names)):
            text = ax.text(j, i, rank_similarities[i, j],
                           ha="center", va="center", color="w")

    ax.set_title("Spearman correlation on poker data")
    fig.tight_layout()
    plt.savefig(out_plotname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute rank similarities')
    parser.add_argument('--outname', required=True, help='output filename')
    parser.add_argument('--inputs', required=True, nargs='*', help='Tournament datas to construct inputs for')
    parser.add_argument('--include-poker-cap', action="store_true", help='Includes row for poker cap')

    args = parser.parse_args()

    # df = pd.read_csv(fname)
    plot_similarities(args.inputs, args.outname, args.include_poker_cap)
