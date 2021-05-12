from run_experiment import compute_score_nash, compute_score, compute_win_rate, compute_score_regularized_nash, compute_score_alpharank
from run_experiment import csv_data_to_matrix
import pandas as pd
import sys
import numpy as np
from utils import spearman
import matplotlib
import matplotlib.pyplot as plt


def compute_cross_rankings(df, score_fns):
    rank_similarities = np.zeros((len(score_fns), len(score_fns)))
    events = set(df['event'])
    for event in list(events):
        event_data = df[df['event'] == event]
        event_matrix, player_ids, game_count_matrix = csv_data_to_matrix(event_data)

        score_results = [fn(event_matrix, game_count_matrix) for fn in score_fns]
        rankings = [np.argsort(res) for res in score_results]
        for i,_ in enumerate(score_results):
            for j,_ in enumerate(score_results):
                rank_similarities[i,j] += spearman(rankings[i],rankings[j])


    rank_similarities /= len(events)

    return rank_similarities

def plot_similarities(df):
    score_fn_names = [
        "compute_score_nash",
        "compute_score",
        "compute_win_rate",
        "compute_score_regularized_nash",
        'compute_score_alpharank',
    ]
    score_fns = [
        compute_score_nash,
        compute_score,
        compute_win_rate,
        compute_score_regularized_nash,
        compute_score_alpharank,
    ]
    rank_similarities = compute_cross_rankings(df, score_fns)
    rank_similarities = (100*rank_similarities).round()/100

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
    plt.savefig("spearman.png")


if __name__ == "__main__":
    fname = sys.argv[1]
    df = pd.read_csv(fname)
    plot_similarities(df)
