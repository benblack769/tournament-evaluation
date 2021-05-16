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

def compute_cross_rankings(payouts, rank_fns):
    # event_matrix = payouts
    game_count_matrix = np.ones_like(payouts)
    rank_similarities = np.zeros((len(rank_fns), len(rank_fns)))

    # score_results = [fn(event_matrix, game_count_matrix) for fn in rank_fns]
    rankings = [rank_fn(payouts, game_count_matrix) for rank_fn in rank_fns]
    # for fn, rank in zip(rank_fns, rankings):
    #     print(rank, "\t", fn)
    for i,_ in enumerate(rank_fns):
        for j,_ in enumerate(rank_fns):
            similarity = spearman(rankings[i],rankings[j])
            rank_similarities[i,j] += similarity

    return rank_similarities

def compute_capped_score(result, game_count):
    # print(result)
    result = np.minimum(750, np.maximum(result*5, -750))
    # print(result)
    return compute_score(result, game_count)


def iterative_ranking(results, game_count, score_fn):
    ranking = []
    results = results.copy()
    game_count = game_count.copy()
    cur_indicies = np.arange(len(results))
    while len(results):
        assert len(results) > 1
        # print(results)
        if np.all(np.equal(results, 0)):
            ranking += list(cur_indicies)
            break
        score = score_fn(results, game_count)
        del_indicies = np.argsort(-score)
        partial_rank = cur_indicies[del_indicies]
        num_ranked = (~(score < 1e-7)).astype(np.int32).sum()
        if num_ranked == len(results) - 1:
            num_ranked += 1
        sliced_rank = del_indicies[:num_ranked]
        unranked = del_indicies[num_ranked:]
        ranking += list(partial_rank[:num_ranked])
        cur_indicies = cur_indicies[unranked]
        results = np.delete(results, sliced_rank, axis=0)
        results = np.delete(results, sliced_rank, axis=1)
        game_count = np.delete(game_count, sliced_rank, axis=0)
        game_count = np.delete(game_count, sliced_rank, axis=1)
    # print("scores",ranking)
    return ranking


def iterative_ranker(score_fn):
    # def score_fn_ranker(payoffs, keys):
    #     return score_fn(payoffs, np.ones_like(payoffs))
    def rank_fn(result, game_count):
        ranking = iterative_ranking(result, game_count, score_fn)
        # print("rank2",ranking)
        return ranking
    return rank_fn

def ranking_fn(score_fn):
    def rank_fn(result, game_count):
        return np.argsort(-score_fn(result, game_count))
    return rank_fn

def plot_similarities(csv_fnames, out_plotname, include_cap=False):
    all_payouts = []

    for fname in csv_fnames:
        names, matchup_results = generate_data(fname)
        payoff_matrix = payoffs_from_matchups(names, matchup_results, clip=False)
        payoff_matrix = np.triu(payoff_matrix)
        all_payouts.append(payoff_matrix)
        # print(names)
        # print(payoff_matrix)

    score_fn_names = [
        "compute_score_nash",
        "compute_score",
        "compute_win_rate",
        "compute_score_regularized_nash",
        'compute_score_alpharank',
    ]
    if include_cap:
        score_fn_names += ['compute_capped_score']
    rank_fns = [
        iterative_ranker(compute_score_nash),
        ranking_fn(compute_score),
        ranking_fn(compute_win_rate),
        ranking_fn(compute_score_regularized_nash),
        iterative_ranker(compute_score_alpharank),
    ]
    if include_cap:
        rank_fns += [ranking_fn(compute_capped_score)]
    # print(all_payouts[0])
    all_rank_sims = [compute_cross_rankings(payouts, rank_fns) for payouts in all_payouts]
    # print(all_rank_sims)
    rank_similarities = np.mean(all_rank_sims,axis=0)
    rank_similarities = (100*rank_similarities).round()/100
    # print(all_payouts[0])

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
