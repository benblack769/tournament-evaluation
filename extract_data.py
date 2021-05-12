# !/usr/local/bin/python
import csv
import numpy as np
from tqdm import tqdm
from collections import defaultdict
np.set_printoptions(linewidth=200)
np.set_printoptions(suppress=True)


def generate_data(included_ratio=1.0):
    name_set = set()
    matchup_payouts = defaultdict(list)
    matchup_counts = defaultdict(int)
    with open("new_logs.csv") as log:
        reader = csv.DictReader(log)
        for row in reader:
            player1 = row["Player 1"]
            player2 = row["Player 2"]
            matchup = (player1, player2)
            matchup2 = (player2, player1)
            name_set.add(player1)
            name_set.add(player2)
            score = float(row["Earnings"])
            matchup_payouts[matchup].append(score)
            matchup_counts[matchup] += 1
            matchup_payouts[matchup2].append(-score)
            matchup_counts[matchup2] += 1
    names = list(sorted(list(name_set)))
    for matchup, payout_list in matchup_payouts.items():
        matchup_payouts[matchup] = np.asarray(payout_list)
    return names, matchup_payouts


def payoffs_from_matchups(agents, payoffs, sample_ratio=1.0):
    payoff_matrix = np.zeros((len(agents), len(agents)))
    for i, agent1 in enumerate(agents):
        for j, agent2 in enumerate(agents):
            matchup = (agent1, agent2)
            payoff_array = payoffs[matchup]

            if len(payoff_array) == 0:
                payoff_matrix[i][j] = 0
                continue
 
            sample_count = int(sample_ratio * len(payoff_array))
            sample_array = np.random.choice(payoff_array, size=sample_count, replace=False)
            summed = 5 * np.sum(sample_array) / sample_count
            clipped = min(max(summed, -750), 750)
            payoff_matrix[i][j] = clipped
    return payoff_matrix


#agents, true_matchup_payoffs = generate_data()
#payoff_matrix = payoffs_from_matchups(agents, true_matchup_payoffs)
