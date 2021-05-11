# !/usr/local/bin/python
import csv
import numpy as np
from tqdm import tqdm
from collections import defaultdict


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
        print(len(payout_list))
    print(matchup_payouts)
    print(names)
    print(sorted(matchup_counts.values()))

    return names, matchup_payouts

generate_data()
