# !/usr/local/bin/python
import glob
import csv
from collections import defaultdict

def generate_payoffs(included_ratio=1.0):
    matchups = defaultdict(int)
    winrates = defaultdict(float)
    matches = defaultdict(int)
    name_set = set()
    final_rows = []
    matchup_payouts = defaultdict(float)
    matchup_counts = defaultdict(int)
    with open("new_logs.csv") as log:
        reader = csv.DictReader(log)
        for row in reader:
            player1 = row["Player 1"]
            player2 = row["Player 2"]
            matchup = (player1, player2)
            matchup2 = (player2, player1)
            matchups[matchup] += 1
            name_set.add(player1)
            name_set.add(player2)
            score = float(row["Earnings"])
            matchup_payouts[matchup] += score
            matchup_counts[matchup] += 1
            matchup_payouts[matchup2] -= score
            matchup_counts[matchup2] += 1
            winrates[player1] += score
            matches[player1] += 1
            winrates[player2] -= score
            matches[player2] += 1
    names = list(name_set)
    final_rows = []
    winrate_list = []
    for name in names:
        values = [name]
        for other_name in names:
            score = matchup_payouts[(name, other_name)]
            count = matchup_counts[(name, other_name)]
            summed = 0
            if count != 0:
                summed = score / count
            summed = summed * 5
            summed = min(summed, 750)
            summed = max(summed, -750)
            values.append(summed)
        wr = sum(values[1:]) / len([v for v in values[1:] if v != 0])
        wr = min(wr, 750)
        wr = max(wr, -750)
        values.append(wr)
        winrate_list.append(wr)
        final_rows.append(values)

    winrate_list, indices = map(list, zip(*sorted(zip(winrate_list, range(1, len(winrate_list)+1)))))
    new_values = []
    for row in final_rows:
        new_row = [row[0]]
        for index in indices[::-1]:
            new_row.append(row[index])
        wr = sum(new_row[1:]) / len([v for v in new_row[1:] if v != 0])
        wr = min(wr, 750)
        wr = max(wr, -750)
        new_row.append(wr)
        new_values.append(new_row)
        
    new_values.sort(key = lambda x: x[-1])
    new_first_row = [""]
    new_first_row += [row[0] for row in new_values[::-1]]
    new_first_row.append("Winrate")
    new_values.sort(key = lambda x: x[-1])
    payoffs = []
    for row in new_values[::-1]:
        payoffs.append(row[1:-1])


    keys = new_first_row[1:]
    return keys, payoffs 
    
