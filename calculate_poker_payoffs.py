import glob
import csv
from collections import defaultdict

# In 2011, the format for the logs changed and remained consistent
names = set()
names.add("Intermission_2pn_2017")
names.add("RobotShark_iro_2pn_2017")
names.add("PPPIMC_2pn_2017")
names.add("ElDonatoro_2pn_2017")

matchups = defaultdict(int)

with open('payoffs.csv', 'w+', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for tournament in sorted(list(glob.glob("./downloads/competitions/2*"))):
        year = tournament[-4:]
        if int(year) < 2017:
            continue

        print(year)
        name_set = set()
        final_rows = []
        matchup_payouts = defaultdict(float)
        matchup_counts = defaultdict(int)
        for logfile in list(glob.glob(tournament + "/**/*0.log", recursive=True)):
            logfile1 = logfile
            logfile2 = logfile[:-5] + "1.log"
            with open(logfile1) as log1:
                with open(logfile2) as log2:
                    reader1 = csv.reader(log1, delimiter=':')
                    reader2 = csv.reader(log2, delimiter=':')
                    for row1, row2 in zip(reader1, reader2):
                        if row1[0] != "STATE" or row2[0] != "STATE":
                            continue
                        first_scores = row1[4]
                        first_scores = first_scores.split('|')
                        second_scores = row2[4]
                        second_scores = second_scores.split('|')
                        first_names = row1[5]
                        first_names = first_names.split('|')
                        second_names = row2[5]
                        second_names = second_names.split('|')
                        assert first_names[0] == second_names[1], "{} {}".format(logfile, row1)
                        assert first_names[1] == second_names[0]
                        #if first_names[0] not in names \
                        #   or first_names[1] not in names \
                        #   or matchups[(first_names[0], first_names[1])] > 2:
                        #    continue
                        matchups[(first_names[0], first_names[1])] += 1
                        name_set.add(second_names[0])
                        name_set.add(second_names[1])
                        score1 = int(float(second_scores[1]))
                        score2 = int(float(first_scores[0]))
                        combined_score = score1 + score2
                        #if "Intermission_2pn_2017" in first_names and "RobotShark_iro_2pn_2017" in first_names:
                        #    print(first_names)
                        #    print(combined_score)
                        matchup_payouts[(first_names[0], first_names[1])] += combined_score
                        matchup_counts[(first_names[0], first_names[1])] += 1
                        score1 = int(float(second_scores[0]))
                        score2 = int(float(first_scores[1]))
                        combined_score = score1 + score2
                        #if "Intermission_2pn_2017" in first_names and "RobotShark_iro_2pn_2017" in first_names:
                        #    print(second_names)
                        #    print(combined_score)
                        matchup_payouts[(second_names[0], second_names[1])] += combined_score
                        matchup_counts[(second_names[0], second_names[1])] += 1
        names = list(name_set)
        first_row = [""]
        first_row += names
        writer.writerow(first_row)
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
            writer.writerow(values)

