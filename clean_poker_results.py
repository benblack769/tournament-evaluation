import glob
import csv
from collections import defaultdict

# In 2011, the format for the logs changed and remained consistent

with open('new_logs.csv', 'w+', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Tournament", "Player 1", "Player 2", "Earnings", "% of Pot", "Player Count"])
    for tournament in sorted(list(glob.glob("./downloads/competitions/2*"))):
        year = tournament[-4:]
        if int(year) < 2017:
            continue

        print(year)
        name_set = set()
        final_rows = []
        for logfile in list(glob.glob(tournament + "/**/*0.log", recursive=True)):
            logfile1 = logfile
            logfile2 = logfile[:-5] + "1.log"
            with open(logfile1) as log1:
                with open(logfile2) as log2:
                    reader1 = csv.reader(log1, delimiter=':')
                    reader2 = csv.reader(log2, delimiter=':')
                    rowCount = 0
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
                        name_set.add(second_names[0])
                        name_set.add(second_names[1])
                        if second_names[0].lower() < second_names[1].lower():
                            score1 = int(float(second_scores[0]))
                            score2 = int(float(first_scores[1]))
                            combined_score = score1 + score2
                            total_score = abs(score1) + abs(score2)
                            if total_score == 0:
                                winrate = 0
                            else:
                                winrate = (combined_score + total_score) / (2 * total_score)
                            final_rows.append([year, second_names[0], second_names[1], combined_score, winrate])
                        else:
                            score1 = int(float(second_scores[1]))
                            score2 = int(float(first_scores[0]))
                            combined_score = score1 + score2
                            total_score = abs(score1) + abs(score2)
                            if total_score == 0:
                                winrate = 0
                            else:
                                winrate = (combined_score + total_score) / (2 * total_score)
                            final_rows.append([year, second_names[1], second_names[0], combined_score, winrate])
                            rowCount+=1
                            #if rowCount > 10:
                            #    break
            #break
        for i, row in enumerate(final_rows):
            row.append(len(name_set))
            writer.writerow(row)
