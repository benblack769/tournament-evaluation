import pandas
import numpy as np


def replace_(s):
    return str(s).replace(" ","_")

def trans_results(res):
    if res == "1-0":
        return 1
    elif res == "0-1":
        return -1
    elif res == "1/2-1/2":
        return 0
    else:
        return float('inf')
        raise RuntimeError("bad result "+res)

def unique_names(l):
    return len(set(l))

def fold_csv_file(in_fname, out_fname):
    with open(in_fname) as file:
        lines = file.readlines()
        header = lines[0]
        lines = lines[1:]
    with open(in_fname,'w') as file:
        file.write("".join([header]+list(sorted(lines))))

    fold_pairs = {}
    cur_event = None
    final_lines = []
    for line in lines:
        event, p1, p2, result = line.split(",")
        result = float(result)
        if cur_event is None or cur_event != event:
            if fold_pairs:
                print(f"transition failed: {event},{cur_event}: {len(fold_pairs)}")
                fold_pairs = {}

        if p2 > p1:
            p1, p2 = p2, p1
            result = -result

        if (p1, p2) in fold_pairs:
            total_result = fold_pairs[(p1,p2)] + result
            final_lines.append(f"{event},{p1},{p2},{total_result}\n")
            del fold_pairs[(p1,p2)]
            # print("found")
        else:
            fold_pairs[(p1,p2)] = result
        cur_event = event

    with open(out_fname,'w') as file:
        file.write("".join([header]+list(sorted(final_lines))))

def main():
    df = pandas.read_csv("chess_data/all_data.csv")
    new_df = pandas.DataFrame({
        'event': df['Event'].transform(replace_),
        'p1': df['White'].transform(replace_),
        'p2': df['Black'].transform(replace_),
        'result': df['Result'].transform(trans_results),
    })
    # remove games with unspecified event
    new_df = new_df[new_df['event'] != ('nan')]

    # clean events with non-decided result
    event_stats = new_df[['event','result']].groupby(['event']).mean()
    bad_events = list(event_stats[np.isinf(event_stats['result'])].index)
    new_df = new_df[~new_df['event'].isin(set(bad_events))]

    # remove tournaments with small numbers of players
    MIN_TOURNAMENT_PLAYERS = 4
    MIN_TOURNAMENT_PAIRINGS = 13
    new_df['players'] = new_df['p1']
    new_df['pairing'] = new_df['p1']+new_df['p2']
    event_stats = new_df[['event','pairing','players']].groupby(['event'])
    uniq_count = event_stats['pairing'].aggregate(unique_names)#.rename({"p2":"uniq"},axis=1)
    uniq_count2 = event_stats['players'].aggregate(unique_names)#.rename({"p2":"uniq"},axis=1)
    del new_df['pairing']
    del new_df['players']
    new_df = pandas.merge(new_df, uniq_count, how='left', on=['event'])#.rename({"p2_x":"p2","p2_y":"num_players"},axis=1)
    new_df = pandas.merge(new_df, uniq_count2, how='left', on=['event'])#.rename({"p2_x":"p2","p2_y":"num_players"},axis=1)
    new_df = new_df[new_df['pairing'] > MIN_TOURNAMENT_PAIRINGS]
    new_df = new_df[new_df['players'] > MIN_TOURNAMENT_PLAYERS]
    new_df = new_df[(new_df['players']*(new_df['players']-1))/2 < new_df['pairing']]
    del new_df['pairing']
    del new_df['players']

    # group identical pairings within each tournament
    # new_df['pairing'] = new_df['p1']+":"+new_df['p2']
    # reord_pairing = new_df['p2']+":"+new_df['p1']
    # new_df['result'] = new_df['result'].where(new_df['p1'] < new_df['p2'], -new_df['result'])
    # new_df['temp'] = new_df['p1'].where(new_df['p1'] < new_df['p2'], new_df['p2'])
    # new_df['p2'] = new_df['p2'].where(new_df['p1'] < new_df['p2'], new_df['p1'])
    # new_df['p1'] = new_df['temp']
    # del new_df['temp']
    new_df.to_csv("clean_data.csv", index=False)
    # fold_csv_file("clean_unfolded_data.csv","clean_data.csv")

    # new_df['temp'] = new_df['p1']
    # new_df['pairing'].where(reord_pairing < new_df['pairing'], reord_pairing, inplace=True)
    agg = new_df.groupby(['event','p1','p2'])
    agg_res = agg['result'].aggregate(num_games=len, result= np.sum)
    # agg_count = agg['result'].aggregate(len)
    # agg_count['count'] = agg_count['result']
    # del agg_count['result']
    # all = pandas.merge(agg_count, agg_res, how='left')
    # agg_res[mes'] = agg['result'].aggregate(len)
    agg_res = agg_res.reset_index()

    # print(new_df[])
    # print(sorted(set(new_df['event'])))
    # print(new_df[new_df['event'] == 'TCEC_Season_17_-_LCZeroCPU_vs_DivP_CPU'])
    agg_res.to_csv("minimatch_data.csv", index=False)

if __name__ == "__main__":
    main()
