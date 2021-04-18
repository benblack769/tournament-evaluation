import argparse
import pandas
import io
import json
import sys

def parse_repeated_pgn(lines):
    games = []
    cur_game = None
    for line in lines:
        # print(line[0])
        if line[0:1] == b"[":
            if cur_game is None:
                cur_game = {}
            end = line.rfind(b"]")
            line = line[1:end]
            space = line.find(b" ")
            key = line[:space]
            value = line[space+1:].strip(b'"')
            # print(key,value)
            cur_game[key.decode("latin1")] = value.decode("latin1")
        else:
            if cur_game is not None:
                # print("game over")
                games.append(cur_game)
                cur_game = None

    return games

def games_to_csv(games):
    df = pandas.read_json(io.StringIO(json.dumps(games)))
    print(df.columns)
    print(df["Result"])
    return df

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='parse regular season pgn')
    # parser.add_argument('fname', type=str)
    # parser.add_argument('outfile', type=str)
    #
    # args = parser.parse_args()
    #
    all_games = []
    for path in sys.argv[1:]:
        with open(path, mode='rb') as file:
            lines = file.readlines()
            games = parse_repeated_pgn(lines)
            all_games += games

    df = games_to_csv(all_games)
    df.to_csv("chess_data/all_data.csv", index=False)
