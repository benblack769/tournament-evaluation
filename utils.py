import numpy as np
from open_spiel.python.egt import alpharank
# from open_spiel.python.egt import alpharank_visualizer
import pyspiel
from open_spiel.python.egt import utils
from open_spiel.python.egt import heuristic_payoff_table
from open_spiel.python.egt.examples import alpharank_example
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.rcdefaults()

np.set_printoptions(precision=3, suppress=True)


# This requires that neither ranking_a nor ranking_b have two entries with the same rank
def spearman(ranking_a, ranking_b):
    assert len(ranking_a) == len(ranking_b)
    ranking_a = np.asarray(ranking_a)
    ranking_b = np.asarray(ranking_b)
    n = len(ranking_a)

    # Produce array of differences
    diff = ranking_a - ranking_b
    diff = diff**2
    summed = sum(diff)
    return 1 - ((6 * summed) / (n * (n**2 - 1)))


def test_spearman():
    keys = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    ranking_a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ranking_b = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    print(keys)
    print(ranking_a)
    print(ranking_b)
    print(spearman(ranking_a, ranking_b))

    ranking_b.reverse()
    print(keys)
    print(ranking_a)
    print(ranking_b)
    print(spearman(ranking_a, ranking_b))

    ranking_b = [2, 1, 4, 3, 6, 5, 8, 7, 10, 9]
    print(keys)
    print(ranking_a)
    print(ranking_b)
    print(spearman(ranking_a, ranking_b))

    ranking_b.reverse()
    print(keys)
    print(ranking_a)
    print(ranking_b)
    print(spearman(ranking_a, ranking_b))


def alpha_rank(payoff_tables, alpha=1e2):
    # Compute Alpha-Rank
    return alpharank.compute(payoff_tables, alpha=1e2)


def test_alpha_rank():
    game = pyspiel.load_matrix_game("matrix_rps")
    payoff_tables = utils.game_payoffs_array(game) * 2
    print()
    print(payoff_tables)
    print()

    # Convert to heuristic payoff tables
    payoff_tables = [heuristic_payoff_table.from_matrix_game(payoff_tables[0]),
                     heuristic_payoff_table.from_matrix_game(payoff_tables[1].T)]
    print(payoff_tables[0]._payoffs)
    print(payoff_tables[1]._payoffs)

    payoffs_are_hpt_format = utils.check_payoffs_are_hpt(payoff_tables)

    # Check if the game is symmetric (i.e., players have identical strategy sets
    # and payoff tables) and return only a single-player’s payoff table if so.
    # This ensures Alpha-Rank automatically computes rankings based on the
    # single-population dynamics.
    _, payoff_tables = utils.is_symmetric_matrix_game(payoff_tables)

    #print(payoff_tables[0].num_players, payoff_tables[0].num_strategies, payoff_tables[0]._payoffs)
    (rhos, rho_m, pi, num_profiles, num_strats_per_population) = alpha_rank(payoff_tables, alpha=1e2)

    # Report results
    alpharank.print_results(payoff_tables, payoffs_are_hpt_format, pi=pi)

    payoff_tables = alpharank_example.get_kuhn_poker_data(num_players=3)
    #is_symmetric, payoff_tables = utils.is_symmetric_matrix_game(payoff_tables)
    print(payoff_tables[0])
    print(payoff_tables[1])
    print(payoff_tables[2])
    #print(payoff_tables[0].num_players, payoff_tables[0].num_strategies, payoff_tables[0]._payoffs)
    #print(payoff_tables[1].num_players, payoff_tables[1].num_strategies, payoff_tables[1]._payoffs)
    #print(payoff_tables[2].num_players, payoff_tables[2].num_strategies, payoff_tables[2]._payoffs)
    print(alpharank.sweep_pi_vs_alpha(payoff_tables, visualize=False))
    #alpharank.compute_and_report_alpharank(payoff_tables, alpha=1e2)


class MaxEntropyNash():

    def __init__(self, payoffs, action_space, action_to_index):
        self.payoffs = payoffs
        self.action_space = action_space
        self.action_to_index = action_to_index

        def generate_dual(act_space):
            # We will generate our dual variables.. intialize them to 0 and attempt to constuct
            # a mixed strategy from it
            dual_variables = {}
            joint = []
            for i, action_a in enumerate(act_space):
                for j, action_b in enumerate(act_space):
                    if j != i:
                        # Since this is a meta_game, both players cannot select the same strategy
                        # (may not apply to all games)
                        joint_action = (action_a, action_b)
                        joint.append(joint_action)
                        dual_variables[joint_action] = 0
            return dual_variables, joint
        self.dual_variables, self.joint = generate_dual(action_space)

        self.payoff_map = {}
        for i, row in enumerate(payoffs):
            for j, pay in enumerate(row):
                self.payoff_map[(action_space[i], action_space[j])] = pay

        self.payoff_gains = {}
        for i, action_a in enumerate(self.action_space):
            for j, action_b in enumerate(self.action_space):
                self.payoff_gains[(action_a, action_b)] = self.payoff_gain(action_a, action_b)

        self.log_grad_descent(self.dual_variables, self.joint, verbose=False, rounds=100)
        probs = {}
        for action in action_space:
            probs[action] = self.P(self.dual_variables, action)
        probs = dict(reversed(sorted(probs.items(), key=lambda item: item[1])))
        for action, prob in probs.items():
            print(action, ": ", prob)
        self.print()

    def payoff(self, action_1, action_2):
        return self.payoff_map[(action_1, action_2)]

    def payoff_gain(self, alt_action, action, maximum=False, positive=True):
        # Calculate M(alt_action, action') & M(action, action')
        diff = 0
        for action_prime in self.action_space:
            if action_prime is not action:
                payoffs_alt = self.payoff_map[(alt_action, action_prime)]
                payoffs_act = self.payoff_map[(action, action_prime)]
                if maximum:
                    if positive:
                        diff += max(0, payoffs_alt - payoffs_act)
                    else:
                        diff += max(0, -(payoffs_alt - payoffs_act))
                else:
                    diff += payoffs_alt - payoffs_act

        return diff

    # Return Z(lambda) of dual variables
    def Z(self, dual_vars):
        sum_one = 0
        for i, action_a in enumerate(self.action_space):
            sum_two = 0
            for j, action_b in enumerate(self.action_space):
                if j != i:
                    sum_two += dual_vars[(action_a, action_b)] * self.payoff_gains[(action_b, action_a)]
            sum_one += np.exp(-sum_two)
        return sum_one

    # Get mixed strategy from dual variables
    def P(self, dual_vars, a):
        sum_one = 0
        for action in self.action_space:
            if action != a:
                sum_one += dual_vars[(a, action)]*self.payoff_gains[(action, a)]

        log_p = -sum_one - np.log(self.Z(dual_vars))
        return np.exp(log_p)

    def regret_pos(self, dual_vars, action, action_prime):
        p_a = self.P(dual_vars, action)
        final_sum = 0
        for p2_action in self.action_space:
            p_p2 = self.P(dual_vars, p2_action)
            p_gain = self.payoff_map[(action_prime, p2_action)] - self.payoff_map[(action, p2_action)]

            final_sum += (p_a*p_p2) * max(0, p_gain)
        return final_sum

    def regret_neg(self, dual_vars, action, action_prime):
        p_a = self.P(dual_vars, action)
        final_sum = 0
        for p2_action in self.action_space:
            p_p2 = self.P(dual_vars, p2_action)
            p_gain = self.payoff_map[(action_prime, p2_action)] - self.payoff_map[(action, p2_action)]
            final_sum += (p_a*p_p2) * max(0, -p_gain)
        return final_sum

    def regret_both(self, dual_vars, action, action_prime):
        p_a = self.P(dual_vars, action)
        pos = neg = 0
        for p2_action in self.action_space:
            p_p2 = self.P(dual_vars, p2_action)
            p_gain = self.payoff_map[(action_prime, p2_action)] - self.payoff_map[(action, p2_action)]

            pos += (p_a*p_p2) * max(0, p_gain)
            neg += (p_a*p_p2) * max(0, -p_gain)
        return pos, neg

    def abs_gain(self, action, action_prime):
        total = 0
        for p2_action in self.action_space:
            p_gain = abs(self.payoff_map[(action_prime, p2_action)] - self.payoff_map[(action, p2_action)])
            total += p_gain
        return total

    def plot(self):
        def sort_dictionary(dictionary):
            sorted_x = sorted(dictionary.items(), key=lambda kv: kv[1])
            return sorted_x

        probs = {}
        for action in self.action_space:
            probs[action] = self.P(self.dual_variables, action)

        sort = sort_dictionary(probs)
        # For tier lists
        objects = []
        performance = []
        for player, value in sort:
            objects.append(player)
            performance.append(value)

        y_pos = np.arange(len(objects))
        plt.barh(y_pos, performance, align="center")
        plt.yticks(y_pos, objects)
        plt.xlabel('Density')
        plt.ylabel('Strategies')
        plt.title('Maximum Entropy Nash Distribution')
        plt.show()

    def print(self):
        print(self.dual_variables)

    def log_grad_descent(self, dual_vars, space, rounds=10, verbose=True):
        def lower_bound_c():
            bound = 0
            for action in self.action_space:
                for action_prime in self.action_space:
                    a_gain = self.abs_gain(action, action_prime)
                    bound += a_gain
            return bound

        c = lower_bound_c()

        for it in tqdm(range(rounds)):
            step_dict = {}
            for pair in space:
                """
                r_pos = regret_pos(dv, action, action_prime)
                r_neg = regret_neg(dv, action, action_prime)
                """

                a1, a2 = pair
                r_pos, r_neg = self.regret_both(dual_vars, a1, a2)

                term = ((r_pos)/(r_pos + r_neg)) - (1/2)
                step = (1/c)*term
                step_dict[pair] = step

            for pair in self.joint:
                dual_vars[pair] = max(0, dual_vars[pair] + step_dict[pair])

            if verbose:
                print("Iteration ", it)
                self.plot(dual_vars)


def test_maxent_nash():
    # This will be our meta_game action space
    action_space = []
    file = 'F19PRUlt.csv'
    # We will survey the csv file to fill our action space
    with open(file, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            if row['Player1'] not in action_space:
                action_space.append(row['Player1'])

            if row['Player2'] not in action_space:
                action_space.append(row['Player2'])

    # Create action_to_index utility dictionary
    index = 0
    action_to_index = {}
    for action in action_space:
        action_to_index[action] = index
        index += 1

    with open(file, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        # Create two dictionaries to keep track of wins and losses over matches
        matches = []
        pay = {}

        for row in csv_reader:
            i = row["Player1"]
            j = row["Player2"]
            match = (i, j)
            matchPrime = (j, i)
            p = float(row["Pay"])
            if (i in action_space) and (j in action_space):
                if (match in matches) or (matchPrime in matches):
                    print("in match")
                    if match in matches:
                        pay[match] += p
                    else:
                        pay[matchPrime] += p
                else:
                    matches.append(match)
                    pay[match] = p

    # Create payoff matrix
    M = np.full((len(action_space), len(action_space)), 0.0)
    # matchup = [[None]*len(action_space) for i in range(len(action_space))]
    for match, p in pay.items():
        name_i, name_j = match

        M[action_to_index[name_i]][action_to_index[name_j]] = p
        M[action_to_index[name_j]][action_to_index[name_i]] = -p
        # matchup[action_to_index[name_i]][action_to_index[name_j]] = (name_i, name_j)
        # matchup[action_to_index[name_j]][action_to_index[name_i]] = (name_j, name_i)

    print(M)
    maxent_nash = MaxEntropyNash(M, action_space, action_to_index)
    #maxent_nash.plot()
    #maxent_nash.print()

from run_experiment import compute_score, compute_normalized_score, compute_win_rate, compute_score_nash
from run_experiment import print_ranking
import numpy as np
import pandas

np.set_printoptions(linewidth=200)

def compute_nash():
    def eval_payoff(score_fn):
        df = pandas.read_csv("payoffs2.csv", index_col=0)
    
        matrix = df.reset_index(drop=True)
        print(matrix.columns.values)
        matrix = matrix.values
        print(matrix)
        scores = score_fn(np.triu(matrix), np.ones_like(matrix))
        print_ranking(scores, df.index)
    
    eval_payoff(compute_score_nash)

def poker_payoffs():
    action_space = []
    action_to_index = {}
    M = []
    poker_file = "payoffs_clipped.csv"
    with open(poker_file, mode="r") as csv_file:
        csv_reader = csv.reader(csv_file)
        first_row = True
        for row in csv_reader:
            if first_row:
                first_row = False
                action_space = row[1:]
                M = np.zeros((len(action_space), len(action_space)))
                index = 0
                for action in action_space:
                    action_to_index[action] = index
                    index += 1
            else:
                name = row[0]
                for i, pay in enumerate(row[1:]):
                    M[i, action_to_index[name]] = pay

    M = -M
    print(M)
    maxent_nash = MaxEntropyNash(M, action_space, action_to_index)
    #alpha = alpharank.compute(M, alpha=1e2)


#print("SPEARMAN")
#test_spearman()
#print("\n\n\nALPHARANK")
#test_alpha_rank()
#print("\n\n\nMAXENT NASH")
#test_maxent_nash()

#import cProfile, pstats, io
#from pstats import SortKey
#pr = cProfile.Profile()
#pr.enable()
## ... do something ...
#poker_payoffs()
#pr.disable()
#s = io.StringIO()
#sortby = SortKey.CUMULATIVE
#ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
#ps.print_stats()
#print(s.getvalue())

#poker_payoffs()
#compute_nash()
