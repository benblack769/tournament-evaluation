from open_spiel.python.egt import alpharank
from open_spiel.python.egt import alpharank_visualizer
import pyspiel
from open_spiel.python.egt import utils
from open_spiel.python.egt import heuristic_payoff_table


game = pyspiel.load_matrix_game("matrix_rps")
payoff_tables = utils.game_payoffs_array(game)

# Convert to heuristic payoff tables
payoff_tables= [heuristic_payoff_table.from_matrix_game(payoff_tables[0]),
                heuristic_payoff_table.from_matrix_game(payoff_tables[1].T)]

payoffs_are_hpt_format = utils.check_payoffs_are_hpt(payoff_tables)

# Check if the game is symmetric (i.e., players have identical strategy sets
# and payoff tables) and return only a single-player’s payoff table if so.
# This ensures Alpha-Rank automatically computes rankings based on the
# single-population dynamics.
_, payoff_tables = utils.is_symmetric_matrix_game(payoff_tables)

# Compute Alpha-Rank
(rhos, rho_m, pi, num_profiles, num_strats_per_population) = alpharank.compute(
    payoff_tables, alpha=1e2)

# Report results
alpharank.print_results(payoff_tables, payoffs_are_hpt_format, pi=pi)
