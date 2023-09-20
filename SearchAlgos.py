"""Search Algos: MiniMax, AlphaBeta
"""
# TODO: you can import more modules, if needed
import time
import numpy as np
import utils
ALPHA_VALUE_INIT = -np.inf
BETA_VALUE_INIT = np.inf


class SearchAlgos:
    def __init__(self, utility, succ, heuristic, perform_move=None, goal=None):
        """The constructor for all the search algos.
        You can code these functions as you like to, 
        and use them in MiniMax and AlphaBeta algos as learned in class
        :param utility: The utility function.
        :param succ: The succesor function.
        :param perform_move: The perform move function.
        :param goal: function that check if you are in a goal state.
        """
        self.utility = utility
        self.succ = succ
        self.perform_move = perform_move
        self.goal = goal
        self.heuristic = heuristic

    def search(self, state, depth, maximizing_player, even_turn, prev_state):
        pass


class MiniMax(SearchAlgos):

    def search(self, state, depth, maximizing_player, even_turn, prev_state):
        if self.utility(state) != 0:
            return self.utility(state) * 10000, None

        if depth == 0:
            if state.turn == 0:
                return 0, None
            else:
                return self.heuristic(state, prev_state), None

        if (even_turn == True and state.turn % 2 == 0) or \
                (even_turn == False and state.turn % 2 == 1):
            moving_agent = True
        else:
            moving_agent = False

        children, moves = self.succ(state, moving_agent)

        if maximizing_player == moving_agent:
            curr_max = -np.inf
            for i, child_state in enumerate(children):
                value, _ = self.search(child_state, depth - 1,
                                       maximizing_player, even_turn, state)
                if value > curr_max:
                    curr_max = value
                    max_move = moves[i]
            return curr_max, max_move

        elif maximizing_player != moving_agent:
            curr_min = np.inf
            for i, child_state in enumerate(children):
                value, _ = self.search(child_state, depth - 1,
                                       maximizing_player, even_turn, state)
                if value < curr_min:
                    curr_min = value
                    min_move = moves[i]
            return curr_min, min_move


class AlphaBeta(SearchAlgos):

    def search(self, state, depth, maximizing_player, even_turn, prev_state,
               alpha=ALPHA_VALUE_INIT, beta=BETA_VALUE_INIT):
        if self.utility(state) != 0:
            return self.utility(state) * 10000, None

        if depth == 0:
            if state.turn == 0:
                return 0, None
            else:
                return self.heuristic(state, prev_state), None

        if (even_turn == True and state.turn % 2 == 0) or \
                (even_turn == False and state.turn % 2 == 1):
            moving_agent = True
        else:
            moving_agent = False

        children, moves = self.succ(state, moving_agent)

        if maximizing_player == moving_agent:
            curr_max = -np.inf
            for i, child_state in enumerate(children):
                value, _ = self.search(child_state, depth - 1,
                                       maximizing_player, even_turn,
                                       state, alpha, beta)
                if value > curr_max:
                    curr_max = value
                    max_move = moves[i]
                if curr_max >= beta:
                    return curr_max, max_move
                alpha = max(alpha, curr_max)
            return curr_max, max_move

        elif maximizing_player != moving_agent:
            curr_min = np.inf
            for i, child_state in enumerate(children):
                value, _ = self.search(child_state, depth - 1,
                                       maximizing_player, even_turn,
                                       state, alpha, beta)
                if value < curr_min:
                    curr_min = value
                    min_move = moves[i]
                if curr_min <= alpha:
                    return curr_min, min_move
                beta = min(beta, curr_min)
            return curr_min, min_move
