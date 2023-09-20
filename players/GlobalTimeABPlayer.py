
from numpy.lib.function_base import copy
from players.AbstractPlayer import AbstractPlayer
import numpy as np
import time
from SearchAlgos import AlphaBeta


class Player(AbstractPlayer):
    def __init__(self, game_time, board=None, my_pos=None, rival_pos=None, turn=0):
        AbstractPlayer.__init__(self, game_time)

        self.board = board
        self.my_pos = my_pos
        self.rival_pos = rival_pos
        self.turn = turn

        self.searchAlgo = AlphaBeta(_utility, _succ, _heuristic)

    def set_game_params(self, board):
        self.board = board
        self.my_pos = np.full(9, -1)
        self.rival_pos = np.full(9, -1)
        self.turn = 0

    def make_move(self, time_limit):
        depth = 1
        maximizing_player = True
        if self.turn % 2 == 0:
            even_turn = True
        else:
            even_turn = False

        if self.turn < 18:
            turn_time_limit = time_limit/100
        else:
            turn_time_limit = time_limit/50

        start_time = time.time()
        while time.time() - start_time < turn_time_limit/21:
            max_val, best_move = self.searchAlgo.search(
                self, depth, maximizing_player, even_turn, None)
            if max_val == 10000:
                break
            depth += 1

        if best_move == None:
            rival_dead_pos = -1
            player_new_pos = int(np.where(self.board == 0)[0][0])
            player_soldier_that_moved = int(np.where(self.my_pos == -1)[0][0])
        else:
            player_new_pos, player_soldier_that_moved, rival_dead_pos = best_move

        if rival_dead_pos != -1:
            rival_idx = int(np.where(self.rival_pos == rival_dead_pos)[0][0])
            self.rival_pos[rival_idx] = -2
            self.board[rival_dead_pos] = 0

        if self.turn < 18:
            self.board[player_new_pos] = 1
            self.my_pos[player_soldier_that_moved] = player_new_pos

        else:
            player_prev_pos = self.my_pos[player_soldier_that_moved]
            self.board[player_prev_pos] = 0
            self.board[player_new_pos] = 1
            self.my_pos[player_soldier_that_moved] = player_new_pos

        self.turn += 1
        return player_new_pos, player_soldier_that_moved, rival_dead_pos

    def set_rival_move(self, move):
        rival_pos, rival_soldier, my_dead_pos = move

        if self.turn < 18:
            self.board[rival_pos] = 2
            self.rival_pos[rival_soldier] = rival_pos

        else:
            rival_prev_pos = self.rival_pos[rival_soldier]
            self.board[rival_prev_pos] = 0
            self.board[rival_pos] = 2
            self.rival_pos[rival_soldier] = rival_pos

        if my_dead_pos != -1:
            self.board[my_dead_pos] = 0
            dead_soldier = int(np.where(self.my_pos == my_dead_pos)[0][0])
            self.my_pos[dead_soldier] = -2

        self.turn += 1

    ########## heuristics functions ###################


def _heuristic(state, prev_state):
    if state.turn < 18:
        return _stage_1_heuristic(state, prev_state)
    else:
        return _stage_2_heuristic(state, prev_state)


def _stage_1_heuristic(state, prev_state):
    curr_heuristic = \
        _heuristic_removed_soldiers(state) + \
        _heuristic_movable_soldiers(state) + \
        _heuristic_incompleat_mills(state) * 10 + \
        _heuristic_compleated_mill(state, prev_state) * 100
    return curr_heuristic


def _stage_2_heuristic(state, prev_state):
    curr_heuristic = \
        _heuristic_removed_soldiers(state) + \
        _heuristic_movable_soldiers(state) + \
        _heuristic_incompleat_mills(state) * 10 + \
        _heuristic_compleated_mill(state, prev_state) * 1000
    return curr_heuristic


def _heuristic_removed_soldiers(state):
    player_removed_soldiers = (state.my_pos == -2).sum()
    rival_removed_soldiers = (state.rival_pos == -2).sum()

    return rival_removed_soldiers - player_removed_soldiers


def _heuristic_compleated_mill(state, prev_state):
    if prev_state == None:
        return 0

    curr_player_removed_soldiers = (state.my_pos == -2).sum()
    prev_player_removed_soldiers = (prev_state.my_pos == -2).sum()
    if curr_player_removed_soldiers == 1 + prev_player_removed_soldiers:
        return -1

    curr_rival_removed_soldiers = (state.rival_pos == -2).sum()
    prev_rival_removed_soldiers = (prev_state.rival_pos == -2).sum()
    if curr_rival_removed_soldiers == 1 + prev_rival_removed_soldiers:
        return 1

    return 0


def _heuristic_goal_state(state):
    return _utility(state)


def _heuristic_incompleat_mills(state):
    temp_board = np.copy(state.board)

    player_incompleat_mills = 0
    rival_incompleat_mills = 0

    empty_spaces = np.where(temp_board == 0)[0]
    for empty_space in empty_spaces:
        temp_board[empty_space] = 1
        if state.is_mill(empty_space, temp_board) == True:
            player_incompleat_mills += 1
        temp_board[empty_space] = 0

    reverse_temp_board = np.copy(temp_board)
    reverse_temp_board[reverse_temp_board == 1] = -10
    reverse_temp_board[reverse_temp_board == 2] = 1
    reverse_temp_board[reverse_temp_board == 1] = 2

    empty_spaces = np.where(reverse_temp_board == 0)[0]
    for empty_space in empty_spaces:
        reverse_temp_board[empty_space] = 1
        if state.is_mill(empty_space, reverse_temp_board) == True:
            rival_incompleat_mills += 1
        reverse_temp_board[empty_space] = 0

    return player_incompleat_mills - rival_incompleat_mills


def _heuristic_movable_soldiers(state):
    player_movable_soldiers = _player_movable_soldiers(state)
    rival_movable_soldiers = _rival_movable_soldiers(state)
    return player_movable_soldiers - rival_movable_soldiers


def _player_movable_soldiers(state):
    temp_board = np.copy(state.board)
    player_soldiers_on_board = np.where(temp_board == 1)[0]
    player_movable_soldiers = 0

    for player_soldier_cell in player_soldiers_on_board:
        direction_list = state.directions(int(player_soldier_cell))
        for direction in direction_list:
            if temp_board[direction] == 0:
                player_movable_soldiers += 1
                break

    return player_movable_soldiers


def _rival_movable_soldiers(state):
    temp_board = np.copy(state.board)
    rival_soldiers_on_board = np.where(temp_board == 2)[0]
    rival_movable_soldiers = 0

    for rival_soldier_cell in rival_soldiers_on_board:
        direction_list = state.directions(int(rival_soldier_cell))
        for direction in direction_list:
            if temp_board[direction] == 0:
                rival_movable_soldiers += 1
                break

    return rival_movable_soldiers

    ########## helper functions for Minimax algorithm ##########


def _utility(state):
    # regular victory
    rival_removed_soldiers = (state.rival_pos == -2).sum()
    if rival_removed_soldiers > 6:
        return 1

    # No movement possible victory
    if (state.rival_pos == -1).sum() == 0 and _rival_movable_soldiers(state) == 0:
        return 1

    # regular loss
    player_removed_soldiers = (state.my_pos == -2).sum()
    if player_removed_soldiers > 6:
        return -1

    # No movement possible loss
    if (state.my_pos == -1).sum() == 0 and _player_movable_soldiers(state) == 0:
        return -1

    return 0


def _succ(state, moving_agent):
    # TODO: at the moment game_time = 0 which is not correct
    board = np.copy(state.board)
    player_pos = np.copy(state.my_pos)
    rival_pos = np.copy(state.rival_pos)

    rival_cells = np.where(board == 2)[0]
    player_cells = np.where(board == 1)[0]
    free_cells = np.where(state.board == 0)[0]

    if state.turn < 18:
        if moving_agent == True:
            return _succ_stage1_player_moves(
                state, board, player_pos,
                rival_pos, rival_cells,
                free_cells, game_time=0)
        elif moving_agent == False:
            return _succ_stage1_rival_moves(
                state, board, player_pos,
                rival_pos, player_cells,
                free_cells, game_time=0)

    elif state.turn >= 18:
        if moving_agent == True:
            return _succ_stage2_player_moves(
                state, board, player_pos,
                rival_pos, player_cells,
                rival_cells, game_time=0)
        elif moving_agent == False:
            return _succ_stage2_rival_moves(
                state, board, player_pos,
                rival_pos, player_cells,
                rival_cells, game_time=0)


def _succ_stage1_player_moves(state, board, player_pos, rival_pos, rival_cells, free_cells, game_time=0):
    children_states = []
    moves = []
    soldier_that_moved = int(np.where(player_pos == -1)[0][0])

    for new_cell in free_cells:
        board[new_cell] = 1
        player_pos[soldier_that_moved] = new_cell

        if state.is_mill(new_cell, board) == False:
            child_state = Player(game_time, np.copy(board),
                                 np.copy(player_pos),
                                 np.copy(rival_pos),
                                 state.turn + 1)
            children_states.append(child_state)
            move = (new_cell, soldier_that_moved, -1)
            moves.append(move)

        elif state.is_mill(new_cell, board) == True:
            for enemy_cell in rival_cells:
                board[enemy_cell] = 0
                enemy_soldier = int(
                    np.where(rival_pos == enemy_cell)[0][0])
                rival_pos[enemy_soldier] = -2

                child_state = Player(game_time, np.copy(board),
                                     np.copy(player_pos),
                                     np.copy(rival_pos),
                                     state.turn + 1)
                children_states.append(child_state)
                move = (new_cell, soldier_that_moved, enemy_cell)
                moves.append(move)

                rival_pos[enemy_soldier] = enemy_cell
                board[enemy_cell] = 2

        board[new_cell] = 0
        player_pos[soldier_that_moved] = -1

    return children_states, moves


def _succ_stage1_rival_moves(state, board, player_pos, rival_pos, player_cells, free_cells, game_time=0):
    children_states = []
    moves = []
    soldier_that_moved = int(np.where(rival_pos == -1)[0][0])

    for new_cell in free_cells:
        board[new_cell] = 2
        rival_pos[soldier_that_moved] = new_cell

        if state.is_mill(new_cell, board) == False:
            child_state = Player(game_time, np.copy(board),
                                 np.copy(player_pos),
                                 np.copy(rival_pos),
                                 state.turn + 1)
            children_states.append(child_state)
            move = (new_cell, soldier_that_moved, -1)
            moves.append(move)

        elif state.is_mill(new_cell, board) == True:
            for enemy_cell in player_cells:
                board[enemy_cell] = 0
                enemy_soldier = int(
                    np.where(player_pos == enemy_cell)[0][0])
                player_pos[enemy_soldier] = -2

                child_state = Player(game_time, np.copy(board),
                                     np.copy(player_pos),
                                     np.copy(rival_pos),
                                     state.turn + 1)
                children_states.append(child_state)
                move = (new_cell, soldier_that_moved, enemy_cell)
                moves.append(move)

                player_pos[enemy_soldier] = enemy_cell
                board[enemy_cell] = 1

        board[new_cell] = 0
        rival_pos[soldier_that_moved] = -1

    return children_states, moves


def _succ_stage2_player_moves(state, board, player_pos, rival_pos, player_cells, rival_cells, game_time=0):
    children_states = []
    moves = []

    for soldier_cell in player_cells:
        direction_list = state.directions(int(soldier_cell))
        soldier_that_moved = int(
            np.where(player_pos == soldier_cell)[0][0])

        for new_cell in direction_list:
            if board[new_cell] == 0:
                board[soldier_cell] = 0
                board[new_cell] = 1
                player_pos[soldier_that_moved] = new_cell

                if state.is_mill(new_cell, board) == False:
                    child_state = Player(game_time, np.copy(board),
                                         np.copy(player_pos),
                                         np.copy(rival_pos),
                                         state.turn + 1)
                    children_states.append(child_state)
                    move = (new_cell, soldier_that_moved, -1)
                    moves.append(move)

                elif state.is_mill(new_cell, board) == True:
                    for enemy_cell in rival_cells:
                        board[enemy_cell] = 0
                        enemy_soldier = int(
                            np.where(rival_pos == enemy_cell)[0][0])
                        rival_pos[enemy_soldier] = -2

                        child_state = Player(game_time, np.copy(board),
                                             np.copy(player_pos),
                                             np.copy(rival_pos),
                                             state.turn + 1)
                        children_states.append(child_state)
                        move = (new_cell, soldier_that_moved, enemy_cell)
                        moves.append(move)

                        rival_pos[enemy_soldier] = enemy_cell
                        board[enemy_cell] = 2

                board[soldier_cell] = 1
                board[new_cell] = 0
                player_pos[soldier_that_moved] = soldier_cell

    return children_states, moves


def _succ_stage2_rival_moves(state, board, player_pos, rival_pos, player_cells, rival_cells, game_time=0):
    children_states = []
    moves = []

    for soldier_cell in rival_cells:
        direction_list = state.directions(int(soldier_cell))
        soldier_that_moved = int(
            np.where(rival_pos == soldier_cell)[0][0])

        for new_cell in direction_list:
            if board[new_cell] == 0:
                board[soldier_cell] = 0
                board[new_cell] = 2
                rival_pos[soldier_that_moved] = new_cell

                if state.is_mill(new_cell, board) == False:
                    child_state = Player(game_time, np.copy(board),
                                         np.copy(player_pos),
                                         np.copy(rival_pos),
                                         state.turn + 1)
                    children_states.append(child_state)
                    move = (new_cell, soldier_that_moved, -1)
                    moves.append(move)

                elif state.is_mill(new_cell, board) == True:
                    for enemy_cell in player_cells:
                        board[enemy_cell] = 0
                        enemy_soldier = int(
                            np.where(player_pos == enemy_cell)[0][0])
                        player_pos[enemy_soldier] = -2

                        child_state = Player(game_time, np.copy(board),
                                             np.copy(player_pos),
                                             np.copy(rival_pos),
                                             state.turn + 1)
                        children_states.append(child_state)
                        move = (new_cell, soldier_that_moved, enemy_cell)
                        moves.append(move)

                        player_pos[enemy_soldier] = enemy_cell
                        board[enemy_cell] = 1

                board[soldier_cell] = 2
                board[new_cell] = 0
                rival_pos[soldier_that_moved] = soldier_cell

    return children_states, moves
