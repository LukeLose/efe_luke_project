import random
from typing import Dict, Tuple
import time

from adversarial_search_problem import (
    Action,
    State as GameState,
    HeuristicAdversarialSearchProblem
)
from go_search_problem import (
    GoState
)
from heuristic_go_problems import (
    GoProblemSimpleHeuristic
)

MAXIMIZER = 0
MIMIZER = 1

def minimax(search_problem: HeuristicAdversarialSearchProblem, game_state: GoState, depth: int, cutoff_depth: int):
    best_action = None
    if search_problem.is_terminal_state(game_state):
            return best_action

    if game_state.player_to_move() == MAXIMIZER:
        best_action, _ = max_value_helper(search_problem, game_state, depth, cutoff_depth)
    else:
        best_action, _ = min_value_helper(search_problem, game_state, depth, cutoff_depth)

    return best_action



def max_value_helper(search_problem_input: GoProblemSimpleHeuristic, curr_state: GoState, depth: int, cutoff_depth: int):
    """
    Helper function to find the maximizing the score/action for the current state.
    """
    curr_max = float('-inf')
    best_action = None
    search_problem = search_problem_input

    if search_problem.is_terminal_state(curr_state):
        return None, search_problem.evaluate_terminal(curr_state)

    if depth == cutoff_depth:
        return None, search_problem.heuristic(curr_state, 0)

    for action in search_problem.get_available_actions(curr_state):
        successor_state = search_problem.transition(curr_state, action)
        _, value = min_value_helper(search_problem, successor_state, depth + 1, cutoff_depth)
        if value >= curr_max:
            curr_max = value
            best_action = action

    return best_action, curr_max



def min_value_helper(search_problem_input: GoProblemSimpleHeuristic, curr_state: GoState, depth: int, cutoff_depth: int):
    """
    Helper function to find the minimizing score/action for the current state.
    """
    curr_min = float('inf')
    best_action = None
    search_problem = search_problem_input

    if search_problem.is_terminal_state(curr_state):
        return None, search_problem.evaluate_terminal(curr_state)

    if depth == cutoff_depth:
        return None, search_problem.heuristic(curr_state, 1)

    for action in search_problem.get_available_actions(curr_state):
        successor_state = search_problem.transition(curr_state, action)
        _, value = max_value_helper(search_problem, successor_state, depth + 1, cutoff_depth)
        if value <= curr_min:
            curr_min = value
            best_action = action

    return best_action, curr_min












def alpha_beta(search_problem: HeuristicAdversarialSearchProblem, game_state: GoState, depth: int, cutoff_depth: int):
    best_action = None

    if search_problem.is_terminal_state(game_state):
            return best_action

    if game_state.player_to_move() == MAXIMIZER:
        best_action, _ = max_value_helper_ab(
            search_problem, game_state, depth, cutoff_depth, float('-inf'), float('inf'))
    else:
        best_action, _ = min_value_helper_ab(
            search_problem, game_state, depth, cutoff_depth, float('-inf'), float('inf'))

    return best_action

def max_value_helper_ab(search_problem_input: GoProblemSimpleHeuristic, curr_state: GoState, depth: int, cutoff_depth: int,
                        alpha: float, beta: float):
    """
    Helper function to find the maximizing the score/action for the current state.
    """
    curr_max = float('-inf')
    best_action = None
    search_problem = search_problem_input

    if search_problem.is_terminal_state(curr_state):
        winning_player = search_problem.evaluate_terminal(curr_state)
        if winning_player == 1:
            return None, float('inf')
        if winning_player == -1:
            return None, float('-inf')
        else:
            return None, 0

    if depth == cutoff_depth:
        return None, search_problem.heuristic(curr_state, 0)

    for action in search_problem.get_available_actions(curr_state):
        successor_state = search_problem.transition(curr_state, action)
        _, value = min_value_helper_ab(search_problem, successor_state, depth + 1, cutoff_depth, alpha, beta)
        if value >= curr_max:
            curr_max = value
            best_action = action
        #pruning step
        alpha = max(alpha, curr_max)
        if alpha >= beta:
            break

    return best_action, curr_max

def min_value_helper_ab(search_problem_input: GoProblemSimpleHeuristic, curr_state: GoState, depth: int, cutoff_depth: int,
                        alpha: float, beta: float):
    """
    Helper function to find the minimizing score/action for the current state.
    """
    curr_min = float('inf')
    best_action = None
    search_problem = search_problem_input

    if search_problem.is_terminal_state(curr_state):
        winning_player = search_problem.evaluate_terminal(curr_state)
        if winning_player == 1:
            return None, float('inf')
        if winning_player == -1:
            return None, float('-inf')
        else:
            return None, 0

    if depth == cutoff_depth:
        return None, search_problem.heuristic(curr_state, 1)

    for action in search_problem.get_available_actions(curr_state):
        successor_state = search_problem.transition(curr_state, action)
        _, value = max_value_helper_ab(search_problem, successor_state, depth + 1, cutoff_depth, alpha, beta)
        if value <= curr_min:
            curr_min = value
            best_action = action
        #pruning step
        beta = min(beta, curr_min)
        if beta <= alpha:
            break
    return best_action, curr_min










def iterative_deepening(search_problem: HeuristicAdversarialSearchProblem, game_state: GoState, end_time):
    depth = 1
    best_action = None
    try:
        while time.perf_counter() < end_time:
            if game_state.player_to_move() == MAXIMIZER:
                best_action_depth, _ = max_value_helper_ids(
                    search_problem, game_state, depth, float('-inf'), float('inf'), end_time)
            else:
                best_action_depth, _ = min_value_helper_ids(
                    search_problem, game_state, depth, float('-inf'), float('inf'), end_time)
            if best_action_depth is not None:
                best_action = best_action_depth
                #Once all computation at one depth is done, go to next
            depth += 1
    except TimeoutError:
        pass
    #print("IDS depth of " + str(depth))
    return best_action


def max_value_helper_ids(search_problem_input: GoProblemSimpleHeuristic, curr_state: GoState, depth: int, alpha: float,
                        beta: float, end_time):
    """
    Helper function to find the maximizing the score/action for the current state.
    """
    #check time first!
    if time.perf_counter() >= end_time:
        raise TimeoutError
    curr_max = float('-inf')
    best_action = None
    search_problem = search_problem_input
    if search_problem.is_terminal_state(curr_state):
        winning_player = search_problem.evaluate_terminal(curr_state)
        if winning_player == 1:
            return None, float('inf')
        if winning_player == -1:
            return None, float('-inf')
        else:
            return None, 0
    if depth == 0:
        return None, search_problem.heuristic(curr_state, 0)
        
    for action in search_problem.get_available_actions(curr_state):
        successor_state = search_problem.transition(curr_state, action)
        _, value = min_value_helper_ids(search_problem, successor_state, depth - 1, alpha, beta, end_time)
        if value >= curr_max:
            curr_max = value
            best_action = action
        #pruning step
        alpha = max(alpha, curr_max)
        if alpha >= beta:
            break
    return best_action, curr_max

def min_value_helper_ids(search_problem_input: GoProblemSimpleHeuristic, curr_state: GoState, depth: int, alpha: float,
                        beta: float, end_time):
    """
    Helper function to find the minimizing the score/action for the current state.
    """
    #check time first!
    if time.perf_counter() >= end_time:
        raise TimeoutError
    curr_min = float('inf')
    best_action = None
    search_problem = search_problem_input
    if search_problem.is_terminal_state(curr_state):
        winning_player = search_problem.evaluate_terminal(curr_state)
        if winning_player == 1:
            return None, float('inf')
        if winning_player == -1:
            return None, float('-inf')
        else:
            return None, 0
    if depth == 0:
        return None, search_problem.heuristic(curr_state, 1)
        
    for action in search_problem.get_available_actions(curr_state):
        successor_state = search_problem.transition(curr_state, action)
        _, value = max_value_helper_ids(search_problem, successor_state, depth - 1, alpha, beta, end_time)
        if value <= curr_min:
            curr_min = value
            best_action = action
        #pruning step
        beta = min(beta, curr_min)
        if alpha >= beta:
            break
    return best_action, curr_min
