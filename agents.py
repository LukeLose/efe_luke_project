from go_search_problem import GoProblem, GoState, Action
from adversarial_search_problem import GameState
from heuristic_go_problems import *
import random
from abc import ABC, abstractmethod
import numpy as np
import time
#from game_runner import run_many
import pickle
import torch
from torch import nn
from adversarial_search import (
    minimax,
    alpha_beta,
    iterative_deepening
)


MAXIMIZER = 0
MIMIZER = 1



# def create_value_agent_from_model():
#     """
#     Create agent object from saved model. This (or other methods like this) will be how your agents will be created in gradescope and in the final tournament.
#     """

#     model_path = "value_model.pt"
#     # TODO: Update number of features for your own encoding size
#     feature_size = 54 #UPDATE THIS NUMBER WITH THE NUMBER FROM YOUR NOTEBOOK
#     #feature_size = 100 #UPDATE THIS NUMBER WITH THE NUMBER FROM YOUR NOTEBOOK
#     model = load_model(model_path, ValueNetwork(feature_size))
#     heuristic_search_problem = GoProblemLearnedHeuristic(model)

#     learned_agent = GreedyAgent(heuristic_search_problem) # MAKE SURE THIS IS GREEDY

#     return learned_agent

class GameAgent():
    # Interface for Game agents
    @abstractmethod
    def get_move(self, game_state: GameState, time_limit: float) -> Action:
        # Given a state and time limit, return an action
        pass


class RandomAgent(GameAgent):
    # An Agent that makes random moves

    def __init__(self):
        self.search_problem = GoProblem()

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        get random move for a given state
        """
        actions = self.search_problem.get_available_actions(game_state)
        return random.choice(actions)

    def __str__(self):
        return "RandomAgent"


class GreedyAgent(GameAgent):
    def __init__(self, search_problem=GoProblemSimpleHeuristic()):
        super().__init__()
        self.search_problem = search_problem

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        get move of agent for given game state.
        Greedy agent looks one step ahead with the provided heuristic and chooses the best available action
        (Greedy agent does not consider remaining time)

        Args:
            game_state (GameState): current game state
            time_limit (float): time limit for agent to return a move
        """
        # Create new GoSearchProblem with provided heuristic
        search_problem = self.search_problem
        end_time = time.perf_counter() + time_limit - 0.00005

        # Player 0 is maximizing
        if game_state.player_to_move() == MAXIMIZER:
            best_value = -float('inf')
        else:
            best_value = float('inf')
        best_action = 25

        # Get Available actions
        actions = search_problem.get_available_actions(game_state)

        # Compare heuristic of every reachable next state
        for action in actions:
            if time.perf_counter() <= end_time:
                new_state = search_problem.transition(game_state, action)
                value = search_problem.heuristic(new_state, new_state.player_to_move())
                if game_state.player_to_move() == MAXIMIZER:
                    if value > best_value:
                        best_value = value
                        best_action = action
                else:
                    if value < best_value:
                        best_value = value
                        best_action = action

        # Return best available action
        return best_action

    def __str__(self):
        """
        Description of agent (Greedy + heuristic/search problem used)
        """
        return "GreedyAgent + " + str(self.search_problem)
    



























class MinimaxAgent(GameAgent):
    def __init__(self, depth=1, search_problem=GoProblemSimpleHeuristic()):
        super().__init__()
        self.depth = depth
        self.search_problem = search_problem

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        Get move of agent for given game state using minimax algorithm

        Args:
            game_state (GameState): current game state
            time_limit (float): time limit for agent to return a move
        Returns:
            best_action (Action): best action for current game state
        """
        # TODO: implement get_move method of MinimaxAgent
        depth = 0
        search_problem = self.search_problem
        cutoff_depth = self.depth

        return minimax(search_problem, game_state, depth, cutoff_depth)

    def __str__(self):
        return f"MinimaxAgent w/ depth {self.depth} + " + str(self.search_problem)
    































class AlphaBetaAgent(GameAgent):
    def __init__(self, depth=1, search_problem=GoProblemSimpleHeuristic()):
        super().__init__()
        self.depth = depth
        self.search_problem = search_problem

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        Get move of agent for given game state using alpha-beta algorithm

        Args:
            game_state (GameState): current game state
            time_limit (float): time limit for agent to return a move
        Returns:
            best_action (Action): best action for current game state
        """
        # TODO: implement get_move algorithm of AlphaBeta Agent
        depth = 0
        search_problem = self.search_problem
        cutoff_depth = self.depth
        return alpha_beta(search_problem, game_state, depth, cutoff_depth)

    def __str__(self):
        return f"AlphaBeta w/ depth {self.depth} + " + str(self.search_problem)
    
























class IterativeDeepeningAgent(GameAgent):
    def __init__(self, cutoff_time=1, search_problem=GoProblemSimpleHeuristic()):
        super().__init__()
        self.cutoff_time = cutoff_time
        self.search_problem = search_problem

    def get_move(self, game_state, time_limit):
        """
        Get move of agent for given game state using iterative deepening algorithm (+ alpha-beta).
        Iterative deepening is a search algorithm that repeatedly searches for a solution to a problem,
        increasing the depth of the search with each iteration.

        The advantage of iterative deepening is that you can stop the search based on the time limit, rather than depth.
        The recommended approach is to modify your implementation of Alpha-beta to stop when the time limit is reached
        and run IDS on that modified version.

        Args:
            game_state (GameState): current game state
            time_limit (float): time limit for agent to return a move
        Returns:
            best_action (Action): best action for current game state
        """
        # TODO: implement get_move algorithm of IterativeDeepeningAgent
        search_problem = self.search_problem
        start_time = time.perf_counter()
        end_time = start_time + time_limit - 0.1
        return iterative_deepening(search_problem, game_state, end_time)

    def __str__(self):
        return f"IterativeDeepneing + " + str(self.search_problem)
    



























class MCTSNode:
    def __init__(self, state, parent=None, children=None, action=None):
        # GameState for Node
        self.state = state

        # Parent (MCTSNode)
        self.parent = parent
        
        # Children List of MCTSNodes
        if children is None:
            children = []
        self.children = children
        
        # Number of times this node has been visited in tree search
        self.visits = 0
        
        # Value of node (number of times simulations from children results in black win)
        self.value = 0
        
        # Action that led to this node
        self.action = action

    def __hash__(self):
        """
        Hash function for MCTSNode is hash of state
        """
        return hash(self.state)


class MCTSAgent(GameAgent):
    def __init__(self, c=np.sqrt(2)):
        """
        Args: 
            c (float): exploration constant of UCT algorithm
        """
        super().__init__()
        self.c = c

        # Initialize Search problem
        self.search_problem = GoProblem()

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        Get move of agent for given game state using MCTS algorithm
        
        Args:
            game_state (GameState): current game state
            time_limit (float): time limit for agent to return a move
        Returns:
            best_action (Action): best action for current game state
        """
        # TODO: Implement MCTS
        node = MCTSNode(game_state)
        end_time = time.perf_counter() + time_limit - 0.5

        while time.perf_counter() < end_time:
            leaf = self.selection(node)
            children = self.expand(leaf)
            results = self.simulate(children)
            self.backpropagate(results, children)
        
        #Find the best child now
        best_child = None
        most_visits = -1
        for child in node.children:
            if child.visits > most_visits:
                most_visits = child.visits
                best_child = child
        return best_child.action



    def utc_policy_calc(self, node: MCTSNode) -> float:
        #Unvisited nodes get selected immediately
        if node.visits == 0:
            return float('inf')
        #Calc values of both UTC sections
        value_estimate = node.value / node.visits
        exploration_term = self.c * np.sqrt((np.log(node.parent.visits)) / node.visits)
        return value_estimate + exploration_term
    

    
    def selection(self, node: MCTSNode) -> MCTSNode:
        best_val = float('-inf')
        if not node.children:
            return node
        best_child = None
        for child in node.children:
            uct_value = self.utc_policy_calc(child)
            if uct_value > best_val:
                best_val = uct_value
                best_child = child
        return self.selection(best_child)
    

    def expand(self, node: MCTSNode):
        children = []
        curr_state = node.state
        actions = self.search_problem.get_available_actions(curr_state)
        for action in actions:
            child_state = self.search_problem.transition(curr_state, action)
            child_node = MCTSNode(child_state, node, None, action)
            children.append(child_node)
            #Not sure if i need this or not
            node.children.append(child_node)
        return children


    def simulate(self, children: list[MCTSNode]):
        results = []
        search_problem = self.search_problem
        for child in children:
            child_state = child.state
            while not search_problem.is_terminal_state(child_state):
                actions = search_problem.get_available_actions(child_state)
                action = np.random.choice(actions)
                child_state = search_problem.transition(child_state, action)
            results.append(search_problem.evaluate_terminal(child_state))
        return results
    
    
    def backpropagate(self, results: list[float], children: list[MCTSNode]):
        for child, result in zip(children, results):
            curr_node = child
            while curr_node is not None:
                player_to_move = curr_node.state.player_to_move()
                curr_node.visits += 1
                if result == -1 and player_to_move == 0:
                    curr_node.value = curr_node.value + 1
                elif result == 1 and player_to_move == 1:
                    curr_node.value = curr_node.value + 1

                curr_node = curr_node.parent


    def __str__(self):
        return "MCTS"
    




























def load_model(path: str, model):
    """
    Load model from file

    Note: you still need to provide a model (with the same architecture as the saved model))

    Input:
        path: path to load model from
        model: Pytorch model to load
    Output:
        model: Pytorch model loaded from file
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

class ValueNetwork(nn.Module):
    def __init__(self, input_size):
      super(ValueNetwork, self).__init__()

      # TODO: What should the output size of a Value function be?
      output_size = 1

      # TODO: Add more layers, non-linear functions, etc.
      self.linear = nn.Linear(input_size, 16)
      self.linear_one = nn.Linear(16, 24)
      self.linear_two = nn.Linear(24, 8)
      self.linear_three = nn.Linear(8, output_size)

      self.sigmoid = nn.Sigmoid()
      self.tanh = nn.Tanh()
      self.relu = nn.ReLU()

    def forward(self, x):
      """
      Run forward pass of network

      Input:
        x: input to network
      Output:
        output of network
      """
      # TODO: Update as more layers are added
      layer_weight_one = self.linear(x)
      activations_one = self.tanh(layer_weight_one)
      layer_weight_two = self.linear_one(activations_one)
      layer_weight_three = self.linear_two(layer_weight_two)
      activations_three = self.tanh(layer_weight_three)
      layer_weight_four = self.linear_three(activations_three)
      return layer_weight_four

class GoProblemLearnedHeuristic(GoProblem):
    def __init__(self, model=None, state=None):
        super().__init__(state=state)
        self.model = model
        
    def __call__(self, model=None):
        """
        Use the model to compute a heuristic value for a given state.
        """
        return self

    def encoding(self, state):
        """
        Get encoding of state (convert state to features)
        Note, this may call get_features() from Task 1. 

        Input:
            state: GoState to encode into a fixed size list of features
        Output:
            features: list of features
        """
        # TODO: get encoding of state (convert state to features)
        features = []
        features = get_features(state)


        return features

    def heuristic(self, state, player_index):
        """
        Return heuristic (value) of current state

        Input:
            state: GoState to encode into a fixed size list of features
            player_index: index of player to evaluate heuristic for
        Output:
            value: heuristic (value) of current state
        """
        # TODO: Compute heuristic (value) of current state
        value = 0
        features = self.encoding(state)
        tensor_features = torch.tensor(features, dtype=torch.float32) 
        prediction_score = self.model(tensor_features)
        #prediction = torch.sigmoid(prediction_score)
        value = prediction_score.item()

        # Note, your agent may perform better if you force it not to pass
        # (i.e., don't select action #25 on a 5x5 board unless necessary)
        return value

    def __str__(self) -> str:
        return "Learned Heuristic"

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, board_size=5):
      super(PolicyNetwork, self).__init__()

      # TODO: What should the output size of the Policy be?
      output_size = 1 + board_size * board_size

      # TODO: Add more layers, non-linear functions, etc.

      self.linear = nn.Linear(input_size, 128)
      self.linear_one = nn.Linear(128, 64)
      self.linear_two = nn.Linear(64, 32)
      self.linear_three = nn.Linear(32, output_size)

      self.sigmoid = nn.Sigmoid()
      self.tanh = nn.Tanh()
      self.relu = nn.ReLU()

    def forward(self, x):
      layer_weight_one = self.linear(x)
      activations_one = self.relu(layer_weight_one)
      layer_weight_two = self.linear_one(activations_one)
      activations_two = self.relu(layer_weight_two)
      layer_weight_three = self.linear_two(activations_two)
      activations_three = self.relu(layer_weight_three)
      layer_weight_four = self.linear_three(activations_three)
      return layer_weight_four
  
def get_features(game_state: GoState):
    # """
    # Map a game state to a list of features.

    # Some useful functions from game_state include:
    #     game_state.size: size of the board
    #     get_pieces_coordinates(player_index): get coordinates of all pieces of a player (0 or 1)
    #     get_pieces_array(player_index): get a 2D array of pieces of a player (0 or 1)
        
    #     get_board(): get a 2D array of the board with 4 channels (player 0, player 1, empty, and player to move). 4 channels means the array will be of size 4 x n x n
    
    #     Descriptions of these methods can be found in the GoState

    # Input:
    #     game_state: GoState to encode into a fixed size list of features
    # Output:
    #     features: list of features
    # """
    # board_size = game_state.size
    
    # # TODO: Encode game_state into a list of features
    features = []
    #piece_coords_0 = game_state.get_pieces_coordinates(0)
    #pieces_array_0 = game_state.get_pieces_array(0)
    board = game_state.get_board()
    #print("coords now")
    #print(piece_coords_0)
    #print("array now")
    #print(pieces_array_0)
    black_board = board[0]
    white_board = board[1]
    player_to_go = board[3][0][0]
    

    black_positions_array = []
    white_positions_array = []
    
    #1-25 feature showing black pieces on the board
    for row in black_board:
        for possible_black_piece in row:
            if possible_black_piece == 1:
                black_positions_array.append(1)
            else:
                black_positions_array.append(0)

    #26-50 feature showing white pieces on the board
    for row in white_board:
        for possible_white_piece in row:
            if possible_white_piece == 1:
                white_positions_array.append(1)
            else:
                white_positions_array.append(0)

    #putting the individual arrays on the larger array
    features = black_positions_array + white_positions_array

    #51-57 feature showing player to move

    #Changed to just 51
    if player_to_go == 0:
        features.append(0)
        # features.append(0)
        # features.append(0)
        # features.append(0)
        # features.append(0)
        # features.append(0)
        # features.append(0)
    else:
        features.append(1)
        # features.append(1)
        # features.append(1)
        # features.append(1)
        # features.append(1)
        # features.append(1)
        # features.append(1)

    #58-59 showing number of black and white pieces
    #Changed to just 52-53
    black_piece_amount = sum(1 for x in black_positions_array if x != 0)
    features.append(black_piece_amount)
    #print(black_piece_amount)
    white_piece_amount = sum(1 for x in white_positions_array if x != 0)
    features.append(white_piece_amount)
    #print(white_piece_amount)

    #60-65 showing number of possible moves for player to move
    #Changed to 54
    player_to_go_possible_moves = len(game_state.legal_actions())
    features.append(player_to_go_possible_moves)

    return features

class PolicyAgent(GameAgent):
    def __init__(self, search_problem, model_path, board_size=5):
        super().__init__()
        self.search_problem = search_problem
        #self.model = load_model(model_path, PolicyNetwork(60, 5))
        self.model = load_model(model_path, PolicyNetwork(54, 5))
        #self.model = load_model(model_path, PolicyNetwork(100, 5))
        self.board_size = board_size

    def encoding(self, state):
        # TODO: get encoding of state (convert state to features)
        features = get_features(state)
        return features

    def get_move(self, game_state, time_limit=1):
        """
        Get best action for current state using self.model

        Input:
            game_state: current state of the game
            time_limit: time limit for search (This won't be used in this agent)
        Output:
            action: best action to take
        """

        # TODO: Select LEGAL Best Action predicted by model
        # The top prediction of your model may not be a legal move!
        features = self.encoding(game_state)
        tensor_features = torch.tensor(features, dtype=torch.float32)
        action_probabilities = self.model(tensor_features)

        predicted_action = torch.argmax(action_probabilities).item()
        worst_action = torch.min(action_probabilities).item()
        action_probabilities[25] = worst_action - 1
        while predicted_action not in self.search_problem.get_available_actions(game_state):
            action_probabilities[predicted_action] -= 2
            predicted_action = torch.argmax(action_probabilities).item()

        # predicted_action = torch.argmin(action_probabilities).item()
        # worst_action = torch.max(action_probabilities).item()
        # action_probabilities[25] = worst_action + 1
        # while predicted_action not in self.search_problem.get_available_actions(game_state):
        #     action_probabilities[predicted_action] += 2
        #     predicted_action = torch.argmin(action_probabilities).item()

        assert predicted_action in self.search_problem.get_available_actions(game_state)
        return predicted_action

    def __str__(self) -> str:
        return "Policy Agent"
    










    



class final_agent_5x5(GameAgent):
    def __init__(self, c=np.sqrt(2)):
        """
        Args: 
            c (float): exploration constant of UCT algorithm
        """
        super().__init__()
        self.c = c
        self.c_initial = c
        self.decay_rate = 0.001
        self.c_min = 1.3
        self.model_path = "value_model.pt"
        self.feature_size = 54
        self.model = load_model(self.model_path, ValueNetwork(self.feature_size))
        self.learned_heuristic = GoProblemLearnedHeuristic(self.model)
        #self.move_number = 0 may use this for counting when the midgame starts.

        # Initialize Search problem
        self.search_problem = GoProblem()
        self.long_move_count = 0

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        # TODO: Implement MCTS

        #Get piece amount to guide what model we want to use
        pieces_1 = len(game_state.get_pieces_coordinates(0))
        pieces_2 = len(game_state.get_pieces_coordinates(1))
        curr_piece_num = pieces_1 + pieces_2

        if (curr_piece_num <= 20 and curr_piece_num >= 15 and self.long_move_count < 14):
            #time_limit = 1.99 
            time_per_move = max((time_limit/2) - 0.01, 0.99)
            self.long_move_count += 1
        else:
            time_per_move = 1 #just in case

        if (curr_piece_num >= 18):
            ids_agent = IterativeDeepeningAgent(1, self.learned_heuristic)
            action = ids_agent.get_move(game_state, 1)
            print("using IDS")
            return action
        node = MCTSNode(game_state)
        end_time = time.perf_counter() + time_per_move - 0.5
        iteration = 0 

        while time.perf_counter() < end_time:
            iteration += 1 
            leaf = self.selection(node)
            children = self.expand(leaf)
            results = self.simulate(children)
            self.backpropagate(results, children)
            self.c = max(self.c_initial - self.decay_rate * iteration, self.c_min)
        
        #Find the best child now
        best_child = None
        most_visits = -1
        for child in node.children:
            if child.visits > most_visits:
                most_visits = child.visits
                best_child = child     
        return best_child.action



    def utc_policy_calc(self, node: MCTSNode) -> float:
        #Unvisited nodes get selected immediately
        if node.visits == 0:
            return float('inf')
        #Calc values of both UTC sections
        value_estimate = node.value / node.visits
        exploration_term = self.c * np.sqrt((np.log(node.parent.visits)) / node.visits)
        return value_estimate + exploration_term
    

    
    def selection(self, node: MCTSNode) -> MCTSNode:
        best_val = float('-inf')
        if not node.children:
            return node
        best_child = None
        for child in node.children:
            uct_value = self.utc_policy_calc(child)
            if uct_value > best_val:
                best_val = uct_value
                best_child = child
        return self.selection(best_child)
    

    def expand(self, node: MCTSNode):
        children = []
        curr_state = node.state
        actions = self.search_problem.get_available_actions(curr_state)
        for action in actions:
            child_state = self.search_problem.transition(curr_state, action)
            child_node = MCTSNode(child_state, node, None, action)
            children.append(child_node)
            #Not sure if i need this or not
            node.children.append(child_node)
        return children


    def simulate(self, children: list[MCTSNode]):
        #This is what we need to fix, by making it not random choices but good choices!
        results = []
        search_problem = self.search_problem
        greedy_learn = GreedyAgent(self.learned_heuristic)
        for child in children:
            child_state = child.state
            move_counter = 0
            player_to_move = child_state.player_to_move()
            while not search_problem.is_terminal_state(child_state):
                if move_counter > 50:
                    results.append(self.learned_heuristic.heuristic(child_state, player_to_move))
                    #print("simulation too long, we used the heuristic")
                    break
                action = greedy_learn.get_move(child_state, 0.0001)
                child_state = search_problem.transition(child_state, action)
                #player_to_move = child_state.player_to_move()
                move_counter += 1
            #print("normal greedy simulate: move count of " + str(move_counter))
            results.append(search_problem.evaluate_terminal(child_state))
        #print("hello_end")
        return results
    
    
    def backpropagate(self, results: list[float], children: list[MCTSNode]):
        for child, result in zip(children, results):
            curr_node = child
            while curr_node is not None:
                player_to_move = curr_node.state.player_to_move()
                curr_node.visits += 1
                if result == -1 and player_to_move == 0:
                    curr_node.value = curr_node.value + 1
                elif result == 1 and player_to_move == 1:
                    curr_node.value = curr_node.value + 1

                curr_node = curr_node.parent


    def __str__(self):
        return "Efe_Luke_SUPERBOT"










def create_value_agent_from_model(model_string: str):
    """
    Create agent object from saved model. This (or other methods like this) will be how your agents will be created in gradescope and in the final tournament.
    """

    model_path = "value_model.pt"
    # TODO: Update number of features for your own encoding size
    #feature_size = 65
    feature_size = 54 #from the 54 features i have
    model = load_model(model_path, ValueNetwork(feature_size))
    heuristic_search_problem = GoProblemLearnedHeuristic(model)

    if model_string == "greedy":
        learned_agent = GreedyAgent(heuristic_search_problem)
    elif model_string == "mm2":
        learned_agent = MinimaxAgent(2, heuristic_search_problem)
    elif model_string == "ab3":
        learned_agent = AlphaBetaAgent(3, heuristic_search_problem)
    elif model_string == "ab4":
        learned_agent = AlphaBetaAgent(4, heuristic_search_problem)
    elif model_string == "ids":
        learned_agent = IterativeDeepeningAgent(5, heuristic_search_problem)
    else:
        raise NameError
    # TODO: Try with other heuristic agents (IDS/AB/Minimax)
    return learned_agent

def get_final_agent_5x5():
    """Called to construct agent for final submission for 5x5 board"""
    learned_agent = final_agent_5x5()
    #learned_agent = create_value_agent_from_model("ab4")
    return learned_agent

def get_final_agent_9x9():
    """Called to construct agent for final submission for 9x9 board"""
    #return GreedyAgent() # construct agent and return it...
    raise NotImplementedError


















def main():
    from game_runner import run_many
    agent2 = GreedyAgent()
    agent1 = final_agent_5x5()
    run_many(agent1, agent2, 4)


if __name__ == "__main__":
    main()
