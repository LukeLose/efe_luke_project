***FINAL REPORT:***


***Overview:***

Given both of our implementations of MCTS and our experimental performance checks against other agents, we decided on using some variation of MCTS as our baseline model. We were also confident that this would be a good move given the fact that we know ALPHAGO uses the MCTS algorithm in its extraordinary performance. We implemented several new features for our MCTS to become a more efficient and accurate model for Go. These were, changing the rollout policy to utilize our value network from part 2, implementing a decaying c value, adding more complex time limitations, limiting the rollout simulation at a set amount of iterations, and more! Below are our intuitions behind every change we made as well as an explanation of our process of our implementations. 

***Better Rollout Policy With Greedy Agent and Value Network:***

The first change we wanted to make was to make an augmentation of the rollout policy of MCTS. The current roll out policy of MCTS has random moves played to simulate the rest of the game given a current state, however, even though it is a fast simulation, we do not believe it is representative of how a real game would play out. Even more so against the advanced bots that our peers and TAs would create. Thus we wanted to change the rollout simulation to play more sensical moves, thus giving us a more accurate representation of how the game would actually play out and provide more confidence in the UTC calculation being an accurate assessment of the "winningness" of a move.

***Decaying C Value:***

In light of our discussions in class and our own research we knew that high level MCTS implementations used a decaying c value for the policy calculation as opposed to a set value of sqrt(2). Hence, we wanted to make this change for our own model  as well. The logic behind this change was that as the tree grew more and more there was more data to figure out which paths were better than others. Thus, it made sense to explore these better options in more depth (exploitation) rather than explore entirely new options at a surface level (exploration). We initially had few ideas on how to do the decaying C value: exponential decay, linear decay,  logarithmic decay. After trying out different iterations we decided to land on linear decay as it seemed to perform the best out of the options. Additionally, we added a min_c value as an instance variable to not allow it to go lower than a set threshold. After some playing around we set this value at 1. 

***Hybrid Model:***

Another addition we made to our existing implementation of MCTS was to combine it with our other agents. Knowing that Alpha Beta and IDS would perform really well when there is a limited number of moves left on the board, we decided to use these search algorithms within our get_move method. More specifically, we edited the get_move method to use IDS to pick a move when there were already more than 17 pieces on the board. With enough time and computation power IDS should theoretically be able to infer the optimal play at any given state (assuming state space is finite). Giving the reins to IDS when there are limited options left in the game, allows us to maximize the chance of reaching that optimal move. 

***Additional Time Complexities:***


Yet another thing we considered to improve our MCTS model from part 1 was to add more advanced time handling for each move. Using some general intuition and some research we saw that midgame generally takes the most time for players of Go. Thus, we changed our implementation so that it can use more time when there are 15-20 pieces on the board. While these cutoff values seem rather arbitrary, these are the values that seemed to yield the best performance for us and we believe they define the midgame generally well. Within this piece range our model chooses the time_limit for that move to be the minimum of 3 seconds or 30% of the remaining time. Since this implementation can potentially lead to our model running out of time we also added an additional check to make sure 30% of the remaining time is greater 1 (if it is less than 1, the model takes 1 second for the move). 

***Simulation Move Count Threshold:***

A further additional improvement we made was a cap to our simulation in our augmented agent. Given that our simulation is no longer random, it is much more likely for our greedy learned simulations to play longer, more advanced games. With this advancement in simulated strategy, we often ran into issues where the simulation of the game ate up too much time, thus reducing the number of games simulated in our tree and reducing our confidence that the UCT heuristic of our tree gives us the best possible move. A greedy, learned agent doesn’t take that long to run itself but when combined with multitudes more game moves per simulation, it does make sense that we end up with much fewer simulated games. With this in mind, we changed it so that we can simulate a maximum of 50 moves per simulation. Thus, we don’t run into the issue of greedy learned vs greedy learned games that happen to last an incredibly long amount of time. If we do run into a game that doesn’t end with a terminal state within 50 moves, we use our ValueNet heuristic, which gives us a decently accurate prediction of who is likely to win given that simulated game state. Previously, our backpropagate was hard coded to take in values of -1 and 1, however, this would not work with the float heuristic values provided by the ValueNet heuristic. Thus, we changed it to accept a continuum of values to add to the value.

***Hard Coding First Two Moves:***

Given that it takes an exponential amount of time to explore multiple depths of outcomes, we wanted to simplify the first two moves as we found that securing the middle of the reduced 5x5 board was a strong strategy. Thus our bot tries to take the complete middle or a square adjascent to the complete middle in its first two moves. This gives it a strong foundation in controlling the center of our GO game while using a truly trivial amount of time to complete these moves, which we can then use the additional time later on in the mid game where it is more critical to further flesh out the a more important game state.

***Closing Statement***

With all these additions to our MCTS we created an entirely new agent that performs quite well against all other agents (find win rates against all other models in the pdf file attached in our submission). This model also has a lot more room for improvement compared to our initial implementation as it has many more hyperparameters to fine tune. While we likely weren’t able to optimize all of them in this limited time, with more time and effort our final model can perform even better! 

