# "Mastering" the game of TicTacToe, Connect Four and Chess with AlphaZero

This project is a high-performance implementation of the AlphaZero algorithm, designed to master two-player, perfect-information games including TicTacToe, Connect Four, and Chess.

Based off of Robert Förster's course [AlphaZero from Scratch - Machine Learning Tutorial](https://www.youtube.com/watch?v=wuSQpLinRB4), I first recreated his tutorial of applying a NN to estimate values and priors in a MCTS (Monte-Carlo Tree Search). AlphaZero is essentially a feedback loop involving self-playing. The algorithm iterates and improves its own model output based on a set of completed games: the result of the MCTS performed at each position (policy) and the final result (value) from the perspective of the player-to-move.

The core challenge of AZ is that many game trajectories are required to be generated. To achieve high GPU utilisation and throughput, I added multi-process game workers that are each in charge of potentially hundreds of threads.

## High-Throughput Pipeline
Each worker manages a game and the MCTS of the current position. The workers send batched requests to eval workers, which in turn batch up requests to the GPU. On the other side, a similar process of unbatching and dispatching responses to the destination worker occurs.

```
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###################
thread/game ──► worker queue ────► group & dispatch ─|─► eval worker ──► re-batch ──|──► GPU
                                                                                    |    estimates 
thread/game ◄── worker inbox ◄──|─ group & dispatch ◄─── eval worker ◄── unbatch ◄──|──  policy & value
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###################
```

## Results
A metrics thread in each worker reports the performance of each worker as well as the total performance of the whole system. On Connect Four, this implementation achieves ~25k nodes/s on a 18C/36T server CPU and RTX 3070 with 32 game workers and 4 eval workers.
Small and medium games (TicTacToe and Connect Four respectively) are an ideal test bed for prototyping and validating correctness. An effective neural net can be trained in a matter of minutes or hours.

I devised a protocol to allow for any two player perfect-information game to interface with my MCTS and AZ self-play framework. While the prior two games have a simple implementation, the protocol allowed me to avoid reinventing the wheel by wrapping an external chess library, `python-chess`.

On chess, the same computer achieved ~2k nodes/s. While part of the slowdown is due to calling a deeper, wider CNN, chess exposes the various inefficiencies. Some ideas are reusing tree, only computing deep MCTS for a fraction of positions, lazy instantiation of nodes, avoiding copying of the `Board` object. I have experimented and achieved ~8k nodes/s.

# Future Directions
Future directions include experimenting with mixed precision inference and training. Transformer architecture. A challenge to train AZ-style model to Grandmaster strength in one day.
Another avenue is explaining chess. Engines have been outperforming humans for almost 30 years. But to this day its outputs are notorious for being unhelpful and not instructive, especially for beginner and intermediate players.
With the development of LLMs, people have. but few have integrated them well.

# Tangent: A Critical View of AlphaZero
On a slight tangent, when I first heard about AlphaZero, I thought it was so profound to imagine a computer fighting with, and learning from, itself in an intellectual duel. But many years later when I read the papers and compared AlphaGo (the version that beat 9-dan pro Lee Sedol in the famous 2016 match) to AlphaGo Zero, I realised it was brutally simpler yet more elegant than I had thought. AlphaGo and traditional MCTS both generate actual rollouts to the end of the game to estimate the value of a position. AlphaGo Zero removes rollouts entirely. To anthropomorphise it, it says "You know what? I've searched deep enough. Rollouts are just as noisy; I'll just guess the outcome from here (conditioned on myself playing in self-play mode with all the associated exploration tweaks)," and in lieu uses its value output -- one of the two targets improved over self-play iterations -- at that leaf position. This is a kind of boot-strapping; to have an accurate estimate of the present requires a good estimate of the future, which in turn requires a good estimate of the future's future. This inductive chain only resolves because at some point the game does definitively end with a known result.

It might be eye-opening for you to reframe AlphaZero as "amortized planning" just as Deepmind did in [(Ruoss et al. 2024)](https://arxiv.org/abs/2402.04494). So much for seeming like a civilised or human-like player, evaluating only thousands of moves per position, compared to the millions of a traditional chess engine like Stockfish. Because a trained AlphaZero-type model will have done the extensive search at train-time over a vast and diverse range of positions. In principle, it will have learnt all the relevant features of a chess board to more-than-adequately provide a value estimate for any given position. However, there is nothing inherent about value estimation that requires self-play RL. A substitute for this would be to look at human games and consult each position's winning statistics. Even better, consult with a chess engine so powerful that we consider it as an 'oracle' for the value and best action.

But how to surpass the teacher? This is where we turn to RL and self-play. AlphaZero uses two blades to sharpen each other (though they are actually the same). This self-duel firstly enriches the training examples for more critical, closely-fought positions to search for that tiny extra edge. Secondly, just like a Grandmaster has an intuition of good candidate moves in a position, we introduce the policy target to focus our MCTS search, "I should have focussed more on these and ignore those." This is a virtuous cycle: MCTS hones the value estimation and prior beliefs; better estimates and beliefs lead to deeper and more accurate searches. The bulk of self-play's benefit doesn't come from white fighting black. The two sides are just puppets. It's the refinement that comes from inside each agent's 'mind' when it plays against itself in the MCTS tree. The self-play match is just a vessel to discover diverse and critical positions and ultimately bring games to a conclusive result.

Finally the true elegance of AlphaZero emerges. As DeepMind described it, this use of MCTS is simultaneously "a powerful policy improvement operator" and "a powerful policy evaluation operator". In other words, the MCTS acts as both the judge and teacher, and though it is brute-force at heart, it turns AlphaZero into a self-sustaining engine of improvement.
