# adaptive_exploration_rl
Adaptive Exploration for Reinforcement Learning. Term project for CSC2516H at University of Toronto.
Project is a collaboration between Geoffrey Harrison (ghharrison@cs.toronto.edu) and Sarah Hindawi (shindawi@cs.toronto.edu). 


The objective of this project is to improve upon Never Give Up, a fairly recent exploration-focused reinforcement learning algorithm, by enabling it to select its exploration rate and discount factors adaptively. We borrow intuition from "Adaptive Discount Factor for Deep Reinforcement Learning in Continuing Tasks with Uncertainty" by Kim et al., modifying their formulations to work with both exploration rate and discount factor, and combining it with the general approach used by Never Give Up. 

The code for never give up that was used as a starting point for this project is available at https://github.com/Coac/never-give-up. Differences from this repo are marked in code with comments reading `New for AERL`. 

Badia, A. P., Sprechmann, P., Vitvitskyi, A., Guo, D., Piot, B., Kapturowski, S., Olivier, L., Arjovsky, M. & Blundell, C. (2020). Never give up: Learning directed exploration strategies. arXiv preprint arXiv:2002.06038.

Le, V. (2021). PyTorch implementation of Never Give Up: Learning Directed Exploration Strategies.
GitHub repository, https://github.com/Coac/never-give-up

Kim, M., Kim, J. S., Choi, M. S., & Park, J. H. (2022). Adaptive Discount Factor for Deep Reinforcement
Learning in Continuing Tasks with Uncertainty. Sensors, 22(19), 7266.