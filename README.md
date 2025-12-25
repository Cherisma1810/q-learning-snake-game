# Snake Game using Q-Learning (Reinforcement Learning)

This project is an implementation of the classic Snake Game where an AI agent
learns to play the game using Q-Learning, a fundamental Reinforcement Learning algorithm.

The agent improves its performance by playing multiple games (epochs) and
learning from rewards and penalties.

## Key Features
- Q-Learning implemented from scratch (no RL libraries)
- Epoch-based learning (continuous improvement)
- ε-greedy exploration strategy
- Visual game interface using Tkinter
- Learning curve plotted after training

## Technologies Used
- Python
- Tkinter (GUI)
- NumPy
- Matplotlib

## How Learning Works
- Each game is treated as one epoch
- The agent chooses actions: move straight, turn left, or turn right
- Rewards:
  - +10 for eating food
  - -0.1 for normal movement
  - -10 for collision
- The Q-table is updated using the Q-learning equation
2. Run the program:

## Output
- AI-controlled Snake game window
- Score and epoch displayed during training
- Learning curve graph after all epochs

## Learning Outcomes
- Understanding of Reinforcement Learning basics
- Practical implementation of Q-Learning
- Experience with state-action-reward design
- Visualization of learning progress

---

## Author
Anamala Muni Cherisma

## Requirements
numpy
matplotlib

