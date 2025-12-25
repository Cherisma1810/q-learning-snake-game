"""
Snake Game with Q-Learning (Epoch-Based Training)

Author  : Anamala Muni Cherisma
Language: Python (Tkinter)
Concept : Reinforcement Learning - Q Learning

Description:
This project implements the classic Nokia Snake Game where
an AI agent learns to play the game using Q-learning.
The agent improves its performance over multiple epochs (games).
"""

from tkinter import *
import random
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# GAME CONFIGURATION
# --------------------------------------------------

WINDOW_WIDTH = 700
WINDOW_HEIGHT = 700
CELL_SIZE = 35
GRID_SIZE = WINDOW_WIDTH // CELL_SIZE

GAME_SPEED = 20          # smaller = faster learning
TOTAL_EPOCHS = 500       # number of games AI will play

BACKGROUND_COLOR = "#121212"
SNAKE_COLOR = "#00FF00"
FOOD_COLOR = "#FF4500"

# Directions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

# Actions (relative to current direction)
STRAIGHT = 0
TURN_RIGHT = 1
TURN_LEFT = 2

# --------------------------------------------------
# Q-LEARNING PARAMETERS
# --------------------------------------------------

Q_table = np.zeros((128, 3))   # 128 states, 3 actions
learning_rate = 0.1           # alpha
discount_factor = 0.9         # gamma

epsilon = 0.3                 # exploration rate
epsilon_decay = 0.995
min_epsilon = 0.01

AI_MODE = True

# --------------------------------------------------
# GLOBAL TRAINING VARIABLES
# --------------------------------------------------

current_epoch = 0
current_score = 0
high_score = 0
scores_per_epoch = []

# --------------------------------------------------
# SNAKE CLASS
# --------------------------------------------------

class Snake:
    def __init__(self):
        self.body = [[5 * CELL_SIZE, 5 * CELL_SIZE]]
        self.direction = RIGHT
        self.graphics = []

    def draw(self):
        """Draw snake on the canvas"""
        for part in self.graphics:
            canvas.delete(part)

        self.graphics.clear()

        for index, (x, y) in enumerate(self.body):
            color_intensity = 50 + int(200 * index / len(self.body))
            color = f"#00{color_intensity:02x}00"

            square = canvas.create_rectangle(
                x, y,
                x + CELL_SIZE, y + CELL_SIZE,
                fill=color, outline=""
            )
            self.graphics.append(square)

# --------------------------------------------------
# FOOD CLASS
# --------------------------------------------------

class Food:
    def __init__(self):
        self.spawn()

    def spawn(self):
        """Place food at a random grid position"""
        self.position = [
            random.randint(0, GRID_SIZE - 1) * CELL_SIZE,
            random.randint(0, GRID_SIZE - 1) * CELL_SIZE
        ]
        self.draw()

    def draw(self):
        canvas.delete("food")
        x, y = self.position
        canvas.create_oval(
            x + 5, y + 5,
            x + CELL_SIZE - 5, y + CELL_SIZE - 5,
            fill=FOOD_COLOR,
            outline="yellow",
            width=2,
            tag="food"
        )

# --------------------------------------------------
# STATE REPRESENTATION
# --------------------------------------------------

def get_state(snake, food):
    """
    State is represented using:
    - danger straight
    - danger left
    - danger right
    - food left/right/up/down
    """
    head_x, head_y = snake.body[0]
    food_x, food_y = food.position

    danger_straight = will_collide(snake, snake.direction)
    danger_left = will_collide(snake, (snake.direction - 1) % 4)
    danger_right = will_collide(snake, (snake.direction + 1) % 4)

    state_bits = [
        danger_straight,
        danger_left,
        danger_right,
        food_x < head_x,
        food_x > head_x,
        food_y < head_y,
        food_y > head_y
    ]

    return int("".join(map(str, map(int, state_bits))), 2)

# --------------------------------------------------
# COLLISION CHECKS
# --------------------------------------------------

def will_collide(snake, direction):
    x, y = snake.body[0]

    if direction == UP:
        y -= CELL_SIZE
    elif direction == DOWN:
        y += CELL_SIZE
    elif direction == LEFT:
        x -= CELL_SIZE
    elif direction == RIGHT:
        x += CELL_SIZE

    if x < 0 or y < 0 or x >= WINDOW_WIDTH or y >= WINDOW_HEIGHT:
        return True

    return [x, y] in snake.body

def collision_detected(position, snake):
    x, y = position
    return (
        x < 0 or y < 0 or
        x >= WINDOW_WIDTH or y >= WINDOW_HEIGHT or
        position in snake.body
    )

# --------------------------------------------------
# MAIN GAME LOOP (Q-LEARNING STEP)
# --------------------------------------------------

def game_step():
    global current_score, high_score, epsilon

    state = get_state(snake, food)

    # ε-greedy action selection
    if random.random() > epsilon:
        action = np.argmax(Q_table[state])
    else:
        action = STRAIGHT

    # Change direction based on action
    if action == TURN_LEFT:
        snake.direction = (snake.direction - 1) % 4
    elif action == TURN_RIGHT:
        snake.direction = (snake.direction + 1) % 4

    # Move snake
    x, y = snake.body[0]
    if snake.direction == UP:
        y -= CELL_SIZE
    elif snake.direction == DOWN:
        y += CELL_SIZE
    elif snake.direction == LEFT:
        x -= CELL_SIZE
    elif snake.direction == RIGHT:
        x += CELL_SIZE

    new_head = [x, y]

    # Collision
    if collision_detected(new_head, snake):
        reward = -10
        Q_table[state][action] += learning_rate * (reward - Q_table[state][action])
        end_epoch()
        return

    snake.body.insert(0, new_head)

    # Eating food
    if new_head == food.position:
        reward = 10
        current_score += 1
        high_score = max(high_score, current_score)
        food.spawn()
    else:
        reward = -0.1
        snake.body.pop()

    next_state = get_state(snake, food)

    # Q-learning update
    Q_table[state][action] += learning_rate * (
        reward + discount_factor * np.max(Q_table[next_state]) - Q_table[state][action]
    )

    # Reduce exploration gradually
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    # Draw updates
    snake.draw()
    food.draw()
    info_label.config(
        text=f"Score: {current_score}   High Score: {high_score}   Epoch: {current_epoch + 1}/{TOTAL_EPOCHS}"
    )

    window.after(GAME_SPEED, game_step)

# --------------------------------------------------
# EPOCH HANDLING
# --------------------------------------------------

def end_epoch():
    global current_epoch, current_score, snake, food

    scores_per_epoch.append(current_score)
    print(f"Epoch {current_epoch + 1} completed | Score: {current_score}")

    current_epoch += 1

    if current_epoch >= TOTAL_EPOCHS:
        window.destroy()
        show_learning_curve()
        return

    current_score = 0
    snake = Snake()
    food = Food()
    game_step()

# --------------------------------------------------
# RESULTS GRAPH
# --------------------------------------------------

def show_learning_curve():
    plt.figure(figsize=(10, 5))
    plt.plot(scores_per_epoch, color="green")
    plt.title("Q-Learning Progress (Score per Epoch)")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.grid(True)
    plt.show()

# --------------------------------------------------
# UI SETUP
# --------------------------------------------------

window = Tk()
window.title("Snake Game with Q-Learning (Epoch-Based)")
window.resizable(False, False)

info_label = Label(
    window,
    text="",
    font=("Consolas", 20),
    bg=BACKGROUND_COLOR,
    fg="#00FF00"
)
info_label.pack()

canvas = Canvas(
    window,
    width=WINDOW_WIDTH,
    height=WINDOW_HEIGHT,
    bg=BACKGROUND_COLOR
)
canvas.pack()

snake = Snake()
food = Food()

game_step()
window.mainloop()

