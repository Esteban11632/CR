from collections import deque
import time
from funcs import actions
import os
import numpy as np
from troop_cards import cards
import torch

class Game:
    def __init__(self):
        self.frame_stack = deque(maxlen=4)
        self.actions = actions()

        self.parent_dir = os.path.dirname(os.path.dirname(__file__))
        self.screenshot_dir = os.path.join(self.parent_dir, 'screenshots')
        self.troops_dir = os.path.join(self.parent_dir, 'troops')
        os.makedirs(self.screenshot_dir, exist_ok=True)

        self.rf_model = self.__setup_roboflow()
        self.card_model = self.__setup_card_roboflow()
    
    def __setup_roboflow(self):
        pass

    def __setup_card_roboflow(self):
        pass

    def get_frames(self):
        for i in range(4):
            screenshot = self.actions.capture_area(os.path.join(self.screenshot_dir, f"frame_{i}.png"))
            self.frame_stack.append(screenshot)
            time.sleep(0.5)
            print(f"Frame {i} captured")
    
    def get_frame_stack(self):
        return self.frame_stack

    def play_step(self, action):
        # Get the action from the model
        action_index, last, action_value = self.actions.get_action(action)

        # If the action value is greater than the last, play the card
        if action_value > last:
            x, y = self.actions.grid_to_pixel(action_index[1], action_index[2])
            self.actions.card_play(x, y, action_index[0])
    
    def compute_reward(self, old_state, new_state):

        # Get the tower health

        reward = 0
        
        old_num_enemies = 0
        new_num_enemies = 0
        for i in range(len(old_state[1])):
            for j in range(len(old_state[1][i])):
                if old_state[1][i][j] != 0:
                    old_num_enemies += 1
        for i in range(len(new_state[1])):
            for j in range(len(new_state[1][i])):
                if new_state[1][i][j] != 0:
                    new_num_enemies += 1
        if new_num_enemies < old_num_enemies:
            reward += 2
        pass

    def get_state(self):

        # confidences = [0.8, 0.7, 0.6] # 80%, 70%, 60% confidence levels
        # grid_region = (666, 46, 544, 780)
        # Grid 18x28
        # 0: friendly, 1: enemy, 2: towers
        battlefield = np.zeros((3, 28, 18), dtype=np.int32)

        # Identify troops and locations in grid with roboflow

        # Get troop positions
        # Using the musketeer as test

        # Get elixir
        elixir = self.actions.count_elixir() / 10.0

        # Get the current cards
        current_cards = self.actions.capture_individual_cards()

        # Use roboflow to identify the cards
        # Get the elixir and cards
        # Then convert it to an array of 8 values
        # Normalize the values to be between 0 and 1 -> id / 100 and elixir / 10
        # return the state
        # return battlefield, elixir, cards
        pass

x = torch.zeros(2017)
x[18] = 20
x[2016] = 10

Game().play_step(x)