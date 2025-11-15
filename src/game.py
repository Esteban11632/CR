import time
from funcs import actions
import os
import numpy as np

class Game:
    def __init__(self):
        self.actions = actions()

        self.parent_dir = os.path.dirname(os.path.dirname(__file__))
        self.screenshot_dir = os.path.join(self.parent_dir, 'screenshots')
        self.troops_dir = os.path.join(self.parent_dir, 'troops')
        os.makedirs(self.screenshot_dir, exist_ok=True)

        # Tower positions
        self.tower_pixel_positions = [(773, 633, 48, 18), (1061, 633, 48, 18), (773, 153, 48, 18), (1061, 153, 48, 18)] # left ally tower, right ally tower, left enemy tower, right enemy tower
        self.tower_grid_positions = [(22, 3), (22, 14), (5, 3), (5, 14)] # left ally tower, right ally tower, left enemy tower, right enemy tower

    def reset(self):
        self.actions.click_battle_start()
        time.sleep(7.5)

        # Return initial state
        return self.get_state()
    
    def exit(self):
        self.actions.return_to_menu()
        time.sleep(6)
        print("Exited")
    
    def step(self, action, state):
        # Get the action from the model
        card, x, y = self.actions.get_action(action)

        # If the action value is greater than the last, play the card
        if card != None:
            x, y = self.actions.grid_to_pixel(x, y)
            self.actions.card_play(x, y, card)

        # Wait for the card to be played
        time.sleep(1)
        
        # Get the new state
        new_state = self.get_state()
        
        # Compute the reward
        reward, done = self.compute_reward(state["battlefield"], new_state["battlefield"])
        return new_state, reward, done
    
    def compute_reward(self, old_state, new_state):

        # Get the tower health
        reward = 0.0

        # Count enemies killed
        old_enemy_count = np.count_nonzero(old_state[1])
        new_enemy_count = np.count_nonzero(new_state[1])
        enemy_killed = old_enemy_count - new_enemy_count
        if enemy_killed > 0:
            reward += enemy_killed * 0.1
        
        # Get old tower health
        old_ally_tower_health = (old_state[2][22][3] * 10000) + (old_state[2][22][14] * 10000) # left ally tower health + right ally tower health
        old_enemy_tower_health = (old_state[2][5][3] * 10000) + (old_state[2][5][14] * 10000) # left enemy tower health + right enemy tower health
        
        # Get new tower health
        new_ally_tower_health = (new_state[2][22][3] * 10000) + (new_state[2][22][14] * 10000) # left ally tower health + right ally tower health
        new_enemy_tower_health = (new_state[2][5][3] * 10000) + (new_state[2][5][14] * 10000) # left enemy tower health + right enemy tower health
        
        # Get tower health change
        ally_tower_health_change = old_ally_tower_health - new_ally_tower_health
        enemy_tower_health_change = old_enemy_tower_health - new_enemy_tower_health
        if ally_tower_health_change > 0:
            reward -= ally_tower_health_change * 0.2
        if enemy_tower_health_change > 0:
            reward += enemy_tower_health_change * 0.2

        # Get if victory or defeat
        result, done = self.actions.detect_game_end()
        if result == "victory":
            reward += 100
        elif result == "defeat":
            reward -= 100

        return reward, done

    def get_state(self):

        # confidences = [0.8, 0.7, 0.6] # 80%, 70%, 60% confidence levels
        # grid_region = (666, 46, 544, 780)
        # Grid 18x28
        # 0: friendly, 1: enemy, 2: towers
        battlefield = np.zeros((3, 28, 18), dtype=np.int32)

        # Get number of enemies
        self.actions.capture_area(os.path.join(self.screenshot_dir, "enemies.png"))

        # Get number of allies
        self.actions.capture_area(os.path.join(self.screenshot_dir, "allies.png"))

        # Get tower health
        for i in range(4):
            tower_health = self.actions.get_tower_health(self.tower_pixel_positions[i])
            if tower_health != "No text" and tower_health < 4500:
                battlefield[2][self.tower_grid_positions[i][0]][self.tower_grid_positions[i][1]] = tower_health / 10000

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
        """state = {
            'battlefield': battlefield,
            'elixir': elixir,
            'cards': current_cards
        }"""
        # return state

"""x = torch.zeros(2017)
x[18] = 20
x[2016] = 10

Game().play_step(18)"""