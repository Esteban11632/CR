import pyautogui
import os
import time
from paddleocr import PaddleOCR
import numpy as np
import cv2
from fastai.vision.all import *
import pathlib
import platform
from troop_cards import cards

# CRITICAL: Force CPU-only mode for PaddleOCR to prevent CUDA conflicts with PyTorch
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class actions:
    def __init__(self):
        self.parent_dir = os.path.dirname(os.path.dirname(__file__))
        # Go up one directory (from src to CR) then to main_images
        self.images_folder = os.path.join(self.parent_dir, 'main_images')
        print(self.images_folder)

        self.models_dir = os.path.join(self.parent_dir, 'models')
        self.card_model_path = os.path.join(self.models_dir, 'card_model.pkl')
        
        # Simplest/fastest config for reading tower health numbers
        """self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang='en'
        )"""

        self.num_cards = 4
        self.grid_x = 18
        self.grid_y = 28
        
        self.TOP_LEFT_X = 666
        self.TOP_LEFT_Y = 46
        self.BOTTOM_RIGHT_X = 1210
        self.BOTTOM_RIGHT_Y = 1016
        self.FIELD_AREA = (self.TOP_LEFT_X, self.TOP_LEFT_Y, self.BOTTOM_RIGHT_X, self.BOTTOM_RIGHT_Y)
        
        self.WIDTH = self.BOTTOM_RIGHT_X - self.TOP_LEFT_X # 1210 - 666 = 544
        self.HEIGHT = self.BOTTOM_RIGHT_Y - self.TOP_LEFT_Y # 1016 - 46 = 970
        
        # Add card bar coordinates for Windows
        self.CARD_BAR_X = 780
        self.CARD_BAR_Y = 850
        self.CARD_BAR_WIDTH = 426
        self.CARD_BAR_HEIGHT = 125

        # Card position to key mapping
        self.card_keys = {
            0: '1',  # Changed from 1 to 0
            1: '2',  # Changed from 2 to 1
            2: '3',  # Changed from 3 to 2
            3: '4'   # Changed from 4 to 3
        }

        self.current_card_positions = {}

        self.rf_model = self.__setup_roboflow()
        self.card_model = self.__setup_card_roboflow()
    
    def __setup_roboflow(self):
        pass

    def __setup_card_roboflow(self):
        # Fix for Windows/Linux path compatibility
        if platform.system() == 'Windows':
            temp = pathlib.PosixPath
            pathlib.PosixPath = pathlib.WindowsPath

        # Restore PosixPath
        try:

            # Load model
            learn = load_learner(self.card_model_path)
            
        finally:
            # Always restore, even if loading fails
            if platform.system() == 'Windows':
                pathlib.PosixPath = temp

        # Return the model
        return learn
    
    def capture_area(self, save_path):
        screenshot = pyautogui.screenshot(region=(self.TOP_LEFT_X, self.TOP_LEFT_Y, self.WIDTH, self.HEIGHT))
        screenshot.save(save_path)

    def demo(self, save_path):
        # True location of the elixir bar
        # screenshot = pyautogui.screenshot(region=(813, 980, 382, 30))

        # Pixel bar
        # screenshot = pyautogui.screenshot(region=(813, 999, 382, 1))

        # Winner region
        # screenshot = pyautogui.screenshot(region=(666, 46, 544, 700))

        # Match over region
        # screenshot = pyautogui.screenshot(region=(666, 46, 544, 700))

        # Grid region
        # screenshot = pyautogui.screenshot(region=(666, 46, 544, 780))

        # screenshot = pyautogui.screenshot(region=(666, 477, 544, 780))

        # Get the squares of the battlefield
        # battlefield[0][0] = (710, 142, 26, 26)
        # upper part y = 142
        # lower part y = 477
        """x = 710
        for i in range(20):
            screenshot = pyautogui.screenshot(region=(x, 142, 26, 20))
            save_path = os.path.join(os.path.dirname(self.script_dir), 'screenshots', f"battlefield_{i+1}.png")
            screenshot.save(save_path)
            x += 26
        """
        """y = 477
        for i in range(15):
            screenshot = pyautogui.screenshot(region=(710, y, 26, 20))
            save_path = os.path.join(os.path.dirname(self.script_dir), 'screenshots', f"battlefield_{i+1}.png")
            screenshot.save(save_path)
            y += 20
        # Get separation of lake"""

        # Left ally tower
        # screenshot = pyautogui.screenshot(region=(773, 633, 48, 18))

        # Right ally tower (change x if necessary)
        # screenshot = pyautogui.screenshot(region=(1061, 633, 48, 18))

        # Left enemy tower (change y if necessary)
        # screenshot = pyautogui.screenshot(region=(773, 153, 48, 18))

        # Right enemy tower (change x and y if necessary)
        screenshot = pyautogui.screenshot(region=(1061, 153, 48, 18))

        screenshot.save(save_path)

    def grid_to_pixel(self, row, col):
        x = 723 + col * 26
        if row < 13:
            y = 152 + row * 20
        else:
            y = 487 + (row - 13) * 20
        return x, y
    
    def get_action(self, action):

        if action == 0:
        # Special case: do nothing
            return None, None, None
    
        # Subtract 1 to account for "do nothing" action
        action_idx = action - 1
        
        # Decode to 3D coordinates
        card = action_idx // (self.grid_x * self.grid_y)  # Which card (0-3)
        remainder = action_idx % (self.grid_x * self.grid_y)
        x = remainder // self.grid_y  # X position (0-17)
        y = remainder % self.grid_y   # Y position (0-27)
        
        return card, x, y
    
    def get_tower_health(self, region):

        # Get the tower screenshot
        tower = pyautogui.screenshot(region=region)
        tower = np.array(tower)
        
        # Convert to grayscale
        tower_gray = cv2.cvtColor(tower, cv2.COLOR_BGR2GRAY)
        
        # Apply binary threshold
        _, tower_bw = cv2.threshold(tower_gray, 215, 255, cv2.THRESH_BINARY)
        
        # Make white numbers black
        tower_bw = cv2.bitwise_not(tower_bw)
        
        # Convert back to 3-channel BGR (numbers are now black)
        tower_rgb = cv2.cvtColor(tower_bw, cv2.COLOR_GRAY2BGR)

        # Show the tower screenshot
        # img = Image.fromarray(tower_rgb)
        # img.show()

        # Get the tower health
        pred = self.ocr.predict(tower_rgb)
        tower_health = pred[0]['rec_texts'][0] if pred and len(pred[0]['rec_texts']) > 0 else "No text"
        if pred and len(pred[0]['rec_texts']) > 0 and tower_health.isnumeric():
            tower_health = int(tower_health)
        else:
            tower_health = "No text"
        return tower_health
    
    def capture_card_area(self, save_path):
        """Capture screenshot of card area"""
        screenshot = pyautogui.screenshot(region=(
            self.CARD_BAR_X, 
            self.CARD_BAR_Y, 
            self.CARD_BAR_WIDTH, 
            self.CARD_BAR_HEIGHT
        ))
        screenshot.save(save_path)
    
    def capture_individual_cards(self):
        """Capture and split card bar into individual card images"""
        screenshot = pyautogui.screenshot(region=(
            self.CARD_BAR_X, 
            self.CARD_BAR_Y, 
            self.CARD_BAR_WIDTH, 
            self.CARD_BAR_HEIGHT
        ))
        
        # Calculate individual card widths
        card_width = self.CARD_BAR_WIDTH // 4
        current_cards = np.zeros(8, dtype=np.float32)
        
        # Split into 4 individual card images
        for i in range(4):
            left = i * card_width
            card_img = screenshot.crop((left, 0, left + card_width, self.CARD_BAR_HEIGHT))
            """save_path = os.path.join(os.path.dirname(self.script_dir), 'screenshots', f"card_{i+1}.png")
            card_img.save(save_path)
            cards.append(save_path)"""

            # Make the card image grayscale and then back to BGR
            card_img = cv2.cvtColor(np.array(card_img), cv2.COLOR_BGR2GRAY)
            card_img = cv2.cvtColor(card_img, cv2.COLOR_GRAY2BGR)

            # Make the prediction
            pred_class, pred_idx, probs = self.card_model.predict(card_img)
            print(f"Predicted: {pred_class}")
            print(f"Confidence: {probs[pred_idx]*100:.2f}%")

            # Parse the prediction
            try:
                if pred_class in cards:
                    current_cards[i * 2] = cards[pred_class]['id'] / 110.0  # Normalize by max_id + 1
                    current_cards[i * 2 + 1] = cards[pred_class]['cost'] / 10.0  # Max cost is 9
                else:
                    print(f"Warning: Card '{pred_class}' not found in dictionary!")
                    # Set to 0 or some default values
                    current_cards[i * 2] = 0.0
                    current_cards[i * 2 + 1] = 0.0
            except Exception as e:
                print(f"Error processing card {i}: {e}")
                current_cards[i * 2] = 0.0
                current_cards[i * 2 + 1] = 0.0
        
        return current_cards
    
    def count_elixir(self):
        # Elixir RGB values
        target = (204, 32, 210)
        tolerance = 80
        count = 0
        
        start_x = 813  # Left position
        y = 999       # Middle of the height
        width = 382    # Total width
        step = width // 9  # 9 gaps between 10 elixir positions
        
        for x in range(start_x, start_x + width, step):
            r, g, b = pyautogui.pixel(x, y)
            if (abs(r - target[0]) <= tolerance) and (abs(g - target[1]) <= tolerance) and (abs(b - target[2]) <= tolerance):
                count += 1
        return count
    
    def update_card_positions(self, detections):
        """
        Update card positions based on detection results
        detections: list of dictionaries with 'class' and 'x' position
        """
        # Sort detections by x position (left to right)
        sorted_cards = sorted(detections, key=lambda x: x['x'])
        
        # Map cards to positions 0-3 instead of 1-4
        self.current_card_positions = {
            card['class']: idx  # Removed +1
            for idx, card in enumerate(sorted_cards)
        }
    
    def card_play(self, x, y, card_index):
        print(f"Playing card {card_index} at position ({x}, {y})")
        if card_index in self.card_keys:
            key = self.card_keys[card_index]
            print(f"Pressing key: {key}")
            pyautogui.press(key)
            time.sleep(0.2)
            print(f"Moving mouse to: ({x}, {y})")
            pyautogui.moveTo(x, y, duration=0.2)
            print("Clicking")
            pyautogui.click()
        else:
            print(f"Invalid card index: {card_index}")
    
    def click_battle_start(self):
        print("Clicking battle start")
        pyautogui.moveTo(930, 800, duration=0.2)
        pyautogui.click()
        print("Battle started")

    def return_to_menu(self):
        print("Returning to menu")
        # Locate the ok button
        ok_img = os.path.join(self.images_folder, "okbutton.png")
        
        print(f"\nLooking for Winner.png at: {ok_img}")
        if os.path.exists(ok_img):
            print("File found!")
        else:
            print("File not found! Check the path.")
        confidences = [0.8, 0.7, 0.6] # 80%, 70%, 60% confidence levels

        region = (666, 46, 544, 970) # Change height

        for confidence in confidences:
            print(f"\nTrying detection with confidence: {confidence}")
            location = None

            try:
                location = pyautogui.locateOnScreen(
                    ok_img, 
                    confidence=confidence, 
                    grayscale=True, 
                    region=region
                )
                if location:
                    print("OK button found!")
                    x, y = pyautogui.center(location)
                    pyautogui.moveTo(x, y, duration=0.2)
                    pyautogui.click()
                    print("Returned to menu")
                    return
            except Exception as e:
                print(f"Error locating OK button: {str(e)}")
        
    def detect_game_end(self):
        done = False
        winner_img = os.path.join(self.images_folder, "Winner.png")
        print(f"\nLooking for Winner.png at: {winner_img}")
        if os.path.exists(winner_img):
            print("File found!")
        else:
            print("File not found! Check the path.")
        confidences = [0.8, 0.7, 0.6] # 80%, 70%, 60% confidence levels

        # Region of the two opponent names and the winner text
        winner_region = (666, 46, 544, 700)

        for confidence in confidences:
            print(f"\nTrying detection with confidence: {confidence}")
            winner_location = None

            # Try to find Winner in region
            try:
                winner_location = pyautogui.locateOnScreen(
                    winner_img, 
                    confidence=confidence, # Try 80%, 70%, 60% confidence levels
                    grayscale=True, 
                    region=winner_region
                )
            except Exception as e:
                print(f"Error locating Winner: {str(e)}")

            if winner_location:
                _, y = pyautogui.center(winner_location)
                print(f"Found 'Winner' at y={y} with confidence {confidence}")
                result = "victory" if y > 350 else "defeat"
                done = True
                return result, done
        return None, done
        

# Get parent directory (one level up from src)
parent_dir = os.path.dirname(os.path.dirname(__file__))
screenshot_dir = os.path.join(parent_dir, 'screenshots')
os.makedirs(screenshot_dir, exist_ok=True)

"""while True:
    elixir = actions().count_elixir()
    print(f"Current elixir: {elixir}")
    time.sleep(0.5)  # Add a small delay to prevent excessive CPU usage"""

# actions().demo(os.path.join(screenshot_dir, 'demo3.png'))
# actions().card_play(1000,500,0)
# actions().capture_area(os.path.join(screenshot_dir, 'current.png'))
# actions().capture_individual_cards()
# print(actions().detect_game_end())
# print(actions().grid_to_pixel(0, 1)) # rows, cols

"""tower_pixel_positions = [(773, 633, 48, 18), (1061, 633, 48, 18), (773, 153, 48, 18), (1061, 153, 48, 18)] # left ally tower, right ally tower, left enemy tower, right enemy tower

action = actions()
for i in tower_pixel_positions:
    print(f"Tower {i} health:")
    print(action.get_tower_health(i))"""

# while True:
#     print(action.get_tower_health(tower_pixel_positions[0]))
cards = actions().capture_individual_cards()
print(cards)