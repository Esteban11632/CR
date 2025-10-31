import torch
import torch.nn as nn
from torchinfo import summary
from torch.distributions import Categorical

class FeatureExtractor(nn.Module):
    def __init__(self, num_actions = 4*18*28 + 1): # num actions = 4 cards, 18x28 grid
        super().__init__()

        self.sequential1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # output [batch, 32, 28, 18]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # output [batch, 64, 28, 18]
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # output [batch, 64, 14, 9]
        )

        self.sequential2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # output [batch, 128, 14, 9]
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # output [batch, 256, 14, 9]
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # output [batch, 256, 7, 4]
        )
        
        # After pooling, calculate flattened size
        # 28x18 -> pool -> 14x9 -> pool -> 7x4 (approximately)
        self.spatial_size = 256 * 7 * 4  # Adjust based on actual pooling

        self.flatten = nn.Flatten(start_dim=1) # Flatten: [batch, 256*7*4]
        
        # Process elixir separately
        self.elixir_fc = nn.Linear(1, 32)

        # Process cards separately
        self.cards_fc = nn.Linear(8, 64)  # 8 inputs (4 cards × 2 values)
        #                           ↑  ↑
        #                           │  └─ 64 output features
        #                           └──── 8 inputs: 4×(id, cost)
        
        # Combine both
        self.fc1 = nn.Linear(self.spatial_size + 32 + 64, 256)
        self.fc2 = nn.Linear(256, num_actions)  # Output: action probabilities
        
    def forward(self, battlefield, elixir, cards):

        x = self.sequential1(battlefield)
        x = self.sequential2(x)
        x = self.flatten(x)
        
        # Process elixir
        elixir = elixir.view(-1, 1)  # Reshape to [batch_size, 1]
        e = torch.relu(self.elixir_fc(elixir))

        # Process cards
        cards = cards.view(-1, 8)  # [batch, 8]
        c = torch.relu(self.cards_fc(cards))  # [batch, 64]
        
        # Concatenate both features
        combined = torch.cat([x, e, c], dim=1)
        
        # Fully connected layers
        features = torch.relu(self.fc1(combined))
        return features


class ActorNetwork(nn.Module):
    def __init__(self, num_actions=4*18*28 + 1):
        super().__init__()
        self.features = FeatureExtractor()
        self.action_head = nn.Linear(256, num_actions)
    
    def forward(self, battlefield, elixir, cards):
        features = self.features(battlefield, elixir, cards)
        return self.action_head(features)
    
    def get_action(self, battlefield, elixir, cards):

        # Forward pass
        logits = self.forward(battlefield, elixir, cards)

        # Apply softmax to get probabilities
        dist = Categorical(logits=logits)

        # Sample an action from the distribution
        action = dist.sample()

        # Calculate the log probability of the action
        log_prob = dist.log_prob(action)


        return action.item(), log_prob


class CriticNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = FeatureExtractor()
        self.value_head = nn.Linear(256, 1)
    
    def forward(self, battlefield, elixir, cards):
        features = self.features(battlefield, elixir, cards)
        return self.value_head(features)

# Usage
model = ActorNetwork()
# battlefield = torch.tensor(battlefield_state)  # [batch, 3, 28, 18]
# elixir = torch.tensor([0.7])  # [batch]

# Print model summary
summary(
    model,
    input_size=[
        (1, 3, 28, 18),  # battlefield
        (1,),            # elixir
        (1, 8)           # cards
    ],
    device="cpu"
)

"""
# Week 1: Get PPO working
class SimpleActor(nn.Module):
    def __init__(self, num_actions=2017):
        super().__init__()
        input_size = 3 * 28 * 18 + 1 + 8  # 1521
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
    
    def forward(self, battlefield, elixir, cards):
        flat = torch.cat([
            battlefield.flatten(start_dim=1),
            elixir.view(-1, 1),
            cards.view(-1, 8)
        ], dim=1)
        return self.network(flat)

class SimpleCritic(nn.Module):
    def __init__(self):
        super().__init__()
        input_size = 3 * 28 * 18 + 1 + 8  # 1521
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, battlefield, elixir, cards):
        flat = torch.cat([
            battlefield.flatten(start_dim=1),
            elixir.view(-1, 1),
            cards.view(-1, 8)
        ], dim=1)
        return self.network(flat)

# Goals:
# - Agent learns to place cards
# - Agent learns basic elixir management
# - PPO training loop works

# If this succeeds, THEN upgrade to your CNN
"""