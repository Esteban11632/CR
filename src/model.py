import torch
import torch.nn as nn
# from torchinfo import summary
from torch.distributions import Categorical
import torch.optim as optim
import os
from PPOMemory import PPOMemory
import numpy as np
import json
from utils import check_gradient_flow

class FeatureExtractor(nn.Module):
    def __init__(self, alpha = 0.001): # num actions = 4 cards, 18x28 grid
        super().__init__()

        self.sequential1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # output [batch, 32, 28, 18]
            nn.ReLU(), # or nn.LeakyReLU(0.01) to prevent dying gradients
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
        self.fc = nn.Linear(self.spatial_size + 32 + 64, 256)
        
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
        features = torch.relu(self.fc(combined))
        return features


class ActorNetwork(nn.Module):
    def __init__(self, num_actions=4*18*28 + 1, alpha = 0.001):
        super().__init__()
        self.features = FeatureExtractor(alpha)
        self.action_head = nn.Linear(256, num_actions)

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        # Load the model
        self.parent_dir = os.path.dirname(os.path.dirname(__file__))
        self.models_dir = os.path.join(self.parent_dir, 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        self.path = os.path.join(self.models_dir, 'actor.pth')
    
    def forward(self, battlefield, elixir, cards):
        features = self.features(battlefield, elixir, cards)
        logits = self.action_head(features)
        dist = Categorical(logits=logits)
        return dist
    
    def save(self):
        torch.save(self.state_dict(), self.path)
        print(f"Actor model saved to {self.path}")

    def load(self):
        if os.path.exists(self.path):
            self.load_state_dict(torch.load(self.path, map_location=self.device))
            print(f"Actor model loaded from {self.path}")
        else:
            print(f"Actor model not found at {self.path}")

class CriticNetwork(nn.Module):
    def __init__(self, alpha = 0.001):
        super().__init__()
        self.features = FeatureExtractor(alpha)
        self.value_head = nn.Linear(256, 1)

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        # Load the model
        self.parent_dir = os.path.dirname(os.path.dirname(__file__))
        self.models_dir = os.path.join(self.parent_dir, 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        self.path = os.path.join(self.models_dir, 'critic.pth')
    
    def forward(self, battlefield, elixir, cards):
        features = self.features(battlefield, elixir, cards)
        return self.value_head(features)
    
    def save(self):
        torch.save(self.state_dict(), self.path)
        print(f"Critic model saved to {self.path}")

    def load(self):
        if os.path.exists(self.path):
            self.load_state_dict(torch.load(self.path, map_location=self.device))
            print(f"Critic model loaded from {self.path}")
        else:
            print(f"Critic model not found at {self.path}")

class Agent():
    def __init__(self, n_actions, gamma = 0.99, alpha = 0.0003, gae_lambda = 0.95, policy_clip = 0.2, batch_size = 64, n_epochs = 10):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.n_epochs = n_epochs
        self.policy_clip = policy_clip
        
        self.actor = ActorNetwork(num_actions=n_actions, alpha=alpha)
        self.critic = CriticNetwork(alpha=alpha)
        self.memory = PPOMemory(batch_size)
    
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self, best_score=None, episode=None):
        """Save models and training state"""
        print("Saving models...")
        self.actor.save()
        self.critic.save()
        
        # Save training state
        if best_score is not None:
            state = {
                'best_score': best_score,
                'episode': episode
            }
            state_path = os.path.join(self.actor.models_dir, 'training_state.json')
            with open(state_path, 'w') as f:
                json.dump(state, f)
            print(f"Training state saved: best_score={best_score}, episode={episode}")
    
    def load_models(self):
        """Load models and training state"""
        print("Loading models...")
        self.actor.load()
        self.critic.load()
        
        # Load training state
        state_path = os.path.join(self.actor.models_dir, 'training_state.json')
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                state = json.load(f)
            print(f"Training state loaded: {state}")
            return state
        return None
    
    def choose_action(self, battlefield, elixir, cards):
        with torch.no_grad():
            # Convert to tensors and add a batch dimension
            battlefield = torch.tensor([battlefield], dtype=torch.float32).to(self.actor.device)
            elixir = torch.tensor([elixir], dtype=torch.float32).to(self.actor.device)
            cards = torch.tensor([cards], dtype=torch.float32).to(self.actor.device)

            # Forward pass
            dist = self.actor(battlefield, elixir, cards)
            value = self.critic(battlefield, elixir, cards)
            action = dist.sample()
            
            # Get the log probability and action
            probs = torch.squeeze(dist.log_prob(action)).item()
            action = torch.squeeze(action).item()
            value = torch.squeeze(value).item()

        return action, probs, value
    
    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k+1] * (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.actor.device)

            # Normalize advantages (IMPORTANT for stability!)
            # advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            values = torch.tensor(values).to(self.actor.device)
            
            for batch in batches:
                batch_states = state_arr[batch]
            
                # Convert each component to tensor
                battlefields = torch.tensor(
                    np.array([s['battlefield'] for s in batch_states]), 
                    dtype=torch.float32
                ).to(self.actor.device)
                
                elixirs = torch.tensor(
                    np.array([s['elixir'] for s in batch_states]), 
                    dtype=torch.float32
                ).to(self.actor.device)
                
                cards = torch.tensor(
                    np.array([s['cards'] for s in batch_states]), 
                    dtype=torch.float32
                ).to(self.actor.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = torch.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(battlefields, elixirs, cards)
                critic_value = self.critic(battlefields, elixirs, cards)

                critic_value = torch.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip) * advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)

                self.actor.optimizer.step()
                self.critic.optimizer.step()

            # Check gradient flow periodically
            if _ == 0 and len(batches) > 0:  # Only first epoch, first batch
                print(f"\nChecking Actor network:")
                check_gradient_flow(self.actor)
                print(f"\nChecking Critic network:")
                check_gradient_flow(self.critic)

        self.memory.clear_memory()

# Usage
"""model = ActorNetwork()
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
)"""

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