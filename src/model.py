import torch
import torch.nn as nn
from torchinfo import summary

class ClashRoyaleModel(nn.Module):
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
        result = torch.relu(self.fc1(combined))
        result = self.fc2(result)
        return result

# Usage
model = ClashRoyaleModel()
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