import json
import torch
import torch.nn as nn
import torch.optim as optim
from enemy_movement_model import EnemyMovementModel

DATA_PATH = "data/raw/training_data.json"
SAVE_PATH = "models/enemy_ai_model.pth"

with open(DATA_PATH, "r") as f:
    training_data = json.load(f)

inputs = []
targets = []

for sample in training_data:
    # Suppose your input is [player_x, player_y, enemy_x, enemy_y, dist]
    input_vec = [
        sample["player_x"],
        sample["player_y"],
        sample["enemy_x"],
        sample["enemy_y"],
        sample["dist"],
    ]

    # If your "target" is a single action like missile_action:
    # e.g., "missile_action" is angle delta? Then 'targets' might be 1-D
    # Or if you actually have two values, e.g. "enemy_vx", "enemy_vy", use that
    # Adjust according to your actual data keys:
    if "missile_action" in sample:
        target_val = [sample["missile_action"]]
        targets.append(target_val)
    else:
        # skip or default
        continue

    inputs.append(input_vec)

# Convert to tensors
inputs = torch.tensor(inputs, dtype=torch.float32)
targets = torch.tensor(targets, dtype=torch.float32)

model = EnemyMovementModel(input_size=5, hidden_size=128, output_size=1)  # Adjust
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 20
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    predictions = model(inputs)
    loss = criterion(predictions, targets)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), SAVE_PATH)
print(f"Model saved to {SAVE_PATH}")
