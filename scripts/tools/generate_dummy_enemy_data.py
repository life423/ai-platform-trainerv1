import numpy as np
import os

def generate_dummy_data(n_samples=1000):
    data = []
    for _ in range(n_samples):
        enemy_x = np.random.uniform(0, 800)
        enemy_y = np.random.uniform(0, 600)
        player_x = np.random.uniform(0, 800)
        player_y = np.random.uniform(0, 600)
        dx = player_x - enemy_x
        dy = player_y - enemy_y
        dist = np.sqrt(dx**2 + dy**2)
        state = np.array([enemy_x, enemy_y, player_x, player_y, dist], dtype=np.float32)
        if abs(dx) > abs(dy):
            action = 1 if dx > 0 else 0
        else:
            action = 3 if dy > 0 else 2
        data.append((state, action))

    os.makedirs("data/raw", exist_ok=True)
    states = np.array([s for s, _ in data])
    actions = np.array([a for _, a in data])
    np.savez("data/raw/enemy_training_data.npz", states=states, actions=actions)
    print("Dummy enemy training data saved to data/raw/enemy_training_data.npz")

if __name__ == "__main__":
    generate_dummy_data()
