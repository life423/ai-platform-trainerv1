# enemy_trainer.py
import numpy as np
import os

class EnemyModel:
    def __init__(self, path):
        weights = np.load(path)
        self.W1 = weights['W1']
        self.b1 = weights['b1']
        self.W2 = weights['W2']
        self.b2 = weights['b2']

    def predict(self, state):
        z1 = np.dot(state, self.W1) + self.b1
        h1 = np.maximum(0, z1)
        z2 = np.dot(h1, self.W2) + self.b2
        return int(np.argmax(z2))


class EnemyTrainer:
    def __init__(self, config, teacher):
        self.config = config
        self.teacher = teacher
        in_dim = config['input_size']
        hid_dim = config['hidden_size']
        out_dim = config['output_size']

        self.W1 = np.random.randn(in_dim, hid_dim) * 0.1
        self.b1 = np.zeros(hid_dim)
        self.W2 = np.random.randn(hid_dim, out_dim) * 0.1
        self.b2 = np.zeros(out_dim)
        self.learning_rate = config['learning_rate']

    def generate_training_data(self, episodes=100):
        static_data_path = "data/raw/enemy_training_data.npz"
        if episodes == 0 and os.path.exists(static_data_path):
            print("Loading dummy data from", static_data_path)
            archive = np.load(static_data_path)
            states, actions = archive['states'], archive['actions']
            return list(zip(states, actions))

        print("Generating data with teacher agent...")
        data = []
        for _ in range(episodes):
            state = self.teacher.reset()
            done = False
            while not done:
                action = self.teacher.decide_action(state)
                data.append((state, action))
                state, done = self.teacher.step(action)
        return data

    def train_model(self, data, epochs=10):
        for _ in range(epochs):
            np.random.shuffle(data)
            for state, action in data:
                z1 = np.dot(state, self.W1) + self.b1
                h1 = np.maximum(0, z1)
                z2 = np.dot(h1, self.W2) + self.b2
                exp_z = np.exp(z2 - np.max(z2))
                probs = exp_z / np.sum(exp_z)

                target = np.zeros_like(probs)
                target[action] = 1.0
                grad_z2 = probs - target
                grad_W2 = np.outer(h1, grad_z2)
                grad_b2 = grad_z2
                grad_h1 = np.dot(self.W2, grad_z2)
                grad_z1 = grad_h1 * (z1 > 0)
                grad_W1 = np.outer(state, grad_z1)
                grad_b1 = grad_z1

                self.W1 -= self.learning_rate * grad_W1
                self.b1 -= self.learning_rate * grad_b1
                self.W2 -= self.learning_rate * grad_W2
                self.b2 -= self.learning_rate * grad_b2

    def save_model(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    def load_model(self, path):
        weights = np.load(path)
        self.W1, self.b1 = weights['W1'], weights['b1']
        self.W2, self.b2 = weights['W2'], weights['b2']

    def predict(self, state):
        z1 = np.dot(state, self.W1) + self.b1
        h1 = np.maximum(0, z1)
        z2 = np.dot(h1, self.W2) + self.b2
        return np.argmax(z2)
