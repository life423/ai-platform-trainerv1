import os
import sys
import json
import numpy as np

# ===== Enemy Teacher (Expert Policy) =====
class EnemyTeacher:
    def reset(self):
        # Not used when loading dummy data
        return np.random.rand(5)

    def decide_action(self, state):
        dx = state[2] - state[0]
        dy = state[3] - state[1]
        # Action mapping: 0=left,1=right,2=down,3=up
        if abs(dx) > abs(dy):
            return 1 if dx > 0 else 0
        else:
            return 3 if dy > 0 else 2

    def step(self, action):
        # Not used when loading dummy data
        next_state = np.random.rand(5)
        done = np.random.rand() < 0.1
        return next_state, done