import numpy as np
import pdb
from typing import Tuple


class Environment():
    def __init__(self) -> None:
        self.map = np.ndarray(shape=(8, 8))
        self.rewardPos = 63
        self.reset()

    def step(self, action: int) -> Tuple[int, float, bool]:
        if action == 0 and self.agentPos // 8 > 0:
            self.agentPos -= 8
        elif action == 1 and self.agentPos % 8 < 7:
            self.agentPos += 1
        elif action == 2 and self.agentPos // 8 < 7:
            self.agentPos += 8
        elif action == 3 and self.agentPos % 8 > 0:
            self.agentPos -= 1
        reward = 0.0
        if self.agentPos == self.rewardPos:
            reward = 1.0
            self.episodeOver = True
        return self.agentPos, reward, self.episodeOver

    def reset(self) -> int:
        self.agentPos = 0
        self.episodeOver = False
        return self.agentPos
