import numpy as np
import pdb


class Environment():
    def __init__(self) -> None:
        self.map = np.ndarray(shape=(8, 8))

    def step(self, action: int) -> None:
        pass
