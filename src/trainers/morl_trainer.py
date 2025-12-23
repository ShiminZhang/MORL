# multiple rl technique will be implemented and test, we need a abstract class to handle the training process
from abc import ABC, abstractmethod

class MORLTrainer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def train(self, train_loader):
        pass

    @abstractmethod
    def evaluate(self, eval_loader):
        pass