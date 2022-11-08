import numpy as np
from numpy import inf

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, greater=True, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            greater (bool): hope score is greater or lower. Default: True
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.greater = greater
        self.delta = delta
        self.counter = 0
        self.best_score = -inf if greater else inf 
        self.early_stop = False
        
    def __call__(self, val_score):

        score = val_score

        if (self.greater and score <= self.best_score - self.delta) or (not self.greater and score >= self.best_score + self.delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0