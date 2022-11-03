import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, checkpoint_dir, patience=7, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss, model_anchor, model_recommender ,model_reasoner):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model_anchor, model_recommender, model_reasoner)
        elif score <= self.best_score - self.delta:
            self.counter += 1
            #print('EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model_anchor, model_recommender, model_reasoner)
            self.counter = 0

    def save_checkpoint(self, val_loss, model_anchor, model_recommender ,model_reasoner):
        '''Saves model when validation loss decrease.'''
        state_anchor = model_anchor.state_dict()
        state_recommender = model_recommender.state_dict()
        state_reasoner = model_reasoner.state_dict()
        filename_anchor = str(self.checkpoint_dir / 'checkpoint-anchor.pt')
        torch.save(state_anchor, filename_anchor)
        #self.logger.info("Saving checkpoint: {} ...".format(filename_anchor))
        filename_recommender = str(self.checkpoint_dir / 'checkpoint-recommender.pt')
        torch.save(state_recommender, filename_recommender)
        #self.logger.info("Saving checkpoint: {} ...".format(filename_recommender))
        filename_reasoner = str(self.checkpoint_dir / 'checkpoint-reasoner.pt')
        torch.save(state_reasoner, filename_reasoner)
        #self.logger.info("Saving checkpoint: {} ...".format(filename_reasoner))