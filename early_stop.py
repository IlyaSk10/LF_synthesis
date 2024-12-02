import torch
import torch.nn as nn
import torch.optim as optim


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        Args:
            patience (int): Сколько эпох ждать после последнего улучшения.
            min_delta (float): Минимальное изменение для учета как улучшение.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, checkpoint):
        if self.best_loss is None:
            self.best_loss = val_loss
            torch.save(checkpoint['model_state_dict'], 'checkpoint.pth')
            print(self.counter)
        elif val_loss < self.best_loss - self.min_delta:
            self.counter = 0
            print(self.counter)
            self.best_loss = val_loss
            torch.save(checkpoint['model_state_dict'], 'checkpoint.pth')
        else:
            self.counter += 1
            print(self.counter)
            if self.counter >= self.patience:
                self.early_stop = True
