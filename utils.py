import torch
import torch.nn.functional as F
import numpy as np

class Accuracy:
    def __init__(self):
        self.accuracy = {}

    def calc_accuracy(self, y_pred: torch.Tensor, y_true: torch.Tensor, epoch: int) -> np.ndarray:
        if epoch not in self.accuracy:
            self.accuracy[epoch] = {
                "hist": np.zeros((5,)),  
                "num-preds": 0,
            }

        batch_size = y_pred.size()[0]
        y_pred_classes = torch.argmax(y_pred, dim=2)

        accuracy = np.zeros((5,))
        y_true = y_true.argmax(dim=2)

        for i in range(batch_size):
            for j in range(5):
                # print(f"{y_pred_classes.size() = }")
                # print(f"{y_true.size() = }\n")
                if y_pred_classes[i][j] == y_true[i][j]:
                    accuracy[j] += 1

        self.accuracy[epoch]['hist'] += accuracy
        self.accuracy[epoch]['num-preds'] += batch_size

        return self.accuracy[epoch]['hist'] / self.accuracy[epoch]['num-preds']
