import torch.nn as nn


class Loss(nn.Module):

    def __init__(self):
        super(Loss, self).__init__()

        self.log_softmax = nn.LogSoftmax(dim=1)

    def my_cross_entropy(self, x, y):
        log_prob = -1.0 * self.log_softmax(x)
        loss = log_prob.gather(1, y.unsqueeze(1))
        loss = loss.mean()

        return loss

    def my_custom_loss(self, x, y):
        # Let your imagination (and mathematical logic) run wild :D
        pass


    def forward(self, x, y):
        loss = self.my_cross_entropy(x, y)

        return loss

