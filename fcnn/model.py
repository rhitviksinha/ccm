import torch.nn as nn
from torch.nn.functional import relu, sigmoid

class Net(nn.Module):
    def __init__(self, rep_size, hidden_size, nobj, nattributes):
        super(Net, self).__init__()
        self.rep_layer = nn.Linear(nobj, rep_size)
        self.hidden_layer = nn.Linear(nobj + rep_size, hidden_size)
        self.attribute_layer = nn.Linear(hidden_size, nattributes)

    def forward(self, x):
        x = x.view(-1, nobj + nrel)
        x_pat_item = x[:, :nobj]
        x_pat_rel = x[:, nobj:]

        rep = relu(self.rep_layer(x_pat_item))
        rep_rel = torch.cat((rep, x_pat_rel), 1)
        hidden = relu(self.hidden_layer(rep_rel))
        output = sigmoid(self.attribute_layer(hidden))

        return output, hidden, rep
