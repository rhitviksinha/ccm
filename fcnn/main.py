from data_loading import load_data
from model import Net
import torch
import torch.nn as nn
import torch.optim as SGD
from training import train
from visualization import plot_rep, plot_dendo

# Load data
names_items, names_relations, names_attributes, input_pats, output_pats, nobj, nrel, nattributes = load_data()

# Set up the model
learning_rate = 0.20
mynet = Net(8, 32, nobj, nattributes)
optimizer = SGD(mynet.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Training phases
epoch_count = 0
nepochs_phase1 = 500
nepochs_phase2 = 1000
nepochs_phase3 = 2500

epoch_count = train(mynet, input_pats, output_pats, input_pats.shape[0], optimizer, criterion, epoch_count, nepochs_additional=nepochs_phase1)
rep1 = get_rep(mynet)
epoch_count = train(mynet, input_pats, output_pats, input_pats.shape[0], optimizer, criterion, epoch_count, nepochs_additional=nepochs_phase2 - nepochs_phase1)
rep2 = get_rep(mynet)
epoch_count = train(mynet, input_pats, output_pats, input_pats.shape[0], optimizer, criterion, epoch_count, nepochs_additional=nepochs_phase3 - nepochs_phase2)
rep3 = get_rep(mynet)

# Visualization
plot_rep(rep1, rep2, rep3, names_items, [nepochs_phase1, nepochs_phase2, nepochs_phase3])
plot_dendo(rep1, rep2, rep3, names_items, [nepochs_phase1, nepochs_phase2, nepochs_phase3])
