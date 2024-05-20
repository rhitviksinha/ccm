import torch
import numpy as np

def train(mynet, input_pats, output_pats, N, optimizer, criterion, epoch_count, nepochs_additional=5000):
    mynet.train()
    for e in range(nepochs_additional):
        error_epoch = 0.
        perm = np.random.permutation(N)
        for p in perm:
            mynet.zero_grad()
            output, hidden, rep = mynet(input_pats[p, :])
            target = output_pats[p, :]
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            error_epoch += loss.item()
        error_epoch = error_epoch / float(N)
        if e % 50 == 0:
            print(f'epoch {epoch_count + e} loss {round(error_epoch, 4)}')
    return epoch_count + nepochs_additional
