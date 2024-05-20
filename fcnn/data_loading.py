import numpy as np
import torch

def load_data():
    with open('../data/sem_items.txt', 'r') as fid:
        names_items = np.array([l.strip() for l in fid.readlines()])
    with open('../data/sem_relations.txt', 'r') as fid:
        names_relations = np.array([l.strip() for l in fid.readlines()])
    with open('../data/sem_attributes.txt', 'r') as fid:
        names_attributes = np.array([l.strip() for l in fid.readlines()])

    nobj = len(names_items)
    nrel = len(names_relations)
    nattributes = len(names_attributes)

    D = np.loadtxt('../data/sem_data.txt')
    input_pats = D[:, :nobj+nrel]
    input_pats = torch.tensor(input_pats, dtype=torch.float)
    output_pats = D[:, nobj+nrel:]
    output_pats = torch.tensor(output_pats, dtype=torch.float)

    return names_items, names_relations, names_attributes, input_pats, output_pats, nobj, nrel, nattributes
