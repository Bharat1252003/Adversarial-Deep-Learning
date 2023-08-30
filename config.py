import torch
import numpy as np

data_dir = r'C:\two_classes'

num_classes = 2
learning_rate = 0.001
epochs = 1
batch_size = 32

# attack side hyperparam
epsilon = 0.5
# slice size for crossover
eta_h = 0.5
eta_w = 0.5
delta = 10

lamda = 0.1

pop_size = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

max_iter = 10