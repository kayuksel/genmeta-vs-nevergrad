import os, time, math
import pandas as pd
import numpy as np
from argparse import ArgumentParser
parser = ArgumentParser(description='Input parameters for Generative Meta-Learning Optimizer')
parser.add_argument('--noise', default=16, type=int, help='Number of Noise Variables for Gen-Meta')
parser.add_argument('--cnndim', default=2, type=int, help='Size of Latent Dimensions for Gen-Meta')
parser.add_argument('--funcd', default=100000, type=int, help='Size of Benchmark Function Dimensions')
parser.add_argument('--iter', default=150, type=int, help='Number of Total Iterations for Solver')
parser.add_argument('--batch', default=64, type=int, help='Number of Evaluations in an Iteration')
parser.add_argument('--rseed', default=2, type=int, help='Random Seed for Network Initialization')
args = parser.parse_args()
import torch
import torch.nn as nn
from gm_utils import *
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from sklearn.metrics import f1_score

ratings_df = pd.read_csv("ratings.dat", sep="::", header=None, names=["userId", "movieId", "rating", "timestamp"], engine='python')

# Create a binary matrix where 1 indicates that a movie has been watched and 0 otherwise
n_users = ratings_df['userId'].nunique()
n_items = ratings_df['movieId'].max()
rank = 64
args.funcd = (n_users + n_items) * rank

print(n_users)
print(n_items)

data = np.zeros((n_users, n_items))
for row in ratings_df.itertuples():
    data[row[1]-1, row[2]-1] = 1

data = torch.from_numpy(data).float().cuda()

# drop all zero rows
data = data[~(data==0).all(axis=1)]

pos_weight = (1.0-data.mean()) / data.mean() 

def reward_func(pop):

    bs = 1000
    num_chunks = math.ceil(len(data) / bs)

    rewards = []
    for row in pop:

        m_pop = row[:n_users*rank].reshape(-1, rank)
        u_pop = row[-n_items*rank:].reshape(-1, rank).T
        recov = m_pop @ u_pop

        recon_loss = 0

        for i in range(num_chunks):
            data_batch = data[i*bs:(i+1)*bs]
            recov_batch = recov[i*bs:(i+1)*bs]
            batch_loss = nn.functional.binary_cross_entropy_with_logits(
                recov_batch, data_batch, pos_weight=pos_weight, reduction='mean')
            recon_loss += batch_loss

        recon_loss /= num_chunks

        rewards.append(recon_loss)

    return torch.stack(rewards)


def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain = 5/3)
        if hasattr(m, 'bias') and m.bias is not None: m.bias.data.zero_()

class Logish(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x * (1 + x.sigmoid()).log()

class LSTMModule(nn.Module):
    def __init__(self, input_size = 1, hidden_size = 1, num_layers = 2):
        super(LSTMModule, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.h = torch.zeros(num_layers, 1, hidden_size, requires_grad=True).cuda()
        self.c = torch.zeros(num_layers, 1, hidden_size, requires_grad=True).cuda()
    def forward(self, x):
        self.rnn.flatten_parameters()
        out, (h_end, c_end) = self.rnn(x, (self.h, self.c))
        self.h.data = h_end.data
        self.c.data = c_end.data
        return out[:,-1, :].flatten()

class Extractor(nn.Module):
    def __init__(self, latent_dim, ks = 5):
        super(Extractor, self).__init__()
        self.conv = nn.Conv1d(args.noise, latent_dim,
            bias = False, kernel_size = ks, padding = (ks // 2) + 1)
        self.conv.weight.data.normal_(0, 0.01)
        self.activation = nn.Sequential(nn.BatchNorm1d(
            latent_dim, track_running_stats = False), Logish())
        self.gap = nn.AvgPool1d(kernel_size = args.batch, padding = 1)
        self.rnn = LSTMModule(hidden_size = latent_dim)
    def forward(self, x):
        y = x.unsqueeze(0).permute(0, 2, 1)
        y = self.rnn(self.gap(self.activation(self.conv(y))))
        return torch.cat([x, y.repeat(args.batch, 1)], dim = 1)

class Generator(nn.Module):
    def __init__(self, noise_dim = 0):
        super(Generator, self).__init__()
        def block(in_feat, out_feat):
            return [nn.Linear(in_feat, out_feat), nn.Tanh()]
        self.model = nn.Sequential(
            *block(noise_dim+args.cnndim, 480), *block(480, 1103), nn.Linear(1103, args.funcd))
        init_weights(self)
        self.extract = Extractor(args.cnndim)
        self.std_weight = nn.Parameter(torch.zeros(args.funcd).cuda())
    def forward(self, x):
        mu = self.model(self.extract(x))
        return mu + (self.std_weight * torch.randn_like(mu))

torch.manual_seed(args.rseed)
torch.cuda.manual_seed(args.rseed)
actor = Generator(args.noise).cuda()
opt_A = torch.optim.AdamW(filter(lambda p: p.requires_grad, actor.parameters()), lr=1e-3)
best_reward = None

start = time.time()

for epoch in range(args.iter):
    torch.cuda.empty_cache()
    opt_A.zero_grad()
    z = torch.randn((args.batch, args.noise)).cuda().requires_grad_()
    action = actor(z)
    rewards =  reward_func(action)
    min_index = rewards.argmin()
    if best_reward is None: best_reward = rewards[min_index]
    actor_loss = rewards.mean()
    actor_loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    opt_A.step()
    with torch.no_grad():
        if rewards[min_index] > best_reward: continue
        best_reward = rewards[min_index]

        row = action[min_index]

        m_pop = row[:n_users*rank].reshape(-1, rank)
        u_pop = row[-n_items*rank:].reshape(-1, rank).T
        recov = m_pop @ u_pop

        recov = (recov.sigmoid() > 0.5).float()
        # calculate the weighted F1 score
        f1 = f1_score(data.flatten().cpu(), recov.flatten().cpu(), average='binary')

        print('gen-meta epoch: %i loss: %f f1: %f time: %f' % (epoch, best_reward.item(), f1, (time.time() - start)))
