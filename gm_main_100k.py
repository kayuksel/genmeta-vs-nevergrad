from argparse import ArgumentParser
import os, time, numpy
parser = ArgumentParser(description='Input parameters for Generative Meta-Learning Optimizer')
parser.add_argument('--noise', default=16, type=int, help='Number of Noise Variables for Gen-Meta')
parser.add_argument('--cnndim', default=2, type=int, help='Size of Latent Dimensions for Gen-Meta')
parser.add_argument('--funcd', default=100000, type=int, help='Size of Benchmark Function Dimensions')
parser.add_argument('--iter', default=200, type=int, help='Number of Total Iterations for Solver')
parser.add_argument('--batch', default=500, type=int, help='Number of Evaluations in an Iteration')
parser.add_argument('--rseed', default=2, type=int, help='Random Seed for Network Initialization')
# hyperparameters for GradInit
parser.add_argument('--gradinit_eta', default=1e-3, type=float, help='The target learning rate')
parser.add_argument('--gradinit_lr', default=1e-2, type=float, help='Step size of GradInit')
parser.add_argument('--gradinit_iters', default=50, type=int, help='Number of GradInit steps.')
parser.add_argument('--gradinit_min_scale', default=1e-2, type=float, help='Set a lower bound for the scale factors')
args = parser.parse_args()
import torch
import torch.nn as nn
from gm_utils import *
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def rastrigin(x, A=10):
    x = x.tanh() * 5
    return (x**2 - A * (2 * math.pi * x).cos()).sum(dim=1) + A * x.shape[1]

def ackley(x, A=20):
    x = x.tanh() * 5
    x1 = -A * (-0.2 * (x.pow(2).mean(dim=1)).sqrt()).exp()
    x2 = (2 * math.pi * x).cos().mean(dim=1).exp()
    return x1 - x2 + A + 2.71828174591064453125

# global minima: -39.16599 * x.shape[1]
def styblinski(x):
    x = x.tanh() * 5
    return (x.pow(4) - 16 * x.pow(2) + 5 * x).sum(dim=1) / 2

def alpine(x):
    x = x.tanh() * 10
    return (x * x.sin() + x / 10).sum(dim=1).abs()


options = {
    "rastrigin": rastrigin,
    "ackley": ackley,
    "styblinski": styblinski,
    "alpine": alpine
}

user_input = ''

while user_input.lower() not in options:
    user_input = input("Select a function to optimize in 100K-dim: ackley, alpine, styblinski, rastrigin\n")

reward_func = options[user_input.lower()]

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
        self.std_weight = nn.Parameter(torch.zeros(args.funcd).cuda()) # changing this for convenience of GradInit
    def forward(self, x):
        mu = self.model(self.extract(x))
        return mu + (self.std_weight * torch.randn_like(mu))

torch.manual_seed(args.rseed)
torch.cuda.manual_seed(args.rseed)
actor = Generator(args.noise).cuda()
opt_A = torch.optim.AdamW(filter(lambda p: p.requires_grad, actor.parameters()), lr=1e-3)
best_reward = None

def gradinit(net, args):
    bias_params = [p for n, p in net.named_parameters() if 'bias' in n]
    weight_params = [p for n, p in net.named_parameters() if 'weight' in n]

    optimizer = RescaleAdam([{'params': weight_params, 'min_scale': args.gradinit_min_scale, 'lr': args.gradinit_lr},
                             {'params': bias_params, 'min_scale': 0, 'lr': args.gradinit_lr}], grad_clip=1.)

    params_list = get_ordered_params(net)
    for total_iters in range(args.gradinit_iters):
        init_inputs = torch.randn((args.batch, args.noise)).cuda().requires_grad_()
        rewards = reward_func(net(init_inputs))
        init_loss = rewards.mean()
        all_grads = torch.autograd.grad(init_loss, params_list, create_graph=True)
        gnorm = sum([g.abs().sum() for g in all_grads])
        optimizer.zero_grad()
        gnorm.backward()
        optimizer.step()

start = time.time()

with torch.backends.cudnn.flags(enabled=False):
    gradinit(actor, args)

for epoch in range(args.iter):
    torch.cuda.empty_cache()
    opt_A.zero_grad()
    z = torch.randn((args.batch, args.noise)).cuda().requires_grad_()
    rewards =  reward_func(actor(z))
    min_index = rewards.argmin()
    if best_reward is None: best_reward = rewards[min_index]
    actor_loss = rewards.mean()
    actor_loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    opt_A.step()
    with torch.no_grad():
        if rewards[min_index] > best_reward: continue
        best_reward = rewards[min_index]
        print('gen-meta trial: %i loss: %f time: %f' % (args.batch*(args.gradinit_iters+epoch), best_reward.item(), (time.time() - start)))