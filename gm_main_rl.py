from argparse import ArgumentParser
parser = ArgumentParser(description='Input parameters for Generative Meta-Learning Optimizer')
parser.add_argument('--noise', default=16, type=int, help='Number of Noise Variables for Gen-Meta')
parser.add_argument('--cnndim', default=2, type=int, help='Size of Latent Dimensions for Gen-Meta')
parser.add_argument('--funcd', default=100000, type=int, help='Size of Benchmark Function Dimensions')
parser.add_argument('--iter', default=200, type=int, help='Number of Total Iterations for Solver')
parser.add_argument('--batch', default=32, type=int, help='Number of Evaluations in an Iteration')
parser.add_argument('--rseed', default=2, type=int, help='Random Seed for Network Initialization')
args = parser.parse_args()
import torch, time, pdb
import torch.nn as nn

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import numpy as np
import gymnasium as gym
env = gym.make("Pendulum-v1")
obs, info = env.reset(seed=42)
action_size = 1
action_scale = 2.0

size = 32
class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        def block(in_feat, out_feat):
            return [nn.Linear(in_feat, out_feat), nn.Mish()]
        self.model = nn.Sequential(*block(obs.shape[0], size // 2),
            *block(size//2, size), *block(size, size//2), *block(size//2, action_size))

        self.model[-1] = nn.Tanh()

    def forward(self, x):
        return self.model(x) * action_scale

    def num_params(self):
        return sum([np.prod(params.size()) for params in self.state_dict().values()])

    def set_params(self, diff_params):
        state_dict = dict()
        for key, params in self.state_dict().items():
            size = params.size()
            state_dict[key] = diff_params[:np.prod(size)].view(*size)
            diff_params = diff_params[np.prod(size):]
        self.load_state_dict(state_dict)

agent = Agent().cuda()
print(agent.num_params())

args.funcd = agent.num_params()

def reward_func(solutions):

    rewards = []
    for sol in solutions:
        agent.set_params(sol)
        observation, info = env.reset(seed=42)

        total_reward = torch.zeros(1).cuda().requires_grad_()

        terminated, truncated = False, False
        while not (terminated or truncated):
            action = agent(torch.from_numpy(observation).float().cuda())
            observation, reward, terminated, truncated, info = env.step(action.detach().cpu().numpy())
            total_reward = total_reward + reward
        
        rewards.append(total_reward)
    return -torch.cat(rewards)

def visualize(sol):

    viz = gym.make("Pendulum-v1", render_mode = 'human')

    agent.set_params(sol)
    observation, info = viz.reset(seed=42)

    terminated, truncated = False, False
    while not (terminated or truncated):
        action = agent(torch.from_numpy(observation).float().cuda())
        observation, reward, terminated, truncated, info = viz.step(action.detach().cpu().numpy())
        viz.render()
    viz.close()
    
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

start = time.time()

for epoch in range(args.iter):
    torch.cuda.empty_cache()
    opt_A.zero_grad()
    z = torch.randn((args.batch, args.noise)).cuda().requires_grad_()
    actions = actor(z)
    rewards =  reward_func(actions)
    min_index = rewards.argmin()
    if best_reward is None: best_reward = rewards[min_index]
    actor_loss = rewards.mean()
    actor_loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    opt_A.step()
    with torch.no_grad():
        if rewards[min_index] > best_reward: continue
        best_reward = rewards[min_index]
        print('gen-meta trial: %i loss: %f time: %f' % (args.batch*epoch, best_reward.item(), (time.time() - start)))
        visualize(actions[min_index])
        