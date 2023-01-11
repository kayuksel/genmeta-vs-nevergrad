import torch, math
from torch import nn

def unitwise_norm(x):
    dim = [1, 2, 3] if x.ndim == 4 else 0
    return torch.sum(x**2, dim=dim, keepdim= x.ndim > 1) ** 0.5

class AGC(torch.optim.Optimizer):
    def __init__(self, params, optim: torch.optim.Optimizer, clipping = 1e-2, eps = 1e-3):
        self.optim = optim
        defaults = dict(clipping=clipping, eps=eps)
        defaults = {**defaults, **optim.defaults}
        super(AGC, self).__init__(params, defaults)
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                param_norm = torch.max(unitwise_norm(
                    p), torch.tensor(group['eps']).cuda())
                grad_norm = unitwise_norm(p.grad)
                max_norm = param_norm * group['clipping']
                trigger = grad_norm > max_norm
                clipped = max_norm / torch.max(grad_norm, torch.tensor(1e-6).cuda())
                p.grad.data.copy_(torch.where(trigger, p.grad * clipped, p.grad))
        self.optim.step(closure)

class RescaleAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 min_scale=0, grad_clip=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, amsgrad=amsgrad, min_scale=min_scale, grad_clip=grad_clip)
        super(RescaleAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RescaleAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        grad_list = []
        alphas = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # State initialization
                amsgrad = group['amsgrad']
                state = self.state[p]
                if len(state) == 0:
                    state['alpha'] = 1.
                    state['init_norm'] = p.norm().item()
                    state['step'] = 0
                    state['cons_step'] = 0
                    # Exponential moving average of gradient values for the weight norms
                    state['exp_avg'] = 0
                    # Exponential moving average of squared gradient values for the weight norms
                    state['exp_avg_sq'] = 0
                    state['cons_exp_avg'] = 0

                curr_norm = p.data.norm().item()
                if state['init_norm'] == 0 or curr_norm == 0: continue

                grad = torch.sum(p.grad * p.data).item() * state['init_norm'] / curr_norm

                if group['grad_clip'] > 0:
                    grad = max(min(grad, group['grad_clip']), -group['grad_clip'])

                beta1, beta2 = group['betas']

                state['cons_step'] += 1
                state['cons_exp_avg'] = state['cons_exp_avg'] * beta1 + grad * (1 - beta1)                    
                steps = state['cons_step']
                exp_avg = state['cons_exp_avg']
                state['exp_avg_sq'] = state['exp_avg_sq'] * beta2 + (grad * grad) * (1 - beta2)
                exp_avg_sq = state['exp_avg_sq']

                bias_correction1 = 1 - beta1 ** steps
                bias_correction2 = 1 - beta2 ** (state['cons_step'] + state['step'])

                denom = math.sqrt(exp_avg_sq / bias_correction2) + group['eps']

                step_size = group['lr'] / bias_correction1

                state['alpha'] = max(state['alpha'] - step_size * exp_avg / denom, group['min_scale'])
                p.data.mul_(state['alpha'] * state['init_norm'] / curr_norm)

        return loss

class Scale(nn.Module):
    def __init__(self):
        super(Scale, self).__init__()
        self.weight = nn.Parameter(torch.ones(1))
    def forward(self, x):
        return x * self.weight

class Bias(nn.Module):
    def __init__(self):
        super(Bias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        return x + self.bias

def get_ordered_params(net):
    param_list = []
    for m in net.modules():
        if isinstance(m, nn.Linear):
            param_list.append(m.weight)
            if m.bias is not None:
                param_list.append(m.bias)
        elif isinstance(m, Scale):
            param_list.append(m.weight)
        elif isinstance(m, Bias):
            param_list.append(m.bias)
    return param_list