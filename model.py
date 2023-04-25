import torch

def make_1d_model(initializer, depth=10):
    modules = [torch.nn.Linear(1, 2), torch.nn.ReLU()]
    for _ in range(depth - 2):
        modules.append(torch.nn.Linear(2, 2))
        modules.append(torch.nn.ReLU())

    modules.extend([torch.nn.Linear(2, 1), torch.nn.ReLU(),])

    model = torch.nn.Sequential(*modules)
    model.apply(initializer)

def init_he_normal(layer: torch.nn.Module, bias=1):
    if not isinstance(layer, torch.nn.Linear): return

    torch.nn.init.kaiming_normal_(layer.weight)
    layer.bias.data.fill_(bias)

def init_rai(layer: torch.nn.Module):
    if not isinstance(layer, torch.nn.Linear): return

    beta_distribution = torch.distributions.Beta(torch.tensor([2]),
                                                 torch.tensor([1]))
    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(layer.weight)

    V = torch.randn(fan_out, fan_in + 1) * 0.6007 / fan_in**0.5
    for i in range(fan_out):
        j = torch.randint(0, fan_in + 1)
        V[i, j] = beta_distribution.sample()
    layer.weight = torch.transpose(V[:, :-1], 0, 1)
    layer.bias = V[:, -1]

make_1d_model(init_he_normal)