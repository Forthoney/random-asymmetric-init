import torch

class LinearWithReLU(torch.nn.Module):
    def __init__(self, in_features, out_features, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear = torch.nn.Linear(in_features, out_features)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        return self.relu(x)


def init_he_normal(layer: torch.nn.Module, bias=1):
    if not isinstance(layer, torch.nn.Linear):
        return

    torch.nn.init.kaiming_normal_(layer.weight)
    layer.bias.data.fill_(bias)


def init_rai(layer: torch.nn.Module):
    if not isinstance(layer, torch.nn.Linear):
        return

    beta_distribution = torch.distributions.Beta(
        torch.tensor([2.0]), torch.tensor([1.0])
    )
    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(layer.weight)

    V = torch.randn(fan_out, fan_in + 1) * 0.6007 / fan_in**0.5
    for i in range(fan_out):
        j = torch.randint(low=0, high=fan_in + 1, size=(1,))
        V[i, j] = beta_distribution.sample()

    layer.weight = torch.nn.Parameter(torch.transpose(V[:, :-1], 0, 1))
    layer.bias = torch.nn.Parameter(V[:, -1])


def make_1d_model(initializer: callable, depth: int = 10, **kwargs) -> torch.nn.Module:
    layers = torch.nn.ModuleList(
        [
            LinearWithReLU(1, 2),
            *[LinearWithReLU(2, 2) for _ in range(depth - 2)],
            LinearWithReLU(2, 1),
        ]
    )

    model = torch.nn.Sequential(*layers)
    model.apply(lambda layer: initializer(layer, **kwargs))

    return model


model = make_1d_model(init_he_normal, bias=0)
print(model)