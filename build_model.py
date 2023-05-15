from typing import Callable, Optional

import torch


class LinearWithReLU(torch.nn.Module):
    """Linear layer followed by vanilla ReLU

    Attributes:
        linear: Linear layer
        relu: ReLU activation
    """

    def __init__(self, in_features, out_features, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear = torch.nn.Linear(in_features, out_features)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        return self.relu(x)


def init_he_normal(layer: torch.nn.Module, bias=1) -> None:
    """Initializes a given layer with He initialization scheme

    Args:
        bias (optional): Value to set the bias to
        layer: The layer to apply the initializaion to
    """
    if not isinstance(layer, torch.nn.Linear):
        return

    torch.nn.init.kaiming_normal_(layer.weight)
    layer.bias.data.fill_(bias)


def init_rai(layer: torch.nn.Module) -> None:
    """Initializes a given layer with random asymmetric initialization

    Args:
        layer: The layer to apply the initialization to
    """
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

    layer.weight = torch.nn.Parameter(V[:, :-1])
    layer.bias = torch.nn.Parameter(V[:, -1])


def make_1d_model(
    initializer: Optional[Callable], depth: int = 10, **kwargs
) -> torch.nn.Module:
    model = torch.nn.Sequential(
        LinearWithReLU(1, 2),
        *[LinearWithReLU(2, 2) for _ in range(depth - 2)],
        LinearWithReLU(2, 1),
    )

    if initializer != None:
        model.apply(lambda layer: initializer(layer, **kwargs))

    return model.float()


def make_mnist_model(
    initializer: Optional[Callable],
    depth: int = 5,
    hidden_size: int = 200,
    layer=LinearWithReLU,
    **kwargs
) -> torch.nn.Module:
    model = torch.nn.Sequential(
        layer(28**2, hidden_size),
        *[layer(hidden_size, hidden_size) for _ in range(depth - 2)],
        layer(hidden_size, 10),
    )

    if initializer != None:
        model.apply(lambda layer: initializer(layer, **kwargs))

    return model.float()
