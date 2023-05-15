import matplotlib.pyplot as plt
from typing import Callable, Tuple
import torch


def visualize(
    f_true: Callable,
    model: torch.nn.Module,
    x_samples: torch.Tensor,
):
    plt.plot(x_samples.numpy(), f_true(x_samples.numpy()), label="True Function")
    plt.plot(
        x_samples.numpy(),
        model(x_samples.view(-1, 1)).detach().numpy(),
        label="Model Prediction",
    )
    plt.legend()
    plt.show()
