import matplotlib.pyplot as plt
from typing import Callable, Tuple
import torch

def visualize(f_true: Callable, model: torch.Module, x_range: Tuple[float, float], resolution: int = 100):
    x_samples = torch.linspace(x_range[0], x_range[1], resolution)

    plt.plot(x_samples.numpy(), f_true(x_samples.numpy()), label='True Function')
    plt.plot(x_samples.numpy(), model(x_samples), label='Model Prediction')
    plt.legend()
    plt.show()
