import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

import build_model
import sample


def train(model, true_fn, n_epochs, optimizer, device):
    criterion = torch.nn.MSELoss()
    x, y = sample.sample_1d(true_fn)
    x = np.expand_dims(x, 1)
    y = np.expand_dims(y, 1)

    tensor_x = torch.tensor(x, dtype=torch.float)
    tensor_y = torch.tensor(y, dtype=torch.float)
    dataloader = DataLoader(
        TensorDataset(tensor_x, tensor_y), batch_size=128, shuffle=True
    )

    for epoch in range(n_epochs):
        for features, targets in dataloader:  # Trainloader will be the training dataset
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            y_pred = model(features)
            loss = criterion(y_pred, targets)
            loss.backward()
            optimizer.step()

    return model


def check_dead(model, x_range, resolution, device):
    x_samples = torch.linspace(x_range[0], x_range[1], resolution).to(device)
    pred = model(x_samples.view(-1, 1)).detach()
    return torch.allclose(pred, pred[0])


def calc_stats(true_fn, x_range, repetitions, epochs, seed, init, device):
    rai_results = []
    for i in range(repetitions):
        torch.manual_seed(seed - i)
        model_rai = build_model.make_1d_model(init).to(device)
        optimizer_rai = torch.optim.Adam(model_rai.parameters(), lr=0.001)
        model_rai = train(model_rai, true_fn, epochs, optimizer_rai, device)
        rai_results.append(check_dead(model_rai, x_range, 20, device))
    return sum(rai_results) / len(rai_results)


rng = np.random.default_rng(seed=0)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

test_fns = [sample.f1, sample.f2, sample.f3]
x_range = (-np.sqrt(3), np.sqrt(3))
results = []
for func in test_fns:
    seed = rng.integers(2**32)
    results.append(
        (
            calc_stats(func, x_range, 100, 10, seed, build_model.init_he_normal, device),
            calc_stats(func, x_range, 100, 10, seed, build_model.init_rai, device),
        )
    )
    print('Finished')

with open("./output/results.txt", "w") as file:
    file.write(str(results))
