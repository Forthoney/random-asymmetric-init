import torch
import model

net = model.make_1d_model(model.init_he_normal)

n_epochs = 10
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

for epoch in range(n_epochs):
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0): # Trainloader will be the training dataset
        x, y = data
        optimizer.zero_grad()

        y_pred = net(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')
