from torchvision import datasets
from copy import deepcopy
import matplotlib.pyplot as plt
import torch

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

train_data = datasets.MNIST(root=".", download=False, train=True)
x_train = deepcopy(train_data.data)[:, :, :, None] / 255.
y_train = deepcopy(train_data.targets)

test_data = datasets.MNIST(root=".", download=False, train=False)
x_val = deepcopy(test_data.data)[:, :, :, None] / 255.
y_val = deepcopy(test_data.targets)

conv_layers = [
    torch.nn.Conv2d(
        in_channels=1,
        out_channels=6,
        kernel_size=5,
        padding='same',
        padding_mode='zeros'
    ),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(
        kernel_size=2
    ),
    torch.nn.Conv2d(
        in_channels=6,
        out_channels=16,
        kernel_size=5
    ),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(
        kernel_size=2
    ),
    torch.nn.Conv2d(
        in_channels=16,
        out_channels=120,
        kernel_size=5
    ),
    torch.nn.ReLU()
]

linear_layers = [
    torch.nn.Linear(
        in_features=120,
        out_features=84
    ),
    torch.nn.ReLU(),
    torch.nn.Linear(
        in_features=84,
        out_features=10
    ),
    torch.nn.Softmax(
        dim=-1
    )
]

layers = conv_layers + [torch.nn.Flatten()] + linear_layers

le_net = torch.nn.Sequential(*layers)

class LeNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.le_net = le_net

    def forward(self, x):
        return self.le_net(x.permute((0, 3, 1, 2))).log()

model = LeNet().to(dtype=x_train.dtype, device=device)

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    nesterov=True
)

loss_fn = torch.nn.NLLLoss()

losses = {"train": [], "val": []}

num_epochs = 5000
batch_size = 1000
val_every = 50

for epoch in range(1, num_epochs + 1):
    optimizer.zero_grad()

    batch_pos = torch.randint(high=x_train.shape[0], size=[batch_size])
    pred = model(x_train[batch_pos].to(device))
    loss = loss_fn(pred, y_train[batch_pos].to(device))

    loss.backward()
    optimizer.step()

with torch.no_grad():
    pred_sub = model(x_val.to(device)).max(dim=1).indices
    accuracy_score = (y_val.to(device) == pred_sub).to(dtype=torch.float).mean().item()
    print(accuracy_score)