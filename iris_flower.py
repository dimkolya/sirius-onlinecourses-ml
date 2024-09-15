import torch
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from IPython.display import clear_output

data = datasets.load_iris()
X = torch.tensor(data['data'])
y = torch.tensor(data['target'], dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# X_train = X
# y_train = y
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=0)

batch_size = 10

n = X_train.size()[1]
hidden = 4
k = 3
model = torch.nn.Sequential(
    torch.nn.Linear(n, hidden),
    torch.nn.Tanh(),
    torch.nn.Linear(hidden, k),
    torch.nn.LogSoftmax(dim=1)
)
model.to(dtype=X.dtype)

# fig, ax = plt.subplots(figsize=(30, 10))
# colors = ["red", "green", 'blue']
# for target in range(3):
#     plt.scatter(X[y == target, 0], X[y == target, 2], label=f"Класс {target}", c=colors[target])
# plt.show()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
)
loss_function = torch.nn.NLLLoss()
losses = []
for epoch in range(1, 1000 + 1):
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = loss_function(y_pred, y_train)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 1000 == 0:
        clear_output(True)
        fig, ax = plt.subplots(figsize=(30, 10))
        plt.title("График ошибки")
        plt.plot(losses, ".-")
        plt.xlabel("Итерация обучения")
        plt.ylabel("Значение ошибки")
        plt.yscale("log")
        plt.grid()
        # plt.show()

log_probs = model(X_test)
y_pred_probs = log_probs.exp()
y_pred = torch.argmax(log_probs, dim=1)
print(y_pred.tolist())
