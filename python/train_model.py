import torch
import torch.nn as nn
import torch.optim as optim
torch.manual_seed(42)

# Tiny network:
# 2 inputs -> 3 hidden neurons -> 2 outputs
class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(2, 4)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(4, 2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


model = TinyNet()

# Tiny training dataset
# Each row has 2 input numbers
X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

# Class labels
y = torch.tensor([
    0,
    1,
    1,
    0
])

loss_function = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.01
)

for epoch in range(3000):
    optimizer.zero_grad()

    output = model(X)

    loss = loss_function(output, y)

    loss.backward()

    optimizer.step()

print("Training finished")

with torch.no_grad():
    predictions = model(X)

    print("Raw outputs:")
    print(predictions)

    print("Predicted classes:")
    print(torch.argmax(predictions, dim=1))

    with open("models/xor_weights.txt", "w") as f:
    # PyTorch stores Linear weights as:
    # [output_features, input_features]
    #
    # NanoInfer expects:
    # [input_features, output_features]
    #
    # So transpose before exporting.

        layer1_weights = model.layer1.weight.detach().t().flatten()
        layer1_bias = model.layer1.bias.detach().flatten()

        layer2_weights = model.layer2.weight.detach().t().flatten()
        layer2_bias = model.layer2.bias.detach().flatten()

        for value in layer1_weights:
            f.write(f"{value.item()} ")

        f.write("\n")

        for value in layer1_bias:
            f.write(f"{value.item()} ")

        f.write("\n")

        for value in layer2_weights:
            f.write(f"{value.item()} ")

        f.write("\n")

        for value in layer2_bias:
            f.write(f"{value.item()} ")

    print("Weights exported to models/xor_weights.txt")