import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

torch.manual_seed(42)


class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 28 x 28 image = 784 pixel values
        self.layer1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(128, 10)

    def forward(self, x):
        # Convert 28x28 image into 784 numbers
        x = x.view(-1, 784)

        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)

        return x


transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

model = MNISTNet()

loss_function = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

print("Training...")

for epoch in range(3):

    for images, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(images)

        loss = loss_function(outputs, labels)

        loss.backward()

        optimizer.step()

    print(f"Epoch {epoch + 1} finished")


correct = 0
total = 0

model.eval()

with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        predictions = torch.argmax(outputs, dim=1)

        correct += (predictions == labels).sum().item()

        total += labels.size(0)


accuracy = correct / total

print(f"Test accuracy: {accuracy * 100:.2f}%")

# -----------------------------
# Export model weights
# -----------------------------

with open("models/mnist_weights.txt", "w") as f:

    # PyTorch stores Linear weights as:
    # [output_features, input_features]
    #
    # NanoInfer expects:
    # [input_features, output_features]
    #
    # So transpose them before exporting.

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

print("MNIST weights exported to models/mnist_weights.txt")


# -----------------------------
# Export one test image
# -----------------------------

test_image, test_label = test_dataset[0]

# Convert 28x28 image into 784 values
flat_image = test_image.view(-1)

with open("models/mnist_test_image.txt", "w") as f:
    for value in flat_image:
        f.write(f"{value.item()} ")

print("Test image exported to models/mnist_test_image.txt")
print("Actual digit:", test_label)


# Check what PyTorch predicts for that exact image
model.eval()

with torch.no_grad():
    output = model(test_image.unsqueeze(0))

    prediction = torch.argmax(output, dim=1).item()

print("PyTorch prediction:", prediction)