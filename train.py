# train.py  (updated version)
import os

import torch
import torch.nn as nn
import torch.optim as optim

os.makedirs("runs", exist_ok=True)  # Creates the runs folder automatically
print(f" Runs folder is ready at: {os.path.abspath('runs')}")
from models.simple_nn import SimpleNN
from models.cnn import CNN
from utils.data_loader import get_data_loaders
from utils.visualize import plot_training_curves, plot_confusion_matrix
from config import config

# === NEW: Create runs directory automatically ===
os.makedirs("runs", exist_ok=True)
print(f" Runs directory ready: {os.path.abspath('runs')}")

# Choose model
if config.model_name == "cnn":
    model = CNN(num_classes=config.num_classes)
else:
    model = SimpleNN(num_classes=config.num_classes)

model.to(config.device)

train_loader, test_loader = get_data_loaders(config.batch_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

train_losses = []
test_accuracies = []
best_acc = 0.0

print(f"Starting training on {config.device} for {config.epochs} epochs...")

for epoch in range(config.epochs):
    # Training
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(config.device), labels.to(config.device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(config.device), labels.to(config.device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)

    print(f"Epoch {epoch + 1}/{config.epochs} | Loss: {avg_loss:.4f} | Test Acc: {accuracy:.2f}%")

    # Save best model
    if accuracy > best_acc:
        best_acc = accuracy
        torch.save(model.state_dict(), config.save_path)
        print(f"New best model saved! Accuracy: {best_acc:.2f}%")

# Final plots
plot_training_curves(train_losses, test_accuracies)
plot_confusion_matrix(all_labels, all_preds)

print(f"Training finished! Best accuracy: {best_acc:.2f}%")
print(f"Model saved to: {config.save_path}")
