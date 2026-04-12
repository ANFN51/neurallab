# evaluate.py
import torch
from models.simple_nn import SimpleNN
from models.cnn import CNN
from utils.data_loader import get_data_loaders
from config import config

model = CNN() if config.model_name == "cnn" else SimpleNN()
model.load_state_dict(torch.load(config.save_path, weights_only=True, map_location=config.device))
model.to(config.device)
model.eval()

_, test_loader = get_data_loaders(config.batch_size)

correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(config.device), labels.to(config.device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Final Test Accuracy: {100 * correct / total:.2f}%")