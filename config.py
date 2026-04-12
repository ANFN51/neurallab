# config.py
import torch
import os


class Config:
    def __init__(self):
        self.batch_size = 64
        self.epochs = 10
        self.learning_rate = 0.001

        # Device (GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model selection: change to "cnn" when you want the better model
        self.model_name = "cnn"
        # Save path - now safely created
        self.save_path = os.path.join("runs", f"{self.model_name}_best.pth")

        self.num_classes = 10


# Global config instance
config = Config()