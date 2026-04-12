# visualize_model.py   ← Place this in the main my_mnist_nn/ folder
import torch
import os
import warnings

warnings.filterwarnings("ignore")

from models.simple_nn import SimpleNN
from models.cnn import CNN
from config import config

# Create output folder
os.makedirs("runs/visualizations", exist_ok=True)


def visualize_model():
    # Load the model based on config
    if config.model_name == "cnn":
        model = CNN(num_classes=config.num_classes)
        model_type = "CNN"
    else:
        model = SimpleNN(num_classes=config.num_classes)
        model_type = "SimpleNN"

    model.to(config.device)
    model.eval()

    print(f"\n🔍 Visualizing {model_type} Model")
    print("=" * 60)

    # 1. Simple text summary (always works)
    print(model)
    print("\nTotal parameters:", sum(p.numel() for p in model.parameters()))
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # 2. Detailed summary with torchinfo (if installed)
    try:
        from torchinfo import summary # type: ignore
        print("\nDetailed Layer Summary:")
        summary(model, input_size=(1, 1, 28, 28), device=config.device)
    except ImportError:
        print("\n⚠️  torchinfo not installed. Install with: pip install torchinfo")

    # 3. Export to ONNX + launch Netron (best interactive visualization)
    # 3. Export to ONNX + launch Netron (best interactive visualization)
    try:
        import onnx  # type: ignore
        import onnxscript  # type: ignore   # this is the one that was missing

        dummy_input = torch.randn(1, 1, 28, 28).to(config.device)
        onnx_path = f"runs/visualizations/{config.model_name}.onnx"

        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=17,  # stable version
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"\n✅ ONNX model saved: {onnx_path}")

        print("🚀 Launching Netron viewer (interactive diagram in browser)...")
        import netron  # type: ignore
        netron.start(onnx_path, browse=True)

    except ImportError as e:
        print(f"\n❌ Missing dependency for ONNX export: {e}")
        print("   Fix with: pip install onnx onnxscript")
    except Exception as e:
        print(f"\n❌ Could not create Netron visualization: {e}")
        print("   You can still use the simple text summary printed above.")

if __name__ == "__main__":
    visualize_model()