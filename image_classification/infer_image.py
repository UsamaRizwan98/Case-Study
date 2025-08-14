import torch
from PIL import Image
import torchvision.transforms as transforms
import argparse
from models import VisionTransformer, SmallResNet

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

CLASSES = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']

def predict_image(path, ckpt_path):
    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    model_type = ckpt.get("model_type", "resnet")  # Default to resnet if missing
    num_classes = ckpt.get("model_config", {}).get("num_classes", 10)

    # Load model
    if model_type == "vit":
        model = VisionTransformer(num_classes=num_classes).to(device)
    else:
        model = SmallResNet(num_classes=num_classes).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Load and preprocess image
    img = Image.open(path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = probs.argmax(1).item()

    print(f"Predicted: {CLASSES[pred_idx]} (Prob: {probs[0][pred_idx].item():.4f})")
    return CLASSES[pred_idx]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--image_path", type=str, required=True, help="Path to image")
    args = parser.parse_args()

    predict_image(args.image_path, args.ckpt)
