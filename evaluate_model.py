import torch
from torch.utils.data import DataLoader
from train_model import DeepfakeDataset, load_data, DATASET_PATH, SEQUENCE_LENGTH, BATCH_SIZE
from model_core import SentinelHybrid
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

def evaluate():
    print("Initializing Evaluation...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load Data
    print("Loading Dataset...")
    list_ids, labels = load_data(DATASET_PATH)
    
    # Use 20% for validation/testing
    split = int(0.8 * len(list_ids))
    val_ids = list_ids[split:]
    
    # Define Transforms (Must match training)
    from torchvision import transforms
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create Dataset
    val_set = DeepfakeDataset(DATASET_PATH, val_ids, labels, transform=val_transform, sequence_length=SEQUENCE_LENGTH)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Load Model
    print("Loading Model...")
    model = SentinelHybrid()
    model.to(device)
    
    if os.path.exists("sentinel_model.pth"):
        model.load_state_dict(torch.load("sentinel_model.pth", map_location=device))
        print("Model loaded successfully.")
    else:
        print("Error: No model found.")
        return

    model.eval()
    
    all_preds = []
    all_labels = []

    print(f"Evaluating on {len(val_ids)} samples...")
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.numpy())
            
            if i % 10 == 0:
                print(f"Processed batch {i}...")

    # Calculate Metrics
    print("\n" + "="*30)
    print("CLASSIFICATION REPORT")
    print("="*30)
    
    # 0 = Real, 1 = Fake
    target_names = ['Real', 'Fake']
    report = classification_report(all_labels, all_preds, target_names=target_names)
    print(report)
    
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    tn, fp, fn, tp = cm.ravel()
    print(f"\nSpecifics:")
    print(f"True Negatives (Correct Real): {tn}")
    print(f"False Positives (Real flagged as Fake): {fp}")
    print(f"False Negatives (Fake flagged as Real): {fn}")
    print(f"True Positives (Correct Fake): {tp}")

if __name__ == "__main__":
    evaluate()
