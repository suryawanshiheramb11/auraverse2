import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
import os
import glob
import logging
import random
from model_core import SentinelHybrid

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SentinelTrainer")

# Configuration
BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 1e-4
SEQUENCE_LENGTH = 5
DATASET_PATH = "/Users/suryawanshiheramb11/Downloads/training data"
DASHBOARD_PATH = "training_dashboard.md"

def update_dashboard(epoch, step, total_steps, loss, acc=None, status="Running"):
    """Writes a live dashboard to a markdown file."""
    with open(DASHBOARD_PATH, "w") as f:
        f.write(f"# Sentinel Training Dashboard\n\n")
        f.write(f"**Status**: {status} 🟢\n")
        f.write(f"**Last Update**: Epoch {epoch}/{EPOCHS} | Step {step}/{total_steps}\n\n")
        f.write(f"## Metrics\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"| :--- | :--- |\n")
        f.write(f"| **Loss** | `{loss:.4f}` |\n")
        if acc:
             f.write(f"| **Accuracy** | `{acc:.2f}%` |\n")
        f.write(f"\n> *This file is automatically updated by the training script.*\n")

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, list_ids, labels, transform=None, sequence_length=10):
        self.root_dir = root_dir
        self.list_ids = list_ids
        self.labels = labels
        self.transform = transform
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.list_ids)
    
    def __getitem__(self, index):
        # ID is the filename
        current_id = self.list_ids[index]
        label = self.labels[current_id]
        file_path = current_id
        
        frames = []
        
        # Check if video or image
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
             # Image: Repeat
             img = cv2.imread(file_path)
             if img is None:
                 # Return zero tensor if fail
                 return torch.zeros((self.sequence_length, 3, 224, 224)), label
             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
             for _ in range(self.sequence_length):
                 frames.append(img)
        else:
            # Video: Extract frames
            cap = cv2.VideoCapture(file_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames <= 0:
                 # Fallback if metadata fails
                 total_frames = 100 
            
            # Simple uniform sampling
            if total_frames > self.sequence_length:
                # Sample 'sequence_length' frames uniformly
                indices = sorted(random.sample(range(total_frames), self.sequence_length))
            else:
                # Take all and pad
                indices = list(range(total_frames))
            
            # FAST RANDOM ACCESS (Turbo Mode)
            extracted = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    extracted.append(frame)
                else:
                    # If frame read fails, append black frame
                    extracted.append(np.zeros((224, 224, 3), dtype=np.uint8))
            cap.release()
            
            # Convert BGR to RGB
            for f in extracted:
                frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                
            # Padding if video too short or failed reads
            while len(frames) < self.sequence_length:
                 if len(frames) > 0:
                     frames.append(frames[-1]) # Duplicate last valid
                 else:
                     # Completely failed video
                     frames.append(np.zeros((224, 224, 3), dtype=np.uint8))

        # Transform
        processed_frames = []
        for f in frames:
            if self.transform:
                processed_frames.append(self.transform(f))
            else:
                processed_frames.append(transforms.ToTensor()(f))
                
        # Stack: (Seq_Len, 3, 224, 224)
        data = torch.stack(processed_frames)
        
        return data, label

def load_data(data_path):
    """
    Scans for subsets recursively:
    - YouTube-real (Real)
    - Celeb-real (Real)
    - Celeb-synthesis (Fake)
    """
    real_files = []
    fake_files = []
    
    # Define known folders
    real_folders = ["YouTube-real", "Celeb-real"]
    fake_folders = ["Celeb-synthesis"]
    
    # Scan Real
    for folder in real_folders:
        path = os.path.join(data_path, folder)
        if os.path.exists(path):
            # Recursive scan for mp4 and jpg
            found = glob.glob(os.path.join(path, "**/*.mp4"), recursive=True) + \
                    glob.glob(os.path.join(path, "**/*.jpg"), recursive=True)
            real_files.extend(found)
            logger.info(f"Found {len(found)} Real samples in {folder}")
            
    # Scan Fake
    for folder in fake_folders:
        path = os.path.join(data_path, folder)
        if os.path.exists(path):
            found = glob.glob(os.path.join(path, "**/*.mp4"), recursive=True) + \
                    glob.glob(os.path.join(path, "**/*.jpg"), recursive=True)
            fake_files.extend(found)
            logger.info(f"Found {len(found)} Fake samples in {folder}")

    list_ids = real_files + fake_files
    labels = {}
    
    for f in real_files:
        labels[f] = 0 # 0 for Real
    for f in fake_files:
        labels[f] = 1 # 1 for Fake
        
    logger.info(f"Total: {len(real_files)} Real, {len(fake_files)} Fake.")
    
    # Shuffle
    random.shuffle(list_ids)
    
    return list_ids, labels

def train():
    logger.info("Initializing Training Environment...")
    
    # Initialize Dashboard
    with open(DASHBOARD_PATH, "w") as f:
        f.write("# Sentinel Training Dashboard\n\n**Status**: Initializing... 🟡")

    # Check if data exists
    if not os.path.exists(DATASET_PATH):
        logger.error(f"Dataset path '{DATASET_PATH}' not found. Please create 'dataset/real' and 'dataset/fake' folders.")
        return

    # Transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Data Loaders
    list_ids, labels = load_data(DATASET_PATH)
    if not list_ids:
        logger.error("No data found!")
        return
        
    # Split 80/20
    split = int(0.8 * len(list_ids))
    train_ids = list_ids[:split]
    val_ids = list_ids[split:]
    
    train_set = DeepfakeDataset(DATASET_PATH, train_ids, labels, transform=train_transform, sequence_length=SEQUENCE_LENGTH)
    val_set = DeepfakeDataset(DATASET_PATH, val_ids, labels, transform=train_transform, sequence_length=SEQUENCE_LENGTH) 
    
    # 0 workers for stability
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Model
    model = SentinelHybrid()
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    logger.info(f"Starting Training for {EPOCHS} epochs on {model.device}...")
    update_dashboard(0, 0, len(train_loader), 0.0, status="Starting Training Loop")

    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training Loop
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(model.device)
            labels = labels.to(model.device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                update_dashboard(epoch+1, i+1, len(train_loader), loss.item())

        epoch_acc = 100 * correct / total
        logger.info(f"Epoch [{epoch+1}/{EPOCHS}] Train Acc: {epoch_acc:.2f}%")
        
        # Validation Loop
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
             for inputs, labels in val_loader:
                inputs = inputs.to(model.device)
                labels = labels.to(model.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        logger.info(f"Epoch [{epoch+1}/{EPOCHS}] Val Acc: {val_acc:.2f}%")
        
        # Save & Final Update for Epoch
        torch.save(model.state_dict(), "sentinel_model.pth")
        update_dashboard(epoch+1, len(train_loader), len(train_loader), running_loss/len(train_loader), acc=val_acc)
        logger.info("Model saved to sentinel_model.pth")

if __name__ == "__main__":
    train()
