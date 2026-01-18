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
BATCH_SIZE = 64 # Turbo Mode: Maximize throughput
EPOCHS = 5 # Reduced for quick fine-tuning
LEARNING_RATE = 1e-4
SEQUENCE_LENGTH = 3 # Fastest temporal processing
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
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
            
            # IO OPTIMIZATION: Linear Read vs Random Seek
            # Seeking (cap.set) is extremely slow on some containers.
            # Faster approach: Read sequentially and pick frames.
            
            # Decide indices
            if total_frames > self.sequence_length:
                 # Uniform sample
                 indices = sorted(random.sample(range(total_frames), self.sequence_length))
            else:
                 indices = list(range(total_frames))
                 
            # Optimize: If indices are spread out, seeking might be needed. 
            # But for speed, let's try to just read and skip.
            
            extracted = []
            target_idx = 0
            current_idx = 0
            
            seq_idx = 0
            while seq_idx < len(indices):
                target = indices[seq_idx]
                
                # If target is far ahead, seeking might be worth it
                if target - current_idx > 20: 
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                    current_idx = target
                
                # Linear read until target
                while current_idx < target:
                    cap.grab() # Fast skip
                    current_idx += 1
                
                # We are at target
                ret, frame = cap.read()
                current_idx += 1
                if ret:
                    extracted.append(frame)
                else:
                    extracted.append(np.zeros((224, 224, 3), dtype=np.uint8))
                
                seq_idx += 1
                
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
    Robust Smart-Loader:
    - Scans directory recursively.
    - Infers label from FOLDER NAMES.
    - "real" in folder name -> Real (0)
    - "fake", "synthesis", "ai" in folder name -> Fake (1)
    """
    logger.info(f"Scanning dataset at: {data_path}")
    
    real_files = []
    fake_files = []
    
    skipped = 0
    
    for root, dirs, files in os.walk(data_path):
        # Fix: Check FULL path for keyword, not just immediate folder name
        # This handles nested folders like "real_train3/train 3"
        path_lower = root.lower()
        
        # Determine label from path
        label = None
        if "real" in path_lower:
            label = "REAL"
        elif any(x in path_lower for x in ["fake", "synthesis", "manipulated", "ai"]):
            label = "FAKE"
            
        if label:
            for file in files:
                if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(root, file)
                    if label == "REAL":
                        real_files.append(full_path)
                    else:
                        fake_files.append(full_path)
        else:
            # If root folder, ignore files
            pass

    # Fallback: if explicit folders exist inside
    if not real_files and not fake_files:
        logger.warning("Smart scan found nothing. Trying specific folder names 'real' and 'fake'...")
        # ... (logic could be added, but the walk above is usually sufficient if user named folders well)
        
    logger.info(f"Found {len(real_files)} Real samples.")
    logger.info(f"Found {len(fake_files)} Fake samples.")
    
    list_ids = real_files + fake_files
    labels = {}
    
    for f in real_files: labels[f] = 0
    for f in fake_files: labels[f] = 1
    
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
    # Reduced resolution for speed (128x128 is sufficient for deepfakes)
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
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
    
    # Optimized Data Loaders (Turbo Mode)
    # pin_memory=False for MPS to avoid overhead/warnings
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=False, prefetch_factor=2)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=False, prefetch_factor=2)
    
    # Model
    model = SentinelHybrid()
    
    # FINE-TUNING LOGIC: Load previous weights if they exist
    model_path = "sentinel_model.pth"
    if os.path.exists(model_path):
        logger.info(f"🔄 Component Found: {model_path}. Loading weights for Fine-Tuning...")
        try:
            state_dict = torch.load(model_path, map_location=model.device)
            model.load_state_dict(state_dict)
            logger.info("✅ Weights loaded successfully. Resuming training.")
        except Exception as e:
            logger.warning(f"⚠️ Could not load weights: {e}. Starting from scratch.")
    else:
        logger.info("🆕 No existing model found. Starting fresh training.")

    model.train()
    
    criterion = nn.CrossEntropyLoss()
    # Lower learning rate for fine-tuning to avoid destroying learned features
    optimizer = optim.Adam(model.parameters(), lr=1e-5) # Reduced from 1e-4
    
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
