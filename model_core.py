import torch
import torch.nn as nn
import timm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SentinelCore")

class SentinelHybrid(nn.Module):
    def __init__(self, sequence_length=10, hidden_size=128, num_classes=2):
        super(SentinelHybrid, self).__init__()
        
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        
        logger.info("Initializing SentinelHybrid Model...")
        
        # Backbone: EfficientNet B4
        # We remove the classifier head to get features
        logger.info("Loading EfficientNet-B4 backbone (pretrained)...")
        self.backbone = timm.create_model('efficientnet_b4', pretrained=True, num_classes=0)
        
        # Get feature dimension of the backbone
        # EfficientNet-B4 usually outputs 1792 features before the classifier
        self.feature_dim = self.backbone.num_features
        logger.info(f"Backbone feature dimension: {self.feature_dim}")
        
        # Temporal Layer: LSTM
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        # Device Optimization for Mac
        self.device = self._detect_device()
        self.to(self.device)
        logger.info(f"Model deployed on: {self.device}")

    def _detect_device(self):
        """
        Automatically detect and assign device.
        Prioritizes 'mps' for Apple Silicon.
        """
        if torch.backends.mps.is_available():
            logger.info("Apple Silicon MPS acceleration detected.")
            return torch.device("mps")
        elif torch.cuda.is_available():
            logger.info("CUDA GPU detected.")
            return torch.device("cuda")
        else:
            logger.warning("No GPU acceleration detected. Falling back to CPU.")
            return torch.device("cpu")

    def forward(self, x):
        """
        Input shape: (Batch, Sequence_Len, 3, 224, 224)
        """
        batch_size, seq_len, c, h, w = x.shape
        
        # Reshape for backbone: (Batch * Sequence_Len, 3, 224, 224)
        x_reshaped = x.view(batch_size * seq_len, c, h, w)
        
        # Extract features
        features = self.backbone(x_reshaped) # Shape: (Batch * Seq_Len, Feature_Dim)
        
        # Reshape for LSTM: (Batch, Sequence_Len, Feature_Dim)
        features_seq = features.view(batch_size, seq_len, -1)
        
        # Temporal processing
        # LSTM output: (Batch, Seq_Len, Hidden_Size)
        # We only care about the final hidden state or output for classification
        # Here we take the output of the last time step
        lstm_out, _ = self.lstm(features_seq)
        last_out = lstm_out[:, -1, :] # Shape: (Batch, Hidden_Size)
        
        # Classification
        logits = self.classifier(last_out)
        
        return logits

def get_model():
    """Factory function to get the model instance."""
    return SentinelHybrid()

if __name__ == "__main__":
    # Smoke test
    model = get_model()
    dummy_input = torch.randn(2, 10, 3, 224, 224).to(model.device)
    output = model(dummy_input)
    print(f"Smoke Test Output Shape: {output.shape}")
    print("Model loaded successfully.")
