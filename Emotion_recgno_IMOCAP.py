# ======================================
# IEMOCAP Emotion Recognition Model
# ======================================

import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from tqdm import tqdm
import logging

# =====================================
# Configuration and Setup
# =====================================

# File paths and directories
EXCEL_PATH = "/content/drive/MyDrive/EANN/dataset/IMOCAP/Filtered_IMOCAP.xlsx"
RESULT_DIR = "/content/drive/MyDrive/EANN/IEMOCAP_RESULT"
MODEL_PATH = os.path.join(RESULT_DIR, "IEMOCAP_bestmodel.pth")

# Create result directory if it doesn't exist
os.makedirs(RESULT_DIR, exist_ok=True)

# Set up logging
log_file = os.path.join(RESULT_DIR, 'training.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def log_and_print(message):
    """Utility function to log message and print to console"""
    logger.info(message)
    print(message)

# Early stopping configuration
early_stop = "off"  # Set to "on" to enable early stopping
log_and_print(f"Early stopping is set to: {early_stop}")

# Set environment variables for CUDA
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
log_and_print("Environment variables set for CUDA debugging")

# CUDA setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
log_and_print(f"Using device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
log_and_print("Random seeds set for reproducibility")

# =====================================
# Hyperparameters
# =====================================

HYPERPARAMS = {
    'batch_size': 16,
    'accumulation_steps': 4,
    'max_length': 128,
    'num_epochs': 100,
    'patience': 5,
    'learning_rate': 1e-5
}

log_and_print("Hyperparameters:")
for key, value in HYPERPARAMS.items():
    log_and_print(f"  {key}: {value}")

# =====================================
# Data Loading and Preprocessing
# =====================================

# Load dataset
try:
    df = pd.read_excel(EXCEL_PATH)
    log_and_print(f"Data loaded successfully from {EXCEL_PATH}")
    log_and_print(f"Dataset shape: {df.shape}")
except Exception as e:
    log_and_print(f"Error loading data: {e}")
    sys.exit(1)

# Convert string representations to lists
for col in ['V1', 'V2', 'V3', 'V4', 'A2']:
    df[col] = df[col].apply(ast.literal_eval)
log_and_print("String representations converted to lists")

# Convert utterances to strings
df['Utterance'] = df['Utterance'].astype(str)

# Encode emotion labels
le = LabelEncoder()
df['Emotion_encoded'] = le.fit_transform(df['Emotion'])
log_and_print("\nEmotion label distribution:")
for emotion, count in df['Emotion'].value_counts().items():
    log_and_print(f"  {emotion}: {count}")

# Prepare features and labels
X = df[['Utterance', 'V1', 'V3', 'V4', 'A2']]
y = df['Emotion_encoded']

# Handle class imbalance through oversampling
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)
log_and_print("\nClass distribution after oversampling:")
for emotion, count in pd.Series(y_resampled).value_counts().items():
    log_and_print(f"  Class {emotion}: {count}")

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, 
    y_resampled, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_resampled
)
log_and_print(f"\nData split: Train size: {len(X_train)}, Test size: {len(X_test)}")

# =====================================
# BERT Tokenizer Setup
# =====================================

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
log_and_print("BERT tokenizer initialized")

def tokenize_text(text, max_length=HYPERPARAMS['max_length']):
    """
    Tokenize input text using BERT tokenizer
    Args:
        text: Input text to tokenize
        max_length: Maximum sequence length
    Returns:
        Tokenized text with padding and attention mask
    """
    return tokenizer(
        text, 
        padding='max_length', 
        truncation=True, 
        max_length=max_length, 
        return_tensors='pt'
    )

# =====================================
# Dataset Class
# =====================================

class IEMOCAPDataset(Dataset):
    """
    Custom Dataset class for IEMOCAP emotion recognition data
    """
    def __init__(self, utterances, v1, v3, v4, a2, labels):
        self.utterances = utterances
        self.v1 = v1
        self.v3 = v3
        self.v4 = v4
        self.a2 = a2
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        utterance = self.utterances.iloc[idx]
        tokenized = tokenize_text(utterance)

        return {
            'input_ids': tokenized['input_ids'].squeeze(),
            'attention_mask': tokenized['attention_mask'].squeeze(),
            'v1': torch.tensor(self.v1.iloc[idx], dtype=torch.float32),
            'v3': torch.tensor(self.v3.iloc[idx], dtype=torch.float32),
            'v4': torch.tensor(self.v4.iloc[idx], dtype=torch.float32),
            'a2': torch.tensor(self.a2.iloc[idx], dtype=torch.float32),
            'label': torch.tensor(self.labels.iloc[idx], dtype=torch.long)
        }

# Create datasets and dataloaders
train_dataset = IEMOCAPDataset(
    X_train['Utterance'], X_train['V1'], X_train['V3'],
    X_train['V4'], X_train['A2'], y_train
)
test_dataset = IEMOCAPDataset(
    X_test['Utterance'], X_test['V1'], X_test['V3'],
    X_test['V4'], X_test['A2'], y_test
)

train_loader = DataLoader(
    train_dataset, 
    batch_size=HYPERPARAMS['batch_size'],
    shuffle=True, 
    num_workers=0
)
test_loader = DataLoader(
    test_dataset, 
    batch_size=HYPERPARAMS['batch_size'],
    shuffle=False, 
    num_workers=0
)

log_and_print(f"DataLoaders created successfully")
log_and_print(f"Number of training batches: {len(train_loader)}")
log_and_print(f"Number of test batches: {len(test_loader)}")


# =====================================
# Model Architecture
# =====================================

class MultimodalEmotionRecognition(nn.Module):
    """
    Multimodal Emotion Recognition model for IEMOCAP dataset.
    Combines text (BERT), audio, and visual features with cross-modal attention.
    """
    def __init__(self, num_classes=9):
        super(MultimodalEmotionRecognition, self).__init__()

        # Text Processing (BERT)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.text_fc = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Audio Processing
        # Process V1 features
        self.audio_v1_fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )
        # Process V3 and V4 features
        self.audio_v34_fc = nn.Sequential(
            nn.Linear(50, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        # Combine audio features
        self.audio_combined_fc = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Visual Processing
        self.visual_fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3)
        )

        # Cross-Modal Attention
        self.text_audio_attention = nn.MultiheadAttention(
            embed_dim=512, 
            num_heads=8, 
            batch_first=True
        )
        self.text_visual_attention = nn.MultiheadAttention(
            embed_dim=512, 
            num_heads=8, 
            batch_first=True
        )

        # Multimodal Fusion
        self.fusion_fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Classification Layer
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask, v1, v3, v4, a2):
        """
        Forward pass of the model.
        
        Args:
            input_ids (tensor): BERT input token IDs
            attention_mask (tensor): BERT attention mask
            v1 (tensor): First audio feature vector
            v3 (tensor): Second audio feature vector
            v4 (tensor): Third audio feature vector
            a2 (tensor): Visual feature vector
        
        Returns:
            tensor: Logits for each emotion class
        """
        # Text Processing
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = self.text_fc(bert_output.pooler_output)

        # Audio Processing
        audio_v1 = self.audio_v1_fc(v1)
        audio_v34 = self.audio_v34_fc(torch.cat([v3, v4], dim=1))
        audio_features = self.audio_combined_fc(torch.cat([audio_v1, audio_v34], dim=1))

        # Visual Processing
        visual_features = self.visual_fc(a2)

        # Cross-Modal Attention
        # Expand dimensions for attention mechanism
        text_features_expanded = text_features.unsqueeze(1)
        audio_features_expanded = nn.functional.pad(audio_features.unsqueeze(1), (0, 256))
        visual_features_expanded = nn.functional.pad(visual_features.unsqueeze(1), (0, 256))

        # Apply attention
        text_audio_attention, _ = self.text_audio_attention(
            text_features_expanded, 
            audio_features_expanded, 
            audio_features_expanded
        )
        text_visual_attention, _ = self.text_visual_attention(
            text_features_expanded, 
            visual_features_expanded, 
            visual_features_expanded
        )

        # Combine attended features
        attended_features = (
            text_features + 
            text_audio_attention.squeeze(1) + 
            text_visual_attention.squeeze(1)
        )

        # Multimodal Fusion
        fused_features = self.fusion_fc(
            torch.cat([attended_features, audio_features, visual_features], dim=1)
        )

        # Classification
        logits = self.classifier(fused_features)

        return logits


# =====================================
# Loss Function
# =====================================

class FocalLoss(nn.Module):
    """
    Focal Loss implementation for handling class imbalance.
    
    Args:
        alpha (tensor): Class weight tensor
        gamma (float): Focusing parameter for hard examples
    """
    def __init__(self, alpha, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha.to(device)
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        Calculate the focal loss.
        
        Args:
            inputs (tensor): Model predictions
            targets (tensor): Ground truth labels
        
        Returns:
            tensor: Calculated focal loss
        """
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.clamp(torch.exp(-ce_loss), min=1e-8, max=1-1e-8)  # Add numerical stability
        focal_loss = self.alpha[targets] * (1 - pt)**self.gamma * ce_loss
        return focal_loss.mean()


# =====================================
# Visualization Functions
# =====================================

def save_plot(plt, filename):
    """
    Save plot to file and display it.
    
    Args:
        plt: matplotlib.pyplot object
        filename (str): Name of the file to save
    """
    # Create full path
    filepath = os.path.join(RESULT_DIR, filename)
    
    # Save plot
    plt.savefig(filepath, bbox_inches='tight', dpi=300)
    
    # Display plot
    plt.show()
    
    # Close plot to free memory
    plt.close()
    
    log_and_print(f"Plot saved to {filepath}")

def plot_training_progress(train_losses, val_losses, train_accuracies, val_accuracies, epoch):
    """
    Plot and save training progress for current epoch.
    
    Args:
        train_losses (list): List of training losses
        val_losses (list): List of validation losses
        train_accuracies (list): List of training accuracies
        val_accuracies (list): List of validation accuracies
        epoch (int): Current epoch number
    """
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.tight_layout()
    save_plot(plt, f'training_progress_epoch_{epoch}.png')

def plot_final_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    """
    Plot and save final training curves.
    
    Args:
        train_losses (list): List of training losses
        val_losses (list): List of validation losses
        train_accuracies (list): List of training accuracies
        val_accuracies (list): List of validation accuracies
    """
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Final Training and Validation Loss')

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Final Training and Validation Accuracy')

    plt.tight_layout()
    save_plot(plt, 'final_training_curves.png')



    # =====================================
# Training Functions
# =====================================

def train_model(model, train_loader, val_loader, num_epochs=HYPERPARAMS['num_epochs'], 
                patience=HYPERPARAMS['patience']):
    """
    Train the emotion recognition model.
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Maximum number of training epochs
        patience: Number of epochs to wait before early stopping
    
    Returns:
        model: Trained model
    """
    log_and_print("\nStarting model training...")
    log_and_print(f"Early stopping is {early_stop}")

    # Setup class weights for Focal Loss
    class_counts = torch.tensor([411] * 9)  # Number of samples per class after oversampling
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()

    # Initialize loss function, optimizer and scheduler
    criterion = FocalLoss(alpha=class_weights.to(device), gamma=2)
    optimizer = optim.AdamW(model.parameters(), 
                           lr=HYPERPARAMS['learning_rate'], 
                           weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    # Training tracking variables
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        optimizer.zero_grad()

        start_time = time.time()

        # Training loop
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            v1 = batch['v1'].to(device)
            v3 = batch['v3'].to(device)
            v4 = batch['v4'].to(device)
            a2 = batch['a2'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask, v1, v3, v4, a2)
            loss = criterion(outputs, labels)
            loss = loss / HYPERPARAMS['accumulation_steps']

            # Backward pass
            loss.backward()

            # Gradient accumulation
            if (i + 1) % HYPERPARAMS['accumulation_steps'] == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Update metrics
            train_loss += loss.item() * HYPERPARAMS['accumulation_steps']
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            # Log batch progress
            if i % 10 == 0:
                log_and_print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f}")

        # Calculate training metrics
        train_loss /= len(train_loader)
        train_accuracy = train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                v1 = batch['v1'].to(device)
                v3 = batch['v3'].to(device)
                v4 = batch['v4'].to(device)
                a2 = batch['a2'].to(device)
                labels = batch['label'].to(device)

                # Forward pass
                outputs = model(input_ids, attention_mask, v1, v3, v4, a2)
                loss = criterion(outputs, labels)

                # Update metrics
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        # Calculate validation metrics
        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Print epoch statistics
        epoch_time = time.time() - start_time
        log_and_print(f'\nEpoch {epoch+1}/{num_epochs}:')
        log_and_print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        log_and_print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        log_and_print(f'Epoch Time: {epoch_time:.2f} seconds')

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Model checkpoint saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), MODEL_PATH)
            log_and_print("New best model saved.")
        else:
            epochs_without_improvement += 1
            if early_stop == "on" and epochs_without_improvement == patience:
                log_and_print(f'Early stopping triggered after {epoch+1} epochs')
                break

        # Plot training progress
        plot_training_progress(train_losses, val_losses, train_accuracies, val_accuracies, epoch+1)

    log_and_print("Training completed.")

    # Plot final training curves
    plot_final_curves(train_losses, val_losses, train_accuracies, val_accuracies)

    return model


# =====================================
# Evaluation Functions
# =====================================

def evaluate_model(model, test_loader):
    """
    Evaluate the trained model.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
    """
    log_and_print("\nStarting model evaluation...")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluation"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            v1 = batch['v1'].to(device)
            v3 = batch['v3'].to(device)
            v4 = batch['v4'].to(device)
            a2 = batch['a2'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask, v1, v3, v4, a2)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Generate and save classification report
    report = classification_report(all_labels, all_preds, 
                                 target_names=le.classes_, 
                                 output_dict=True)
    
    # Print classification report
    log_and_print("\nClassification Report:")
    log_and_print("\n" + classification_report(all_labels, all_preds, target_names=le.classes_))

    # Save classification report to file
    report_path = os.path.join(RESULT_DIR, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(classification_report(all_labels, all_preds, target_names=le.classes_))
    log_and_print(f"Classification report saved to {report_path}")

    # Generate and save confusion matrix
    plot_confusion_matrix(all_labels, all_preds, le.classes_)
    
    # Generate and save per-class metrics
    plot_per_class_metrics(report)

    log_and_print("Evaluation completed. All results saved in: " + RESULT_DIR)

# =====================================
# Main Execution
# =====================================

if __name__ == "__main__":
    try:
        # Initialize model
        model = MultimodalEmotionRecognition()
        model = model.to(device)
        log_and_print("Model initialized successfully")
        log_and_print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

        # Debug information
        log_and_print("\nDebug information:")
        batch = next(iter(train_loader))
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        v1 = batch['v1'].to(device)
        v3 = batch['v3'].to(device)
        v4 = batch['v4'].to(device)
        a2 = batch['a2'].to(device)

        # Print shapes for debugging
        log_and_print(f"Input shapes:")
        log_and_print(f"  Input IDs: {input_ids.shape}")
        log_and_print(f"  Attention Mask: {attention_mask.shape}")
        log_and_print(f"  V1: {v1.shape}")
        log_and_print(f"  V3: {v3.shape}")
        log_and_print(f"  V4: {v4.shape}")
        log_and_print(f"  A2: {a2.shape}")

        # Test forward pass
        with torch.no_grad():
            outputs = model(input_ids, attention_mask, v1, v3, v4, a2)
            log_and_print(f"Model output shape: {outputs.shape}")

        # Train model
        trained_model = train_model(model, train_loader, test_loader)

        # Evaluate model
        evaluate_model(trained_model, test_loader)

        log_and_print("Script execution completed successfully")
        
    except Exception as e:
        log_and_print(f"An error occurred: {str(e)}")
        import traceback
        log_and_print(traceback.format_exc())
