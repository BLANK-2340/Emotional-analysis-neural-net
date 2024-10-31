# ======================================
# Emotion Recognition Model
# Author: Armaan singh 
# Last Updated: October 2024
# ======================================

# Import required libraries
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
from torch.cuda.amp import autocast, GradScaler
import time
from tqdm import tqdm
import os
import GPUtil

# =====================================
# Configuration and Setup
# =====================================

# File paths and directories
EXCEL_PATH = "/content/drive/MyDrive/EANN/dataset/MELD/MELD.xlsx"
RESULT_DIR = "/content/drive/MyDrive/EANN/MELD_result"
MODEL_PATH = os.path.join(RESULT_DIR, "MELD_bestmodel.pth")

# Create result directory if it doesn't exist
os.makedirs(RESULT_DIR, exist_ok=True)

# Early stopping configuration
early_stop = "off"  # Set to "off" to disable early stopping

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configure CUDA device and optimize operations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
print(f"Using device: {device}")

# =====================================
# Utility Functions
# =====================================

def print_gpu_usage():
    """
    Print current GPU memory usage and utilization.
    """
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]
        print(f"GPU Memory Usage: {gpu.memoryUsed} / {gpu.memoryTotal} MB")
        print(f"GPU Memory Utilization: {gpu.memoryUtil * 100:.2f}%")

def save_plot(plt, filename):
    """
    Save plot both to file and display in notebook.
    Args:
        plt: matplotlib.pyplot object
        filename: name of the file to save
    """
    # Display in notebook
    plt.show()
    
    # Save to file
    filepath = os.path.join(RESULT_DIR, filename)
    plt.savefig(filepath)
    print(f"Plot saved to {filepath}")
    plt.close()

# =====================================
# Data Loading and Preprocessing
# =====================================

# Load the dataset
print("Loading dataset...")
df = pd.read_excel(EXCEL_PATH)
print("Dataset loaded successfully.")

# Convert string representations to lists
for col in ['V1', 'V2', 'V3', 'V4', 'A2']:
    df[col] = df[col].apply(ast.literal_eval)
print("String representations converted to lists.")

# Ensure utterances are strings
df['Utterance'] = df['Utterance'].astype(str)

# Encode emotion labels
le = LabelEncoder()
df['Emotion_encoded'] = le.fit_transform(df['Emotion'])
print("Emotion labels encoded.")

# Print dataset information
print("\nDataset Information:")
print(f"Number of samples: {len(df)}")
print(f"Number of features: {len(df.columns)}")
print("\nEmotion distribution:")
print(df['Emotion'].value_counts())

# Print feature dimensions
print("\nFeature dimensions:")
for col in ['V1', 'V2', 'V3', 'V4', 'A2']:
    print(f"{col}: {len(df[col].iloc[0])}")

# =====================================
# Model Configuration
# =====================================

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print("BERT tokenizer initialized.")

# Hyperparameters
HYPERPARAMS = {
    'batch_size': 128,
    'accumulation_steps': 2,
    'max_length': 128,
    'num_epochs': 30,
    'patience': 5,
    'learning_rate': 2e-5
}

# Function to tokenize text
def tokenize_text(text, max_length=HYPERPARAMS['max_length']):
    """
    Tokenize input text using BERT tokenizer.
    Args:
        text: Input text to tokenize
        max_length: Maximum sequence length
    Returns:
        Tokenized text with padding and attention mask
    """
    return tokenizer(text, 
                    padding='max_length', 
                    truncation=True, 
                    max_length=max_length, 
                    return_tensors='pt')

# =====================================
# Data Preparation
# =====================================

# Prepare features and labels
X = df[['Utterance', 'V1', 'V3', 'V4', 'A2']]
y = df['Emotion_encoded']

# Oversample to handle class imbalance
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

print("\nClass distribution after oversampling:")
print(pd.Series(y_resampled).value_counts())

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, 
    y_resampled, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_resampled
)

print("\nData split into train and test sets.")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# =====================================
# Dataset Class
# =====================================

class MELDDataset(Dataset):
    """
    Custom Dataset class for MELD emotion recognition data.
    Handles multimodal data including text, audio, and visual features.
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
train_dataset = MELDDataset(X_train['Utterance'], X_train['V1'], X_train['V3'], 
                           X_train['V4'], X_train['A2'], y_train)
test_dataset = MELDDataset(X_test['Utterance'], X_test['V1'], X_test['V3'], 
                          X_test['V4'], X_test['A2'], y_test)

train_loader = DataLoader(train_dataset, 
                         batch_size=HYPERPARAMS['batch_size'], 
                         shuffle=True, 
                         num_workers=4, 
                         pin_memory=True)
test_loader = DataLoader(test_dataset, 
                        batch_size=HYPERPARAMS['batch_size'], 
                        shuffle=False, 
                        num_workers=4, 
                        pin_memory=True)

print("\nDataLoaders created successfully.")
print(f"Number of batches in train_loader: {len(train_loader)}")
print(f"Number of batches in test_loader: {len(test_loader)}")

# =====================================
# Model Architecture
# =====================================

class MultimodalEmotionRecognition(nn.Module):
    """
    Multimodal Emotion Recognition model combining text, audio, and visual features.
    Uses BERT for text, custom networks for audio and visual processing,
    and implements cross-modal attention mechanism.
    """
    def __init__(self, num_classes=7):
        super(MultimodalEmotionRecognition, self).__init__()

        # Text Processing (BERT-based)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.text_fc = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Audio Processing
        # Process V1 features (512-dim)
        self.audio_v1_fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )
        # Process V3 and V4 features (combined 50-dim)
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

        # Cross-Modal Attention Mechanism
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

        # Final Classification Layers
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
            input_ids: BERT input token IDs
            attention_mask: BERT attention mask
            v1: Audio feature vector 1
            v3: Audio feature vector 3
            v4: Audio feature vector 4
            a2: Visual feature vector
        Returns:
            logits: Raw model outputs for each emotion class
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
        attended_features = text_features + text_audio_attention.squeeze(1) + text_visual_attention.squeeze(1)

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
    Focuses more on hard examples by down-weighting easy examples.
    """
    def __init__(self, alpha, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha.to(device)
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha[targets] * (1 - pt)**self.gamma * ce_loss
        return focal_loss.mean()




# =====================================
# Training Functions
# =====================================

def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    """
    Plot and save training curves showing loss and accuracy metrics.
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accuracies: List of training accuracies
        val_accuracies: List of validation accuracies
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
    
    # Save and display the plot
    save_plot(plt, 'training_curves.png')

def train_model(model, train_loader, val_loader, num_epochs=HYPERPARAMS['num_epochs'], 
                patience=HYPERPARAMS['patience']):
    """
    Train the model with early stopping and learning rate scheduling.
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Maximum number of training epochs
        patience: Number of epochs to wait before early stopping
    Returns:
        model: Trained model
    """
    print("Starting model training...")
    print(f"Early stopping is {early_stop}")

    # Initialize loss function with class weights
    class_counts = torch.tensor([4709] * 7)  # All classes have 4709 samples after oversampling
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    criterion = FocalLoss(alpha=class_weights.to(device), gamma=2)

    # Initialize optimizer and schedulers
    optimizer = optim.AdamW(model.parameters(), 
                           lr=HYPERPARAMS['learning_rate'], 
                           weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    scaler = GradScaler()

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

            # Forward pass with mixed precision
            with autocast():
                outputs = model(input_ids, attention_mask, v1, v3, v4, a2)
                loss = criterion(outputs, labels)
                loss = loss / HYPERPARAMS['accumulation_steps']

            # Backward pass
            scaler.scale(loss).backward()

            # Gradient accumulation
            if (i + 1) % HYPERPARAMS['accumulation_steps'] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # Update metrics
            train_loss += loss.item() * HYPERPARAMS['accumulation_steps']
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

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
                with autocast():
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
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        print(f'Epoch Time: {epoch_time:.2f} seconds')

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Model checkpoint saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print("New best model saved.")
        else:
            epochs_without_improvement += 1
            if early_stop == "on" and epochs_without_improvement == patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break

        # Print GPU usage
        print_gpu_usage()

    print("Training completed.")

    # Plot final training curves
    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)

    return model



# =====================================
# Evaluation Functions
# =====================================

def plot_confusion_matrix(all_labels, all_preds, class_names):
    """
    Plot and save confusion matrix.
    Args:
        all_labels: True labels
        all_preds: Predicted labels
        class_names: Names of emotion classes
    """
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    # Save and display the plot
    save_plot(plt, 'confusion_matrix.png')

def plot_per_class_metrics(report):
    """
    Plot and save per-class performance metrics.
    Args:
        report: Classification report dictionary
    """
    classes = list(report.keys())[:-3]  # Exclude 'accuracy', 'macro avg', and 'weighted avg'
    precision = [report[c]['precision'] for c in classes]
    recall = [report[c]['recall'] for c in classes]
    f1_score = [report[c]['f1-score'] for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label='Precision')
    ax.bar(x, recall, width, label='Recall')
    ax.bar(x + width, f1_score, width, label='F1-score')

    ax.set_ylabel('Scores')
    ax.set_title('Per-class Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()

    fig.tight_layout()
    
    # Save and display the plot
    save_plot(plt, 'per_class_metrics.png')

def evaluate_model(model, test_loader):
    """
    Evaluate the model and generate performance metrics and visualizations.
    Args:
        model: Trained model
        test_loader: DataLoader for test data
    """
    print("Starting model evaluation...")
    model.eval()
    all_preds = []
    all_labels = []

    # Evaluation loop
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

            # Forward pass with mixed precision
            with autocast():
                outputs = model(input_ids, attention_mask, v1, v3, v4, a2)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Generate classification report
    report = classification_report(all_labels, all_preds, 
                                 target_names=le.classes_, 
                                 output_dict=True)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=le.classes_))

    # Save classification report to file
    report_path = os.path.join(RESULT_DIR, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(classification_report(all_labels, all_preds, target_names=le.classes_))
    print(f"\nClassification report saved to {report_path}")

    # Generate visualizations
    plot_confusion_matrix(all_labels, all_preds, le.classes_)
    plot_per_class_metrics(report)

    print("Evaluation completed. Results saved in:", RESULT_DIR)

# =====================================
# Main Execution
# =====================================

if __name__ == "__main__":
    print("\n" + "="*50)
    print("MELD Emotion Recognition Model Training")
    print("="*50 + "\n")

    # Create result directory if it doesn't exist
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
        print(f"Created result directory: {RESULT_DIR}")

    # Instantiate the model
    model = MultimodalEmotionRecognition()
    model = model.to(device)
    print("Model instantiated successfully.")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Debug information
    print("\nDebug information:")
    batch = next(iter(train_loader))
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    v1 = batch['v1'].to(device)
    v3 = batch['v3'].to(device)
    v4 = batch['v4'].to(device)
    a2 = batch['a2'].to(device)

    # Print input shapes for debugging
    print(f"Input shapes:")
    print(f"  Input IDs: {input_ids.shape}")
    print(f"  Attention Mask: {attention_mask.shape}")
    print(f"  V1: {v1.shape}")
    print(f"  V3: {v3.shape}")
    print(f"  V4: {v4.shape}")
    print(f"  A2: {a2.shape}")

    # Test forward pass
    print("\nTesting forward pass...")
    with torch.no_grad():
        with autocast():
            outputs = model(input_ids, attention_mask, v1, v3, v4, a2)
        print(f"Model output shape: {outputs.shape}")

    # Train the model
    print("\nStarting model training...")
    trained_model = train_model(model, train_loader, test_loader)

    # Evaluate the model
    print("\nStarting model evaluation...")
    evaluate_model(trained_model, test_loader)

    print("\nTraining and evaluation completed.")
    print(f"All results have been saved in: {RESULT_DIR}")
