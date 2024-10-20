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

# Early stopping toggle
early_stop = "on"  # Set to "off" to disable early stopping

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set device and optimize CUDA operations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
print(f"Using device: {device}")

# Function to print GPU usage
def print_gpu_usage():
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]
        print(f"GPU Memory Usage: {gpu.memoryUsed} / {gpu.memoryTotal} MB")
        print(f"GPU Memory Utilization: {gpu.memoryUtil * 100:.2f}%")

# Load the dataset
file_path = 'File path'
df = pd.read_excel(file_path)
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

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print("BERT tokenizer initialized.")

# Hyperparameters
batch_size = 128
accumulation_steps = 2
max_length = 128
num_epochs = 30
patience = 5
learning_rate = 2e-5

# Function to tokenize text
def tokenize_text(text, max_length=max_length):
    return tokenizer(text, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')

# Prepare features and labels
X = df[['Utterance', 'V1', 'V3', 'V4', 'A2']]
y = df['Emotion_encoded']

# Oversample to handle class imbalance
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

print("\nClass distribution after oversampling:")
print(pd.Series(y_resampled).value_counts())

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

print("\nData split into train and test sets.")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Custom Dataset class
class MELDDataset(Dataset):
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
train_dataset = MELDDataset(X_train['Utterance'], X_train['V1'], X_train['V3'], X_train['V4'], X_train['A2'], y_train)
test_dataset = MELDDataset(X_test['Utterance'], X_test['V1'], X_test['V3'], X_test['V4'], X_test['A2'], y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

print("\nDataLoaders created successfully.")
print(f"Number of batches in train_loader: {len(train_loader)}")
print(f"Number of batches in test_loader: {len(test_loader)}")

class MultimodalEmotionRecognition(nn.Module):
    def __init__(self, num_classes=7):
        super(MultimodalEmotionRecognition, self).__init__()

        # Text Modality
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.text_fc = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Audio Modality
        self.audio_v1_fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )
        self.audio_v34_fc = nn.Sequential(
            nn.Linear(50, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        self.audio_combined_fc = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Visual Modality
        self.visual_fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3)
        )

        # Cross-Modal Attention
        self.text_audio_attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        self.text_visual_attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)

        # Fusion Layer
        self.fusion_fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Classification Layers
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask, v1, v3, v4, a2):
        # Text Modality
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = self.text_fc(bert_output.pooler_output)

        # Audio Modality
        audio_v1 = self.audio_v1_fc(v1)
        audio_v34 = self.audio_v34_fc(torch.cat([v3, v4], dim=1))
        audio_features = self.audio_combined_fc(torch.cat([audio_v1, audio_v34], dim=1))

        # Visual Modality
        visual_features = self.visual_fc(a2)

        # Cross-Modal Attention
        text_features_expanded = text_features.unsqueeze(1)
        audio_features_expanded = nn.functional.pad(audio_features.unsqueeze(1), (0, 256))
        visual_features_expanded = nn.functional.pad(visual_features.unsqueeze(1), (0, 256))

        text_audio_attention, _ = self.text_audio_attention(text_features_expanded, audio_features_expanded, audio_features_expanded)
        text_visual_attention, _ = self.text_visual_attention(text_features_expanded, visual_features_expanded, visual_features_expanded)

        attended_features = text_features + text_audio_attention.squeeze(1) + text_visual_attention.squeeze(1)

        # Fusion
        fused_features = self.fusion_fc(torch.cat([attended_features, audio_features, visual_features], dim=1))

        # Classification
        logits = self.classifier(fused_features)

        return logits

# Focal Loss implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha.to(device)
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha[targets] * (1 - pt)**self.gamma * ce_loss
        return focal_loss.mean()

# Training function
def train_model(model, train_loader, val_loader, num_epochs=num_epochs, patience=patience):
    print("Starting model training...")
    print(f"Early stopping is {early_stop}")

    # Compute class weights for Focal Loss
    class_counts = torch.tensor([4709] * 7)  # All classes have 4709 samples after oversampling
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()

    criterion = FocalLoss(alpha=class_weights.to(device), gamma=2)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    scaler = GradScaler()

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        optimizer.zero_grad()

        start_time = time.time()

        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            v1 = batch['v1'].to(device)
            v3 = batch['v3'].to(device)
            v4 = batch['v4'].to(device)
            a2 = batch['a2'].to(device)
            labels = batch['label'].to(device)

            with autocast():
                outputs = model(input_ids, attention_mask, v1, v3, v4, a2)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * accumulation_steps
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

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
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                v1 = batch['v1'].to(device)
                v3 = batch['v3'].to(device)
                v4 = batch['v4'].to(device)
                a2 = batch['a2'].to(device)
                labels = batch['label'].to(device)

                with autocast():
                    outputs = model(input_ids, attention_mask, v1, v3, v4, a2)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        epoch_time = time.time() - start_time

        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        print(f'Epoch Time: {epoch_time:.2f} seconds')

        scheduler.step(val_loss)


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join('/content/drive/MyDrive/EANN', 'best_model.pth'))
            print("New best model saved.")
        else:
            epochs_without_improvement += 1
            if early_stop == "on" and epochs_without_improvement == patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break

        # Print GPU usage
        print_gpu_usage()

    print("Training completed.")

    # Save the final model
    torch.save(model.state_dict(), os.path.join('/content/drive/MyDrive/EANN', 'final_model.pth'))
    print("Final model saved.")

    # Plot final training curves
    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)

    return model

# Function to plot final training curves
def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.tight_layout()
    plt.show()  # Display the plot in the terminal
    plt.savefig(os.path.join('/content/drive/MyDrive/EANN', 'final_training_curves.png'))
    plt.close()

# Evaluation function
def evaluate_model(model, test_loader):
    print("Starting model evaluation...")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            v1 = batch['v1'].to(device)
            v3 = batch['v3'].to(device)
            v4 = batch['v4'].to(device)
            a2 = batch['a2'].to(device)
            labels = batch['label'].to(device)

            with autocast():
                outputs = model(input_ids, attention_mask, v1, v3, v4, a2)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Print classification report
    report = classification_report(all_labels, all_preds, target_names=le.classes_, output_dict=True)
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=le.classes_))

    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_preds, le.classes_)

    # Plot per-class metrics
    plot_per_class_metrics(report)

    print("Evaluation completed. Visualizations have been displayed.")

# Function to plot confusion matrix
def plot_confusion_matrix(all_labels, all_preds, class_names):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()  # Display the plot in the terminal
    plt.savefig(os.path.join('/content/drive/MyDrive/EANN', 'confusion_matrix.png'))
    plt.close()

# Function to plot per-class metrics
def plot_per_class_metrics(report):
    classes = list(report.keys())[:-3]  # Exclude 'accuracy', 'macro avg', and 'weighted avg'
    precision = [report[c]['precision'] for c in classes]
    recall = [report[c]['recall'] for c in classes]
    f1_score = [report[c]['f1-score'] for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width, precision, width, label='Precision')
    rects2 = ax.bar(x, recall, width, label='Recall')
    rects3 = ax.bar(x + width, f1_score, width, label='F1-score')

    ax.set_ylabel('Scores')
    ax.set_title('Per-class Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()

    fig.tight_layout()
    plt.show()  # Display the plot in the terminal
    plt.savefig(os.path.join('/content/drive/MyDrive/EANN', 'per_class_metrics.png'))
    plt.close()

# Main execution block
if __name__ == "__main__":
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

    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Attention Mask shape: {attention_mask.shape}")
    print(f"V1 shape: {v1.shape}")
    print(f"V3 shape: {v3.shape}")
    print(f"V4 shape: {v4.shape}")
    print(f"A2 shape: {a2.shape}")

    # Forward pass for debugging
    with torch.no_grad():
        with autocast():
            outputs = model(input_ids, attention_mask, v1, v3, v4, a2)
        print(f"Model output shape: {outputs.shape}")

    # Train the model
    trained_model = train_model(model, train_loader, test_loader)

    # Evaluate the model
    evaluate_model(trained_model, test_loader)

    print("Training and evaluation completed. Visualizations have been displayed and saved.")