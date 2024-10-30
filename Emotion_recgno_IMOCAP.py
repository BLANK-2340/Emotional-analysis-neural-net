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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_and_print(message):
    logger.info(message)
    print(message)

# Early stopping toggle
early_stop = "off"  # Set to "on" to enable early stopping, "off" to disable
log_and_print(f"Early stopping is set to: {early_stop}")

# Set environment variables
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
log_and_print("Environment variables set for CUDA debugging")

# CUDA setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
log_and_print(f"Using device: {device}")

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
log_and_print("Random seeds set")

# Hyperparameters
batch_size = 16
accumulation_steps = 4
max_length = 128
num_epochs = 50
patience = 5
learning_rate = 1e-5

log_and_print(f"Hyperparameters: batch_size={batch_size}, accumulation_steps={accumulation_steps}, "
              f"max_length={max_length}, num_epochs={num_epochs}, patience={patience}, learning_rate={learning_rate}")

# Load and preprocess data
file_path = '/content/drive/MyDrive/EANN/Filtered_IMOCAP.xlsx'
df = pd.read_excel(file_path)
log_and_print(f"Data loaded from {file_path}")

for col in ['V1', 'V2', 'V3', 'V4', 'A2']:
    df[col] = df[col].apply(ast.literal_eval)
log_and_print("String representations converted to lists")

df['Utterance'] = df['Utterance'].astype(str)

le = LabelEncoder()
df['Emotion_encoded'] = le.fit_transform(df['Emotion'])
log_and_print("Emotion labels encoded")

X = df[['Utterance', 'V1', 'V3', 'V4', 'A2']]
y = df['Emotion_encoded']

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)
log_and_print("Data oversampled to handle class imbalance")

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)
log_and_print(f"Data split into train and test sets. Train size: {len(X_train)}, Test size: {len(X_test)}")

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
log_and_print("BERT tokenizer initialized")

def tokenize_text(text, max_length=max_length):
    return tokenizer(text, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')

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

train_dataset = MELDDataset(X_train['Utterance'], X_train['V1'], X_train['V3'], X_train['V4'], X_train['A2'], y_train)
test_dataset = MELDDataset(X_test['Utterance'], X_test['V1'], X_test['V3'], X_test['V4'], X_test['A2'], y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
log_and_print(f"DataLoaders created. Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

class MultimodalEmotionRecognition(nn.Module):
    def __init__(self, num_classes=9):
        super(MultimodalEmotionRecognition, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.text_fc = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

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

        self.visual_fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3)
        )

        self.text_audio_attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        self.text_visual_attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)

        self.fusion_fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask, v1, v3, v4, a2):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = self.text_fc(bert_output.pooler_output)

        audio_v1 = self.audio_v1_fc(v1)
        audio_v34 = self.audio_v34_fc(torch.cat([v3, v4], dim=1))
        audio_features = self.audio_combined_fc(torch.cat([audio_v1, audio_v34], dim=1))

        visual_features = self.visual_fc(a2)

        text_features_expanded = text_features.unsqueeze(1)
        audio_features_expanded = nn.functional.pad(audio_features.unsqueeze(1), (0, 256))
        visual_features_expanded = nn.functional.pad(visual_features.unsqueeze(1), (0, 256))

        text_audio_attention, _ = self.text_audio_attention(text_features_expanded, audio_features_expanded, audio_features_expanded)
        text_visual_attention, _ = self.text_visual_attention(text_features_expanded, visual_features_expanded, visual_features_expanded)

        attended_features = text_features + text_audio_attention.squeeze(1) + text_visual_attention.squeeze(1)

        fused_features = self.fusion_fc(torch.cat([attended_features, audio_features, visual_features], dim=1))

        logits = self.classifier(fused_features)

        return logits

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha.to(device)
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.clamp(torch.exp(-ce_loss), min=1e-8, max=1-1e-8)
        focal_loss = self.alpha[targets] * (1 - pt)**self.gamma * ce_loss
        return focal_loss.mean()

def train_model(model, train_loader, val_loader, num_epochs=num_epochs, patience=patience):
    log_and_print(f"Starting model training... Early stopping is {early_stop}")

    class_counts = torch.tensor([411] * 9)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()

    criterion = FocalLoss(alpha=class_weights.to(device), gamma=2)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

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

            outputs = model(input_ids, attention_mask, v1, v3, v4, a2)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps

            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * accumulation_steps
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            if i % 10 == 0:
                log_and_print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f}")

        train_loss /= len(train_loader)
        train_accuracy = train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

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

        log_and_print(f'Epoch {epoch+1}/{num_epochs}:')
        log_and_print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        log_and_print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        log_and_print(f'Epoch Time: {epoch_time:.2f} seconds')

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join('/content/drive/MyDrive/EANN', 'best_model.pth'))
            log_and_print("New best model saved.")
        else:
            epochs_without_improvement += 1
            if early_stop == "on" and epochs_without_improvement == patience:
                log_and_print(f'Early stopping triggered after {epoch+1} epochs')
                break

        # Plot and save training progress
        plot_training_progress(train_losses, val_losses, train_accuracies, val_accuracies, epoch+1)

    log_and_print("Training completed.")

    torch.save(model.state_dict(), os.path.join('/content/drive/MyDrive/EANN', 'final_model.pth'))
    log_and_print("Final model saved.")

    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)

    return model

def plot_training_progress(train_losses, val_losses, train_accuracies, val_accuracies, epoch):
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
    plt.savefig(os.path.join('/content/drive/MyDrive/EANN', f'training_progress_epoch_{epoch}.png'))
    plt.close()
    log_and_print(f"Training progress plot saved for epoch {epoch}")

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
    plt.savefig(os.path.join('/content/drive/MyDrive/EANN', 'final_training_curves.png'))
    plt.close()
    log_and_print("Final training curves saved")

def evaluate_model(model, test_loader):
    log_and_print("Starting model evaluation...")
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

            outputs = model(input_ids, attention_mask, v1, v3, v4, a2)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(all_labels, all_preds, target_names=le.classes_, output_dict=True)
    log_and_print("Classification Report:")
    log_and_print("\n" + classification_report(all_labels, all_preds, target_names=le.classes_))

    plot_confusion_matrix(all_labels, all_preds, le.classes_)
    plot_per_class_metrics(report)

    log_and_print("Evaluation completed. Visualizations have been saved.")

def plot_confusion_matrix(all_labels, all_preds, class_names):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join('/content/drive/MyDrive/EANN', 'confusion_matrix.png'))
    plt.close()
    log_and_print("Confusion matrix saved")

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
    plt.savefig(os.path.join('/content/drive/MyDrive/EANN', 'per_class_metrics.png'))
    plt.close()
    log_and_print("Per-class metrics plot saved")

# Function to clear CUDA cache and reset the model if needed
def reset_model():
    global model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = MultimodalEmotionRecognition().to(device)
    log_and_print("Model reset and CUDA cache cleared.")

if __name__ == "__main__":
    try:
        model = MultimodalEmotionRecognition()
        model = model.to(device)
        log_and_print("Model instantiated successfully.")
        log_and_print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

        log_and_print("\nDebug information:")
        batch = next(iter(train_loader))
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        v1 = batch['v1'].to(device)
        v3 = batch['v3'].to(device)
        v4 = batch['v4'].to(device)
        a2 = batch['a2'].to(device)

        log_and_print(f"Input IDs shape: {input_ids.shape}")
        log_and_print(f"Attention Mask shape: {attention_mask.shape}")
        log_and_print(f"V1 shape: {v1.shape}")
        log_and_print(f"V3 shape: {v3.shape}")
        log_and_print(f"V4 shape: {v4.shape}")
        log_and_print(f"A2 shape: {a2.shape}")

        with torch.no_grad():
            outputs = model(input_ids, attention_mask, v1, v3, v4, a2)
        log_and_print(f"Model output shape: {outputs.shape}")

        trained_model = train_model(model, train_loader, test_loader)
        evaluate_model(trained_model, test_loader)

        log_and_print("Script execution completed successfully.")
    except Exception as e:
        log_and_print(f"An error occurred during script execution: {e}")
        import traceback
        log_and_print(traceback.format_exc())

# reset_model()
