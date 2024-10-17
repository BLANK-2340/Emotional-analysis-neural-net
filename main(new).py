import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import (AutoTokenizer, AutoModel, 
                          RobertaModel, RobertaTokenizer, 
                          ViTModel, ViTFeatureExtractor)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pprint import pprint
from transformers import ViTConfig, ViTModel


# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Manually set the path and dataset here
DATASET_PATH = r"C:\Users\armaa\Downloads\dataset\IMOCAP\IMOCAP.xlsx"  # Change this path as needed
DATASET_NAME = "IEMOCAP"  # Change this to "IEMOCAP" when using IEMOCAP dataset

# Function to load data
def load_data():
    return pd.read_excel(DATASET_PATH)

# Data Preprocessing
def preprocess_data(df):
    # Handle missing values
    df.dropna(inplace=True)
    
    # Encode emotions
    label_encoder = LabelEncoder()
    df['Emotion'] = label_encoder.fit_transform(df['Emotion'])
    
    # Convert string representations of lists to actual lists
    def parse_feature(feature):
        return np.array([float(x) for x in feature.strip('[]').split(',')])
    
    for col in ['V1', 'V2', 'V3', 'V4', 'A2']:
        df[col] = df[col].apply(parse_feature)
    
    # Project V2 in IEMOCAP from 384 to 512 dimensions
    if DATASET_NAME == 'IEMOCAP':
        def adjust_v2(row):
            if len(row['V2']) == 384:
                return np.pad(row['V2'], (0, 128), 'constant')
            else:
                return row['V2']
        df['V2'] = df.apply(adjust_v2, axis=1)
    
    # Normalize features
    from sklearn.preprocessing import StandardScaler
    
    for col in ['V1', 'V3', 'V4', 'A2']:
        scaler = StandardScaler()
        features = np.stack(df[col].values)
        df[col] = list(scaler.fit_transform(features))
    
    return df, label_encoder

# Dataset Class
class EmotionDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.v1 = df['V1'].values
        self.v2_text = df['Utterance'].values
        self.v2 = df['V2'].values
        self.v3 = df['V3'].values
        self.v4 = df['V4'].values
        self.a2 = df['A2'].values
        self.labels = df['Emotion'].values
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        v1 = torch.tensor(self.v1[idx], dtype=torch.float32)
        v2_text = str(self.v2_text[idx])
        v2 = torch.tensor(self.v2[idx], dtype=torch.float32)
        v3 = torch.tensor(self.v3[idx], dtype=torch.float32)
        v4 = torch.tensor(self.v4[idx], dtype=torch.float32)
        a2 = torch.tensor(self.a2[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        encoding = self.tokenizer.encode_plus(
            v2_text,
            add_special_tokens=True,
            max_length=128,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()

        return {
            'v1': v1,
            'v2_input_ids': input_ids,
            'v2_attention_mask': attention_mask,
            'v2_embedding': v2,
            'v3': v3,
            'v4': v4,
            'a2': a2,
            'label': label
        }

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class AdvancedEmotionRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(AdvancedEmotionRecognitionModel, self).__init__()

        # Audio Sub-network
        self.audio_conv1 = nn.Conv1d(in_channels=1, out_channels=256, kernel_size=3, padding=1)
        self.audio_bn1 = nn.BatchNorm1d(256)
        self.audio_swish = Swish()
        self.audio_se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(256, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(16, 256, kernel_size=1),
            nn.Sigmoid()
        )
        self.audio_tcn = nn.ModuleList([
            nn.Conv1d(256, 256, kernel_size=3, padding=d, dilation=d)
            for d in [1, 2, 4]
        ])
        self.audio_gru = nn.GRU(256, 256, batch_first=True, bidirectional=True, dropout=0.3)
        self.audio_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8)

        # Text Sub-network
        self.text_model = RobertaModel.from_pretrained('roberta-large')
        self.text_proj = nn.Linear(1024, 512)
        self.text_gru = nn.GRU(512, 256, batch_first=True, bidirectional=True, dropout=0.3)
        self.text_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8)

        # Visual Sub-network
        vit_config = ViTConfig.from_pretrained('google/vit-base-patch16-224')
        vit_config.image_size = 1
        vit_config.patch_size = 1
        self.visual_model = ViTModel(vit_config)
        self.visual_proj = nn.Linear(vit_config.hidden_size, 512)
        self.visual_tcn = nn.ModuleList([
            nn.Conv1d(512, 256, kernel_size=3, padding=d, dilation=d)
            for d in [1, 2, 4]
        ])
        self.visual_gru = nn.GRU(256, 256, batch_first=True, bidirectional=True, dropout=0.3)
        self.visual_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8)

        # Fusion Layers
        self.cross_attn_audio = nn.MultiheadAttention(embed_dim=256, num_heads=8)
        self.cross_attn_text = nn.MultiheadAttention(embed_dim=256, num_heads=8)
        self.cross_attn_visual = nn.MultiheadAttention(embed_dim=256, num_heads=8)

        self.gnn_layers = nn.ModuleList([
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 256),
        ])

        self.modality_gate = nn.Sequential(
            nn.Linear(256 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
            nn.Softmax(dim=1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            Swish(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, v1, v2_input_ids, v2_attention_mask, v2_embedding, v3, v4, a2):
        batch_size = v1.size(0)

        # Audio Processing
        audio_features = torch.cat([v1, v3, v4], dim=1).unsqueeze(1)  # Shape: [B, 1, 562]
        audio_features = self.audio_swish(self.audio_bn1(self.audio_conv1(audio_features)))  # [B, 256, L]
        se_weight = self.audio_se(audio_features)
        audio_features = audio_features * se_weight
        for conv in self.audio_tcn:
            res = audio_features
            audio_features = F.prelu(conv(audio_features), torch.tensor(0.25).to(audio_features.device))
            audio_features = audio_features + res
        audio_features = audio_features.permute(0, 2, 1)  # [B, L, 256]
        audio_output, _ = self.audio_gru(audio_features)
        audio_output = audio_output.permute(1, 0, 2)  # [L, B, 512]
        audio_output, _ = self.audio_attn(audio_output, audio_output, audio_output)
        audio_output = audio_output.mean(dim=0)  # [B, 512]
        audio_output = F.layer_norm(audio_output, [512])

        # Text Processing
        text_outputs = self.text_model(input_ids=v2_input_ids, attention_mask=v2_attention_mask)
        text_features = self.text_proj(text_outputs.last_hidden_state)  # [B, L, 512]
        text_output, _ = self.text_gru(text_features)
        text_output = text_output.permute(1, 0, 2)  # [L, B, 512]
        text_output, _ = self.text_attn(text_output, text_output, text_output)
        text_output = text_output.mean(dim=0)  # [B, 512]
        text_output = F.layer_norm(text_output, [512])

        # Visual Processing
        visual_features = a2  # [B, 512]
        # Reshape visual_features to match ViT input requirements
        visual_features = visual_features.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, 512]
        visual_features = self.visual_model(pixel_values=visual_features).last_hidden_state
        visual_features = self.visual_proj(visual_features)  # [B, L, 512]
        visual_features = visual_features.permute(0, 2, 1)  # [B, 512, L]
        for conv in self.visual_tcn:
            res = visual_features
            visual_features = F.mish(conv(visual_features))
            visual_features = visual_features + res
        visual_features = visual_features.permute(0, 2, 1)  # [B, L, 256]
        visual_output, _ = self.visual_gru(visual_features)
        visual_output = visual_output.permute(1, 0, 2)  # [L, B, 512]
        visual_output, _ = self.visual_attn(visual_output, visual_output, visual_output)
        visual_output = visual_output.mean(dim=0)  # [B, 512]
        visual_output = F.layer_norm(visual_output, [512])

        # Cross-Modal Attention
        audio_cross, _ = self.cross_attn_audio(audio_output.unsqueeze(0), torch.stack([text_output, visual_output]), torch.stack([text_output, visual_output]))
        text_cross, _ = self.cross_attn_text(text_output.unsqueeze(0), torch.stack([audio_output, visual_output]), torch.stack([audio_output, visual_output]))
        visual_cross, _ = self.cross_attn_visual(visual_output.unsqueeze(0), torch.stack([audio_output, text_output]), torch.stack([audio_output, text_output]))

        # GNN Fusion
        modality_features = torch.stack([audio_cross.squeeze(0), text_cross.squeeze(0), visual_cross.squeeze(0)], dim=1)  # [B, 3, 256]
        for layer in self.gnn_layers:
            modality_features = layer(modality_features)
        modality_features = modality_features.mean(dim=1)  # [B, 256]

        # Modality Gating
        gate_input = torch.cat([audio_output, text_output, visual_output], dim=1)  # [B, 512*3]
        gate_weights = self.modality_gate(gate_input)  # [B, 3]
        fused_features = (gate_weights[:, 0].unsqueeze(1) * audio_output +
                          gate_weights[:, 1].unsqueeze(1) * text_output +
                          gate_weights[:, 2].unsqueeze(1) * visual_output)

        # Classification
        logits = self.classifier(fused_features)  # [B, num_classes]

        return logits

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    best_val_f1 = 0
    train_losses, train_accs, val_losses, val_accs = [], [], [], []

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_acc = 0, 0
        train_preds, train_labels = [], []

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            v1, v2_input_ids, v2_attention_mask, v2_embedding, v3, v4, a2, labels = [b.to(device) for b in batch.values()]

            optimizer.zero_grad()
            outputs = model(v1, v2_input_ids, v2_attention_mask, v2_embedding, v3, v4, a2)
            loss = criterion(outputs, labels)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            total_acc += (preds == labels).sum().item()
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        avg_train_loss = total_loss / len(train_loader)
        avg_train_acc = total_acc / len(train_loader.dataset) * 100
        train_f1 = f1_score(train_labels, train_preds, average='weighted')
        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)

        # Validation
        model.eval()
        total_val_loss, total_val_acc = 0, 0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"):
                v1, v2_input_ids, v2_attention_mask, v2_embedding, v3, v4, a2, labels = [b.to(device) for b in batch.values()]

                outputs = model(v1, v2_input_ids, v2_attention_mask, v2_embedding, v3, v4, a2)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                total_val_acc += (preds == labels).sum().item()
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_acc = total_val_acc / len(val_loader.dataset) * 100
        val_f1 = f1_score(val_labels, val_preds, average='weighted')

        val_losses.append(avg_val_loss)
        val_accs.append(avg_val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.2f}%, Train F1: {train_f1:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.2f}%, Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'best_model.pth')
            print("Best model saved.")

    return train_losses, train_accs, val_losses, val_accs

# Evaluation function
def evaluate_model(model, test_loader, criterion, device, label_encoder):
    model.eval()
    total_loss, total_acc = 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            v1, v2_input_ids, v2_attention_mask, v2_embedding, v3, v4, a2, labels = [b.to(device) for b in batch.values()]

            outputs = model(v1, v2_input_ids, v2_attention_mask, v2_embedding, v3, v4, a2)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            total_acc += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    accuracy = total_acc / len(test_loader.dataset) * 100
    precision = precision_score(all_labels, all_preds, average='weighted') * 100
    recall = recall_score(all_labels, all_preds, average='weighted') * 100
    f1 = f1_score(all_labels, all_preds, average='weighted') * 100

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Test Precision: {precision:.2f}%")
    print(f"Test Recall: {recall:.2f}%")
    print(f"Test F1 Score: {f1:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

    # ROC Curve
    n_classes = len(label_encoder.classes_)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve((np.array(all_labels) == i).astype(int), 
                                      torch.nn.functional.softmax(torch.tensor(outputs), dim=1)[:, i].numpy())
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {label_encoder.classes_[i]} (area = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    return all_preds, all_labels

def print_tensor_info(tensor_dict):
    print("\nTensor Information:")
    for key, tensor in tensor_dict.items():
        print(f"{key}:")
        print(f"  Shape: {tensor.shape}")
        print(f"  Type: {tensor.dtype}")
        print(f"  Device: {tensor.device}")
        if tensor.numel() > 0:
            print(f"  Min: {tensor.min().item():.4f}")
            print(f"  Max: {tensor.max().item():.4f}")
            print(f"  Mean: {tensor.float().mean().item():.4f}")
        print()

def gather_input_info(model, train_loader):
    # Get a single batch from the train loader
    batch = next(iter(train_loader))
    
    # Extract tensors from the batch
    v1 = batch['v1']
    v2_input_ids = batch['v2_input_ids']
    v2_attention_mask = batch['v2_attention_mask']
    v2_embedding = batch['v2_embedding']
    v3 = batch['v3']
    v4 = batch['v4']
    a2 = batch['a2']
    labels = batch['label']

    # Print information about each tensor
    print_tensor_info({
        'v1': v1,
        'v2_input_ids': v2_input_ids,
        'v2_attention_mask': v2_attention_mask,
        'v2_embedding': v2_embedding,
        'v3': v3,
        'v4': v4,
        'a2': a2,
        'labels': labels
    })

    # Print model summary
    print("Model Summary:")
    print(model)

    # Print total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal number of parameters: {total_params}")

# Main execution
def main(num_epochs=20, batch_size=32, learning_rate=2e-4):
    # Load and preprocess data
    df = load_data()
    df, label_encoder = preprocess_data(df)

    # Split data
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Emotion'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['Emotion'])

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('roberta-large')

    # Create datasets and dataloaders
    train_dataset = EmotionDataset(train_df, tokenizer)
    val_dataset = EmotionDataset(val_df, tokenizer)
    test_dataset = EmotionDataset(test_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initialize model, loss function, optimizer, and scheduler
    num_classes = df['Emotion'].nunique()
    model = AdvancedEmotionRecognitionModel(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=num_epochs)

    # Gather and print input information
    gather_input_info(model, train_loader)

    # Train the model
    train_losses, train_accs, val_losses, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device
    )

    # Plot training and validation curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.show()

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load('best_model.pth'))
    all_preds, all_labels = evaluate_model(model, test_loader, criterion, device, label_encoder)

    return model, label_encoder

if __name__ == "__main__":
    main(num_epochs=20, batch_size=32, learning_rate=2e-4)