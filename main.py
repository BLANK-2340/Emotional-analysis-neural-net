import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import numpy as np
import ast
import gc

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Enable mixed precision training
scaler = torch.amp.GradScaler()

# 1. Data Loading and Preprocessing
print("Loading data...")
file_path = '/content/drive/MyDrive/EANN/MELD.xlsx' 
df = pd.read_excel(file_path)

print("Preprocessing data...")
for col in ['V1', 'V2', 'V3', 'V4', 'A2']:
    df[col] = df[col].apply(ast.literal_eval)

le = LabelEncoder()
df['Emotion_encoded'] = le.fit_transform(df['Emotion'])

X = df[['V1', 'V2', 'V3', 'V4', 'A2']]
y = df['Emotion_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

class MELDDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'V1': torch.tensor(self.features.iloc[idx]['V1'], dtype=torch.float32),
            'V2': torch.tensor(self.features.iloc[idx]['V2'], dtype=torch.float32),
            'V3': torch.tensor(self.features.iloc[idx]['V3'], dtype=torch.float32),
            'V4': torch.tensor(self.features.iloc[idx]['V4'], dtype=torch.float32),
            'A2': torch.tensor(self.features.iloc[idx]['A2'], dtype=torch.float32),
            'label': torch.tensor(self.labels.iloc[idx], dtype=torch.long)
        }

# Create datasets and dataloaders
train_dataset = MELDDataset(X_train, y_train)
test_dataset = MELDDataset(X_test, y_test)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

# 2. Model Architecture
class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()
        self.fc_v1 = nn.Linear(512, 256)
        self.bn_v1 = nn.BatchNorm1d(256)
        self.dropout_v1 = nn.Dropout(0.3)
        self.conv_v34 = nn.Conv1d(2, 64, kernel_size=3, padding=1)
        self.pool_v34 = nn.MaxPool1d(2)
        self.fc_v34 = nn.Linear(64 * 12, 128)
        self.fc_final = nn.Linear(384, 256)
    
    def forward(self, v1, v3, v4):
        a_v1 = self.dropout_v1(self.bn_v1(F.relu(self.fc_v1(v1))))
        v34 = torch.cat([v3.unsqueeze(1), v4.unsqueeze(1)], dim=1)
        a_v34 = self.conv_v34(v34)
        a_v34 = self.pool_v34(a_v34)
        a_v34 = a_v34.view(a_v34.size(0), -1)
        a_v34 = F.relu(self.fc_v34(a_v34))
        combined = torch.cat([a_v1, a_v34], dim=1)
        a_final = F.relu(self.fc_final(combined))
        return a_final

class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=512, batch_first=True),
            num_layers=2
        )
        self.fc = nn.Linear(512, 256)
    
    def forward(self, v2):
        t = torch.utils.checkpoint.checkpoint(self.transformer, v2)
        t_final = F.relu(self.fc(t))
        return t_final

class VisualEncoder(nn.Module):
    def __init__(self):
        super(VisualEncoder, self).__init__()
        self.fc = nn.Linear(512, 256)
    
    def forward(self, a2):
        v_final = F.relu(self.fc(a2))
        return v_final

class CrossModalAttention(nn.Module):
    def __init__(self, dim=256):
        super(CrossModalAttention, self).__init__()
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
    
    def forward(self, query, key, value):
        q = self.query_proj(query)
        k = self.key_proj(key)
        v = self.value_proj(value)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        return out

class MemoryEfficientTensorFusion(nn.Module):
    def __init__(self, input_dim=256, output_dim=512):
        super(MemoryEfficientTensorFusion, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc = nn.Linear(input_dim * 3, output_dim)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, audio, text, visual):
        fused = torch.cat([audio, text, visual], dim=1)
        fused_features = F.relu(self.fc(fused))
        fused_features = self.dropout(fused_features)
        return fused_features

class EmotionClassifier(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionClassifier, self).__init__()
        self.audio_encoder = AudioEncoder()
        self.text_encoder = TextEncoder()
        self.visual_encoder = VisualEncoder()
        self.cross_attn_at = CrossModalAttention()
        self.cross_attn_tv = CrossModalAttention()
        self.cross_attn_av = CrossModalAttention()
        self.tfn = MemoryEfficientTensorFusion()
        self.fc1 = nn.Linear(512, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, v1, v2, v3, v4, a2):
        audio = self.audio_encoder(v1, v3, v4)
        text = self.text_encoder(v2)
        visual = self.visual_encoder(a2)
        
        audio_text = self.cross_attn_at(audio, text, text)
        text_visual = self.cross_attn_tv(text, visual, visual)
        audio_visual = self.cross_attn_av(audio, visual, visual)
        
        audio_final = audio + audio_text + audio_visual
        text_final = text + text_visual
        visual_final = visual + audio_visual
        
        fused_features = self.tfn(audio_final, text_final, visual_final)
        
        x = F.relu(self.fc1(fused_features))
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits

# 3. Training Loop with Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha[targets] * (1 - pt)**self.gamma * ce_loss
        return focal_loss.mean()

def train_model(model, train_loader, val_loader, num_epochs=100):
    model.to(device)
    
    class_counts = df['Emotion'].value_counts().sort_index()
    class_weights = 1 / torch.Tensor(class_counts.values)
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(device)
    
    criterion = FocalLoss(alpha=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            v1, v2, v3, v4, a2, labels = [b.to(device) for b in batch.values()]
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(v1, v2, v3, v4, a2)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # Clear memory
            del v1, v2, v3, v4, a2, labels, outputs, loss
            torch.cuda.empty_cache()
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                v1, v2, v3, v4, a2, labels = [b.to(device) for b in batch.values()]
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(v1, v2, v3, v4, a2)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                # Clear memory
                del v1, v2, v3, v4, a2, labels, outputs, loss
                torch.cuda.empty_cache()
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()
    
    return model, train_losses, val_losses, train_accs, val_accs

# 4. Evaluation and Visualization functions
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluation"):
            v1, v2, v3, v4, a2, labels = [b.to(device) for b in batch.values()]
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(v1, v2, v3, v4, a2)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Clear memory
            del v1, v2, v3, v4, a2, labels, outputs
            torch.cuda.empty_cache()
    
    return np.array(all_preds), np.array(all_labels)

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    plt.close()

def plot_roc_curve(y_true, y_pred_proba, classes):
    n_classes = len(classes)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve((y_true == i).astype(int), y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))

    for i, color in zip(range(n_classes), plt.cm.rainbow(np.linspace(0, 1, n_classes))):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {classes[i]} (area = {roc_auc[i]:0.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()

def plot_loss_accuracy(train_losses, val_losses, train_accs, val_accs):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('loss_accuracy.png')
    plt.close()

# Main execution
if __name__ == "__main__":
    print("Initializing model...")
    model = EmotionClassifier()
    
    print("Starting training...")
    trained_model, train_losses, val_losses, train_accs, val_accs = train_model(model, train_loader, test_loader)
    
    print("Evaluating model...")
    y_pred, y_true = evaluate_model(trained_model, test_loader)
    
    print("Generating visualizations...")
    plot_confusion_matrix(y_true, y_pred, le.classes_)
    
    trained_model.eval()
    y_pred_proba = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Getting probabilities"):
            v1, v2, v3, v4, a2, _ = [b.to(device) for b in batch.values()]
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = trained_model(v1, v2, v3, v4, a2)
            y_pred_proba.extend(F.softmax(outputs, dim=1).cpu().numpy())
            
            # Clear memory
            del v1, v2, v3, v4, a2, outputs
            torch.cuda.empty_cache()
    
    y_pred_proba = np.array(y_pred_proba)
    plot_roc_curve(y_true, y_pred_proba, le.classes_)
    
    plot_loss_accuracy(train_losses, val_losses, train_accs, val_accs)
    
    print("Evaluation metrics:")
    print(classification_report(y_true, y_pred, target_names=le.classes_))
    
    print("Training and evaluation completed. Visualizations saved as PNG files.")