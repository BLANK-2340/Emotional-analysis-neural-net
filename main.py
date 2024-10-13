# Emotion Recognition Model Implementation using PyTorch

# Import necessary libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc)
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt
import seaborn as sns
import ast
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the datasets
print("Loading datasets...")
meld_df = pd.read_excel(r"C:\Users\armaa\Downloads\MELD.xlsx")
iemocap_df = pd.read_excel(r"C:\Users\armaa\Downloads\IMOCAP.xlsx")
print("Datasets loaded successfully.")

# Function to safely convert string representations of arrays to numpy arrays
def str_to_array(s):
    try:
        return np.array(ast.literal_eval(s))
    except:
        return np.zeros(1)  # Return an array of zeros if conversion fails

# Preprocess the data
def preprocess_data(df, label_encoder, dataset_name):
    # Drop rows with missing 'Utterance' or 'utterance'
    if 'Utterance' in df.columns:
        df = df.dropna(subset=['Utterance'])
    elif 'utterance' in df.columns:
        df = df.dropna(subset=['utterance'])

    # Encode emotion labels
    df['Emotion_Label'] = label_encoder.fit_transform(df['Emotion'])

    # Convert features to numpy arrays
    for feature in ['V1', 'V2', 'V3', 'V4', 'A2']:
        df[feature] = df[feature].apply(str_to_array)

    # Adjust features based on dataset
    if dataset_name == 'MELD':
        # Ensure V2 has dimension 512
        df['V2'] = df['V2'].apply(lambda x: x if len(x) == 512 else np.zeros(512))
    elif dataset_name == 'IEMOCAP':
        # Ensure V2 has dimension 384
        df['V2'] = df['V2'].apply(lambda x: x if len(x) == 384 else np.zeros(384))

    return df

print("Preprocessing MELD dataset...")
meld_le = LabelEncoder()
meld_df = preprocess_data(meld_df.copy(), meld_le, 'MELD')
print("MELD dataset preprocessed.")

print("Preprocessing IEMOCAP dataset...")
iemocap_le = LabelEncoder()
iemocap_df = preprocess_data(iemocap_df.copy(), iemocap_le, 'IEMOCAP')
print("IEMOCAP dataset preprocessed.")

# Select dataset (MELD or IEMOCAP)
# To use IEMOCAP, change 'meld_df' to 'iemocap_df' and adjust variables accordingly
df = iemocap_df  # Change to 'df = iemocap_df' for IEMOCAP
le = iemocap_le  # Change to 'le = iemocap_le' for IEMOCAP
dataset_name = 'IEMOCAP'  # Change to 'IEMOCAP' for IEMOCAP

# Prepare input features and labels
print("Preparing input features and labels...")
V1 = np.vstack(df['V1'].values)
V2 = np.vstack(df['V2'].values)
V3 = np.vstack(df['V3'].values)
V4 = np.vstack(df['V4'].values)
A2 = np.vstack(df['A2'].values)

# Adjust the feature dimensions as per the provided dimensions
if dataset_name == 'MELD':
    # MELD dimensions
    # Audio features: V1 (512), V3 (25), V4 (25) => Total: 562
    # Text feature: V2 (512)
    # Visual feature: A2 (512)
    audio_features = np.hstack([V1, V3, V4])  # Shape: (num_samples, 562)
    text_features = V2  # Shape: (num_samples, 512)
    visual_features = A2  # Shape: (num_samples, 512)
elif dataset_name == 'IEMOCAP':
    # IEMOCAP dimensions
    # Audio features: V1 (512), V3 (25), V4 (25) => Total: 562
    # Text feature: V2 (384)
    # Visual feature: A2 (512)
    audio_features = np.hstack([V1, V3, V4])  # Shape: (num_samples, 562)
    text_features = V2  # Shape: (num_samples, 384)
    visual_features = A2  # Shape: (num_samples, 512)

print(f"Audio features shape: {audio_features.shape}")
print(f"Text features shape: {text_features.shape}")
print(f"Visual features shape: {visual_features.shape}")

labels = df['Emotion_Label'].values
num_classes = len(np.unique(labels))
print(f"Number of classes: {num_classes}")

# Normalize features
print("Normalizing features...")
scaler_audio = StandardScaler()
audio_features = scaler_audio.fit_transform(audio_features)

scaler_text = StandardScaler()
text_features = scaler_text.fit_transform(text_features)

scaler_visual = StandardScaler()
visual_features = scaler_visual.fit_transform(visual_features)
print("Features normalized.")

# Split data into training and validation sets
print("Splitting data into training and validation sets...")
X_audio_train, X_audio_val, X_text_train, X_text_val, X_visual_train, X_visual_val, y_train, y_val = train_test_split(
    audio_features, text_features, visual_features, labels, test_size=0.2, random_state=42
)
print("Data split completed.")

# Compute class weights
print("Computing class weights...")
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
# Cap class weights to a maximum value
max_weight = 10.0
class_weights = np.clip(class_weights, a_min=None, a_max=max_weight)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
class_weight_dict = dict(enumerate(class_weights.cpu().numpy()))
print(f"Class weights: {class_weight_dict}")

# Create custom dataset
class EmotionDataset(Dataset):
    def __init__(self, audio_features, text_features, visual_features, labels):
        self.audio_features = torch.tensor(audio_features, dtype=torch.float32)
        self.text_features = torch.tensor(text_features, dtype=torch.float32)
        self.visual_features = torch.tensor(visual_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'audio': self.audio_features[idx],
            'text': self.text_features[idx],
            'visual': self.visual_features[idx],
            'label': self.labels[idx]
        }

# Create datasets and dataloaders
print("Creating datasets and dataloaders...")
batch_size = 32
train_dataset = EmotionDataset(X_audio_train, X_text_train, X_visual_train, y_train)
val_dataset = EmotionDataset(X_audio_val, X_text_val, X_visual_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
print("Datasets and dataloaders created.")

# Define the model architecture
print("Building the model...")

class AudioSubNetwork(nn.Module):
    def __init__(self, input_dim):
        super(AudioSubNetwork, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, 512)
        self.activation1 = nn.LeakyReLU(0.1)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.activation2 = nn.LeakyReLU(0.1)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.activation3 = nn.ELU()
    def forward(self, x):
        x = self.bn(x)
        x = self.fc1(x)
        x = self.activation1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.activation2(x)
        x = self.bn2(x)
        x = self.fc3(x)
        x = self.activation3(x)
        return x

class TextSubNetwork(nn.Module):
    def __init__(self, input_dim):
        super(TextSubNetwork, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, 512)
        self.activation1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(512, 256)
        self.activation2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(256, 128)
        self.activation3 = nn.SELU()
    def forward(self, x):
        x = self.bn(x)
        x = self.fc1(x)
        x = self.activation1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.activation2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.activation3(x)
        return x

class VisualSubNetwork(nn.Module):
    def __init__(self, input_dim):
        super(VisualSubNetwork, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, 256)
        self.activation1 = nn.SiLU()  # Swish activation
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.activation2 = nn.ReLU()
    def forward(self, x):
        x = self.bn(x)
        x = self.fc1(x)
        x = self.activation1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.activation2(x)
        return x

class FusionNetwork(nn.Module):
    def __init__(self, num_classes):
        super(FusionNetwork, self).__init__()
        self.num_classes = num_classes
        self.num_heads = 4  # Reduced number of heads
        self.transformer_layers = 2  # Reduced number of layers
        self.embedding_dim = 128
        self.model_dim = self.embedding_dim
        
        # Positional Encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, 3, self.embedding_dim))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=self.num_heads,
            dim_feedforward=512,  # Reduced for simplicity
            activation='gelu',
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.transformer_layers)
        
        self.flatten = nn.Flatten()
        self.gate = nn.Linear(self.model_dim * 3, self.model_dim * 3)
        self.sigmoid = nn.Sigmoid()
        
        self.fc1 = nn.Linear(self.model_dim * 3, 256)
        self.activation1 = nn.LeakyReLU(0.1)
        self.bn = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, 128)
        self.activation2 = nn.ELU()
        self.fc_out = nn.Linear(128, num_classes)
    
    def forward(self, audio_emb, text_emb, visual_emb):
        # Stack modality embeddings to form a sequence
        x = torch.stack([audio_emb, text_emb, visual_emb], dim=1)  # Shape: (batch_size, 3, 128)
        x = x + self.positional_encoding  # Add positional encoding
        
        # Pass through transformer layers
        x = self.transformer_encoder(x)
        
        # Flatten
        x = self.flatten(x)  # Shape: (batch_size, 3 * 128)
        
        # Dynamic Modality Gating
        gate = self.sigmoid(self.gate(x))
        x = x * gate
        
        # Final layers
        x = self.fc1(x)
        x = self.activation1(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.activation2(x)
        x = self.fc_out(x)
        
        return x

class EmotionRecognitionModel(nn.Module):
    def __init__(self, audio_input_dim, text_input_dim, visual_input_dim, num_classes):
        super(EmotionRecognitionModel, self).__init__()
        self.audio_subnet = AudioSubNetwork(audio_input_dim)
        self.text_subnet = TextSubNetwork(text_input_dim)
        self.visual_subnet = VisualSubNetwork(visual_input_dim)
        self.fusion_net = FusionNetwork(num_classes)
        
    def forward(self, audio_input, text_input, visual_input):
        audio_emb = self.audio_subnet(audio_input)
        text_emb = self.text_subnet(text_input)
        visual_emb = self.visual_subnet(visual_input)
        
        output = self.fusion_net(audio_emb, text_emb, visual_emb)
        return output

# Initialize the model
audio_input_dim = audio_features.shape[1]
text_input_dim = text_features.shape[1]
visual_input_dim = visual_features.shape[1]

model = EmotionRecognitionModel(audio_input_dim, text_input_dim, visual_input_dim, num_classes)
model.to(device)
print("Model built successfully.")

# Define loss function and optimizer
print("Defining loss function and optimizer...")
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

criterion = FocalLoss(alpha=class_weights, gamma=2)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=500
)
print("Loss function and optimizer defined.")

# Training loop
print("Starting training...")
num_epochs = 500
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(1, num_epochs +1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        audio_input = batch['audio'].to(device)
        text_input = batch['text'].to(device)
        visual_input = batch['visual'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(audio_input, text_input, visual_input)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        running_loss += loss.item() * labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if (i + 1) % 50 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Step [{i+1}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}')
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    
    # Validation
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch in val_loader:
            audio_input = batch['audio'].to(device)
            text_input = batch['text'].to(device)
            visual_input = batch['visual'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(audio_input, text_input, visual_input)
            loss = criterion(outputs, labels)
            
            val_running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    val_epoch_loss = val_running_loss / val_total
    val_epoch_acc = val_correct / val_total
    val_losses.append(val_epoch_loss)
    val_accuracies.append(val_epoch_acc)
    
    print(f'Epoch {epoch}/{num_epochs}, '
          f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, '
          f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}')
    
    # Early stopping condition (optional)
    # Implement early stopping logic here if needed

print("Training completed.")

# Plot training and validation loss and accuracy
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Evaluate the model
print("Evaluating the model on validation data...")
model.eval()
all_labels = []
all_predictions = []
all_outputs = []
with torch.no_grad():
    for batch in val_loader:
        audio_input = batch['audio'].to(device)
        text_input = batch['text'].to(device)
        visual_input = batch['visual'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(audio_input, text_input, visual_input)
        _, predicted = torch.max(outputs.data, 1)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
        all_outputs.extend(outputs.cpu().numpy())

# Classification Report
print("Classification Report:")
unique_labels = np.unique(np.concatenate((y_train, y_val)))
target_names = le.inverse_transform(unique_labels)
print(classification_report(all_labels, all_predictions, labels=unique_labels, target_names=target_names, zero_division=1))

# Confusion Matrix
print("Confusion Matrix:")
cm = confusion_matrix(all_labels, all_predictions, labels=unique_labels)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names, cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve and AUC
print("Plotting ROC Curve...")
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# Binarize the labels
y_val_binarized = label_binarize(all_labels, classes=unique_labels)
y_pred_scores = np.array(all_outputs)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for idx, label in enumerate(unique_labels):
    fpr[idx], tpr[idx], _ = roc_curve(y_val_binarized[:, idx], y_pred_scores[:, idx])
    roc_auc[idx] = auc(fpr[idx], tpr[idx])

# Plot ROC curves for the classes
plt.figure()
for idx, label in enumerate(unique_labels):
    plt.plot(fpr[idx], tpr[idx], label=f'Class {le.inverse_transform([label])[0]} (AUC = {roc_auc[idx]:0.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.show()

# Save the model
print("Saving the model...")
torch.save(model.state_dict(), 'emotion_recognition_model.pth')
print("Model saved as 'emotion_recognition_model.pth'.")

# Print hyperparameters and parameters
print("\nHyperparameters:")
print(f"Batch size: {batch_size}")
print(f"Number of epochs: {num_epochs}")
print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
print(f"Optimizer: {optimizer}")
print(f"Loss function: {criterion}")
print(f"Scheduler: {scheduler}")
print(f"Device: {device}")

print("\nModel Architecture:")
print(model)

print("\nTraining completed. You can now use this model for emotion recognition tasks.")
