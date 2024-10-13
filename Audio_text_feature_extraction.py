import os
import cv2
import torch
import pandas as pd
import numpy as np
from torchvision import models, transforms
from torch import nn
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sentence_transformers import SentenceTransformer
import librosa
from torchaudio.transforms import MFCC
from tqdm import tqdm
import time

# Define paths
main_excel_path = r"C:\Users\armaa\Downloads\archive\iemocap_session1.xlsx"
video_folder_path = r"C:\Users\armaa\Downloads\archive\IEMOCAP_full_release\Session1\dialog\video"
audio_folder_path = r"C:\Users\armaa\Downloads\archive\IEMOCAP_full_release\Session1\dialog\audio"
output_excel_path = r"C:\Users\armaa\Downloads\archive\output.xlsx"


# Load required models
print("Loading models...")
voice_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
voice_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").eval()
text_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ResNet-50 model
resnet50 = models.resnet50(pretrained=True)
resnet50 = nn.Sequential(*list(resnet50.children())[:-1])  # Remove the classification layer
resnet50.eval()

# Define image transformation
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), 512).to(x.device)
        c0 = torch.zeros(1, x.size(0), 512).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Initialize LSTM model
lstm_model = LSTMModel(input_dim=2048, hidden_dim=512, output_dim=512, num_layers=1)
lstm_model.eval()

print("Models loaded successfully.")

# Audio processing functions
def extract_voice_embeddings(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    input_values = voice_processor(waveform.squeeze(), return_tensors="pt", sampling_rate=sample_rate).input_values
    with torch.no_grad():
        hidden_states = voice_model(input_values).last_hidden_state
    embeddings = hidden_states.mean(dim=1)
    return embeddings.squeeze().numpy()[:512]

def extract_mfccs(audio_path, n_mfcc=25):
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    mfcc_transform = MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc)
    mfcc = mfcc_transform(waveform).mean(dim=2).squeeze()
    return mfcc.numpy()[:25]

def extract_combined_features(audio_path):
    y, sr = librosa.load(audio_path, mono=True)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y).mean()
    rms = librosa.feature.rms(y=y).mean()
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = np.mean(pitches[pitches > 0])
    feature_vector = np.array([spectral_centroid, zero_crossing_rate, rms, tempo, pitch])
    feature_matrix = np.tile(feature_vector, (5, 1))
    flattened_features = feature_matrix.flatten()
    return flattened_features[:25]

def extract_utterance_embeddings(text):
    with torch.no_grad():
        embeddings = text_model.encode([text], convert_to_tensor=True, show_progress_bar=False)
    return embeddings.squeeze().numpy()

# Video processing functions
def extract_frames(video_path, num_frames=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def select_effective_frames(frames, num_frames=10):
    def frame_entropy(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        return -np.sum(hist * np.log2(hist + 1e-7))

    entropies = [frame_entropy(frame) for frame in frames]
    sorted_indices = sorted(range(len(entropies)), key=lambda k: entropies[k], reverse=True)
    return [frames[i] for i in sorted_indices[:num_frames]]

def process_video(video_path):
    frames = extract_frames(video_path)
    effective_frames = select_effective_frames(frames)
    frame_vectors = []
    for frame in effective_frames:
        input_tensor = preprocess(frame).unsqueeze(0)
        with torch.no_grad():
            frame_vector = resnet50(input_tensor).squeeze().numpy()
        frame_vectors.append(frame_vector)
    frame_tensor = torch.tensor(np.array(frame_vectors)).unsqueeze(0)  # Shape: 1x10x2048
    with torch.no_grad():
        video_vector = lstm_model(frame_tensor).squeeze().numpy()  # Shape: 512
    return video_vector

def process_all_data():
    main_df = pd.read_excel(main_excel_path)
    data = []

    total_files = len(main_df)
    start_time = time.time()

    print(f"Starting processing of {total_files} files...")

    for index, row in tqdm(main_df.iterrows(), total=total_files, desc="Processing", unit="file"):
        file_name = row['File_name']
        utterance = row['utterance']
        audio_path = os.path.join(audio_folder_path, f"{file_name}.wav")
        video_path = os.path.join(video_folder_path, f"{file_name}.mp4")

        if os.path.exists(audio_path) and os.path.exists(video_path):
            # Process audio and text
            V1 = extract_voice_embeddings(audio_path)
            V2 = extract_mfccs(audio_path)
            V3 = extract_combined_features(audio_path)
            V4 = extract_utterance_embeddings(utterance)

            # Process video
            A2 = process_video(video_path)

            data.append({
                "File Name": f"{file_name}",
                "V1": V1.tolist(),
                "V2": V2.tolist(),
                "V3": V3.tolist(),
                "V4": V4.tolist(),
                "A2": A2.tolist()
            })

        if (index + 1) % 100 == 0:
            elapsed_time = time.time() - start_time
            files_per_second = (index + 1) / elapsed_time
            estimated_total_time = total_files / files_per_second
            estimated_remaining_time = estimated_total_time - elapsed_time

            print(f"\nProcessed {index + 1}/{total_files} files")
            print(f"Elapsed time: {elapsed_time:.2f} seconds")
            print(f"Estimated time remaining: {estimated_remaining_time:.2f} seconds")
            print(f"Current file: {file_name}.mp4")

    df = pd.DataFrame(data)
    df.to_excel(output_excel_path, index=False)
    print(f"\nAll feature extraction completed and saved to {output_excel_path}")
    print(f"Total processing time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    process_all_data()