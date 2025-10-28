import os
import pandas as pd
import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score, accuracy_score
from tqdm import tqdm

# ==============================
# CONFIGURATION
# ==============================
TRAIN_CSV = "train.csv"
META_CSV = "metadata.csv"
TRAIN_AUDIO_DIR = "train"
MODEL_PATH = "saved_model/voice_classifier.pth"

SAMPLE_RATE = 16000
NUM_MFCC = 20
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
#  FEATURE EXTRACTION
# ==============================
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=NUM_MFCC)
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        feat = np.concatenate((np.mean(mfcc, axis=1), np.mean(spec_contrast, axis=1)))
        return feat
    except Exception as e:
        print(f"⚠️ Error processing {file_path}: {e}")
        return np.zeros(NUM_MFCC + 7)

# ==============================
#  DATASET CLASS
# ==============================
class VoiceDataset(Dataset):
    def __init__(self, csv_file, audio_dir, label_map):
        self.data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.label_map = label_map

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_name = self.data.iloc[idx]['File_name']
        file_path = os.path.join(self.audio_dir, file_name)
        feat = extract_features(file_path)
        label_id = int(self.data.iloc[idx]['target'])
        label_id = self.label_map.get(label_id, 0)
        return torch.tensor(feat, dtype=torch.float32), torch.tensor(label_id, dtype=torch.long)

# ==============================
#  MODEL
# ==============================
class VoiceClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(VoiceClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ==============================
#  EVALUATION FUNCTION
# ==============================
def evaluate_model(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating", ncols=100):
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            predicted = torch.argmax(outputs, dim=1)
            preds.extend(predicted.cpu().numpy())
            labels.extend(y.cpu().numpy())

    print("\n Evaluation Report:")
    print(classification_report(labels, preds, digits=3))
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    print(f" Accuracy: {acc*100:.2f}%")
    print(f" Macro F1: {f1*100:.4f}%")

# ==============================
# 6 MAIN
# ==============================
def main():
    print("🔹 Loading metadata...")
    df_meta = pd.read_csv(META_CSV)
    df_train = pd.read_csv(TRAIN_CSV)

    # Label map
    unique_labels = sorted(df_meta['Label ID'].unique())
    label_map = {label_id: i for i, label_id in enumerate(unique_labels)}
    num_classes = len(unique_labels)

    # Determine input dimension
    first_file = os.path.join(TRAIN_AUDIO_DIR, df_train.iloc[0]['File_name'])
    input_dim = len(extract_features(first_file))

    print("🔹 Loading evaluation dataset...")
    eval_dataset = VoiceDataset(TRAIN_CSV, TRAIN_AUDIO_DIR, label_map)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("🔹 Loading saved model...")
    model = VoiceClassifier(input_dim, num_classes).to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    print(" Running Evaluation...")
    evaluate_model(model, eval_loader)

if __name__ == "__main__":
    main()
