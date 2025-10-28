import os
import pandas as pd
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, f1_score, accuracy_score
from tqdm import tqdm

# ==============================
# 1️⃣ CONFIGURATION
# ==============================
TRAIN_CSV = "train.csv"
TEST_CSV = "test.csv"
META_CSV = "metadata.csv"
TRAIN_AUDIO_DIR = "train"
TEST_AUDIO_DIR = "test"

SAMPLE_RATE = 16000
NUM_MFCC = 20
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# 2️⃣ FEATURE EXTRACTION
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
# 3️⃣ DATASET CLASS
# ==============================
class VoiceDataset(Dataset):
    def __init__(self, csv_file, audio_dir, label_map=None, train_mode=True):
        self.data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.train_mode = train_mode
        self.label_map = label_map

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_name = self.data.iloc[idx]['File_name']
        file_path = os.path.join(self.audio_dir, file_name)
        feat = extract_features(file_path)

        if self.train_mode:
            label_id = int(self.data.iloc[idx]['target'])
            if self.label_map is not None:
                # Ensure label_id is valid according to metadata
                label_id = self.label_map.get(label_id, 0)
            return torch.tensor(feat, dtype=torch.float32), torch.tensor(label_id, dtype=torch.long)
        else:
            return torch.tensor(feat, dtype=torch.float32), file_name

# ==============================
# 4️⃣ MODEL
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
# 5️⃣ TRAINING & EVALUATION
# ==============================
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for x, y in loop:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"\n📘 Epoch [{epoch+1}/{epochs}] | Avg Loss: {avg_loss:.4f}")
        evaluate_model(model, val_loader, name="Validation")

def evaluate_model(model, loader, name="Validation"):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            predicted = torch.argmax(outputs, dim=1)
            preds.extend(predicted.cpu().numpy())
            labels.extend(y.cpu().numpy())

    print(f"\n📊 {name} Results:")
    print(classification_report(labels, preds, digits=3))
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    print(f"✅ Accuracy: {acc*100:.2f}% | 🎯 Macro F1: {f1:.4f}")
    return acc, f1

# ==============================
# 6️⃣ MAIN PIPELINE
# ==============================
def main():
    print("🔹 Loading dataset CSVs...")
    df_train = pd.read_csv(TRAIN_CSV)
    df_meta = pd.read_csv(META_CSV)

    # Build label map (Label ID → numeric index)
    unique_labels = sorted(df_meta['Label ID'].unique())
    label_map = {label_id: i for i, label_id in enumerate(unique_labels)}
    num_classes = len(unique_labels)
    print(f"📁 Found {num_classes} classes.")

    # Dataset split
    full_dataset = VoiceDataset(TRAIN_CSV, TRAIN_AUDIO_DIR, label_map, train_mode=True)
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    # Determine input feature size
    first_file = os.path.join(TRAIN_AUDIO_DIR, df_train.iloc[0]['File_name'])
    input_dim = len(extract_features(first_file))

    # Model, Loss, Optimizer
    model = VoiceClassifier(input_dim, num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print("\n🚀 Starting Training...")
    train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS)

    os.makedirs("saved_model", exist_ok=True)
    torch.save(model.state_dict(), "saved_model/voice_classifier.pth")
    print("\n✅ Model saved at saved_model/voice_classifier.pth")

    # ======================
    # 🔹 TEST EVALUATION
    # ======================
    print("\n🔹 Generating predictions on test set...")
    test_dataset = VoiceDataset(TEST_CSV, TEST_AUDIO_DIR, label_map, train_mode=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    preds, files = [], []
    with torch.no_grad():
        for x, file_names in tqdm(test_loader, desc="Predicting", leave=False):
            x = x.to(DEVICE)
            outputs = model(x)
            predicted = torch.argmax(outputs, dim=1).cpu().numpy()
            preds.extend(predicted)
            files.extend(file_names)

    # Reverse map: numeric idx → Label ID
    inv_label_map = {v: k for k, v in label_map.items()}
    final_preds = [inv_label_map[p] for p in preds]

    submission = pd.DataFrame({"File_name": files, "target": final_preds})
    submission.to_csv("submission.csv", index=False)
    print("\n📄 submission.csv generated successfully!")

# ==============================
# 7️⃣ RUN
# ==============================
if __name__ == "__main__":
    main()
