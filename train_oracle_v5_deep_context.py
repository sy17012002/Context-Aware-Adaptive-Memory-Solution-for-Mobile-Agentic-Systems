import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ==========================================
# 1. HYPERPARAMETERS (THE FIX)
# ==========================================
SEQ_LENGTH = 15       # Increased from 4 to 15! 
BATCH_SIZE = 32       
EMBEDDING_DIM = 64    
HIDDEN_DIM = 128      
EPOCHS = 50           
LEARNING_RATE = 0.0005 

# ==========================================
# 2. DATA PREP
# ==========================================
print("Loading data and generating deep context...")
df = pd.read_csv("full_ml_ready_app_usage.csv")
df = df[df['app_name'] != 'SpringBoard'].reset_index(drop=True)

def assign_location(row):
    h = row['hour_of_day']
    is_weekend = row['is_weekend']
    if h >= 22 or h <= 7: return 0  
    if is_weekend == 0 and 9 <= h <= 17: return 1  
    if is_weekend == 0 and (h == 8 or h == 18): return 2  
    return 3  

df['location_id'] = df.apply(assign_location, axis=1)

label_encoder = LabelEncoder()
df['app_id'] = label_encoder.fit_transform(df['app_name'])
num_unique_apps = len(label_encoder.classes_)

scaler = StandardScaler()
context_features = scaler.fit_transform(df[['duration_sec', 'hour_of_day', 'is_weekend', 'location_id']])

app_sequence = df['app_id'].values
X_apps, X_context, y = [], [], []

# Sliding window will now grab 15 apps at a time
for i in range(len(app_sequence) - SEQ_LENGTH):
    X_apps.append(app_sequence[i : i + SEQ_LENGTH])
    X_context.append(context_features[i : i + SEQ_LENGTH])
    y.append(app_sequence[i + SEQ_LENGTH])

X_apps_tensor = torch.tensor(X_apps, dtype=torch.long)
X_context_tensor = torch.tensor(X_context, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

class ContextAppDataset(Dataset):
    def __init__(self, X_apps, X_context, y):
        self.X_apps, self.X_context, self.y = X_apps, X_context, y
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X_apps[idx], self.X_context[idx], self.y[idx]

dataset = ContextAppDataset(X_apps_tensor, X_context_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ==========================================
# 3. BUILD THE ARCHITECTURE
# ==========================================
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1, bias=False)
        self.dropout = nn.Dropout(0.3) # Added to fight overfitting

    def forward(self, lstm_outputs):
        attn_weights = F.softmax(self.attention(lstm_outputs), dim=1) 
        attn_weights = self.dropout(attn_weights) # Randomly drop connections so it doesn't memorize
        context_vector = torch.sum(attn_weights * lstm_outputs, dim=1)
        return context_vector

class UltimateAttentionPredictor(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_context_features):
        super(UltimateAttentionPredictor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        lstm_input_dim = embed_dim + num_context_features
        
        self.lstm = nn.LSTM(lstm_input_dim, hidden_dim, batch_first=True, dropout=0.3, num_layers=2)
        self.attention = Attention(hidden_dim)
        
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, vocab_size)

    def forward(self, x_apps, x_context):
        embedded = self.embedding(x_apps) 
        combined_input = torch.cat((embedded, x_context), dim=2) 
        
        lstm_out, _ = self.lstm(combined_input)
        attn_out = self.attention(lstm_out)
        
        x = self.fc1(attn_out)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.fc2(x)
        return out

num_context_features = context_features.shape[1]
model = UltimateAttentionPredictor(num_unique_apps, EMBEDDING_DIM, HIDDEN_DIM, num_context_features)

# ==========================================
# 4. TRAINING LOOP
# ==========================================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5) # Added weight_decay (L2 regularization)

print(f"\nStarting Phase 5: Deep Context Training (Sequence Length: {SEQ_LENGTH})...")
for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch_X_apps, batch_X_context, batch_y in dataloader:
        optimizer.zero_grad()
        predictions = model(batch_X_apps, batch_X_context)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted_classes = torch.max(predictions, 1)
        correct += (predicted_classes == batch_y).sum().item()
        total += batch_y.size(0)

    acc = (correct / total) * 100
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {total_loss/len(dataloader):.4f} - Accuracy: {acc:.2f}%")

print("\n✅ Final Model saved.")
torch.save(model.state_dict(), "oracle_model_v5_deep_context.pth")