import sqlite3
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import random_split, Dataset, DataLoader
import torch.nn as nn
import math
from tqdm import tqdm
import wandb
import random
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

class OSRSMarketDataset(Dataset):
    def __init__(self):
        self.all_sequences = []
        # Connect to the database
        conn = sqlite3.connect('osrsmarketdata.sqlite')
        cur = conn.cursor()
        # Fetch all relevant data for 6-hour interval
        cur.execute("SELECT typeid, timestamp, avgHighPrice, highPriceVolume, avgLowPrice, lowPriceVolume FROM marketdata WHERE interval = 86400 ORDER BY typeid, timestamp;")
        all_data = cur.fetchall()
        # Group data by typeid
        item_data = defaultdict(list)
        for row in all_data:
            typeid, timestamp, avgHighPrice, highPriceVolume, avgLowPrice, lowPriceVolume = row
            item_data[typeid].append((timestamp, avgHighPrice, highPriceVolume, avgLowPrice, lowPriceVolume))
        
        # Function to forward-fill missing prices
        def forward_fill_prices(item_data_list):
            last_high_price = None
            last_low_price = None
            for entry in item_data_list:
                timestamp, high_price, high_volume, low_price, low_volume = entry
                if high_price is None:
                    high_price = last_high_price
                else:
                    last_high_price = high_price
                if low_price is None:
                    low_price = last_low_price
                else:
                    last_low_price = low_price
                yield (timestamp, high_price, high_volume, low_price, low_volume)
        
        # Function to create sequences of 128 consecutive 6 hours
        def create_sequences(item_data_list):
            sequences = []
            n = len(item_data_list)
            for i in range(n - 128):
                # Check if the first hour of the sequence has highPriceVolume or lowPriceVolume as 0
                first_hour = item_data_list[i]
                if first_hour[2] == 0 or first_hour[4] == 0:
                    continue  # Skip this sequence

                sequence_data = []
                for j in range(i, i+129):
                    _, avgHighPrice, highPriceVolume, avgLowPrice, lowPriceVolume = item_data_list[j]
                    sequence_data.append([avgHighPrice, avgLowPrice, highPriceVolume, lowPriceVolume])

                sequence_data_np = np.array(sequence_data)

                # Extract the highPriceVolume and lowPriceVolume columns (indices 2 and 3)
                high_volume = sequence_data_np[:, 2]
                low_volume = sequence_data_np[:, 3]

                # Calculate the fraction of zeros in each column
                fraction_high_zero = np.mean(high_volume == 0)
                fraction_low_zero = np.mean(low_volume == 0)
                
                if fraction_high_zero > 0.01 or fraction_low_zero > 0.01:
                    continue


                high_prices = np.array([x[0] for x in sequence_data], dtype=np.float32)
                low_prices = np.array([x[1] for x in sequence_data], dtype=np.float32)
                high_volumes = np.array([x[2] for x in sequence_data], dtype=np.float32)
                low_volumes = np.array([x[3] for x in sequence_data], dtype=np.float32)
                
                # Calculate log returns (skip first element)
                log_high_returns = np.log(high_prices[1:]/high_prices[:-1])
                log_low_returns = np.log(low_prices[1:]/low_prices[:-1])

                # Transform volumes (log1p and skip first element)
                log_high_vol = np.log1p(high_volumes[1:])
                log_low_vol = np.log1p(low_volumes[1:])

                # Create feature matrix
                features = np.column_stack((log_high_returns, log_low_returns,
                                        log_high_vol, log_low_vol)) 
                price_features = features[:120, :2]
                price_mean = price_features.mean()    # average of all values in both columns
                price_std = price_features.std()
                vol_features = features[:120, 2:]            
                vol_mean = vol_features.mean()
                vol_std = vol_features.std()
                if price_std == 0:
                    price_std = 1000
                if vol_std == 0:
                    vol_std = 1000                
                sequence =     np.column_stack((
        (features[:, :2] - price_mean) / price_std,  # Apply same scaling to both columns
        (features[:, 2:] - vol_mean) / vol_std       # Apply volume scaling as before
    ))
                
                sequences.append(sequence)
            return sequences
        
        # Process each item
        for typeid, data_list in item_data.items():
            # Find the first entry where both avgHighPrice and avgLowPrice are not None
            start_index = 0
            for idx, entry in enumerate(data_list):
                if entry[1] is not None and entry[3] is not None:
                    start_index = idx
                    break
            # Take the sublist from start_index onward
            filtered_data_list = data_list[start_index:]
            # Forward-fill missing prices in this sublist
            filled_data_list = list(forward_fill_prices(filtered_data_list))
            # Check if there's enough data to create sequences
            total_hours = len(filled_data_list)
            if total_hours < 129:
                continue  # Not enough data

            seq = np.array(filled_data_list, dtype=np.float32)

            # Skip sequences with volume issues
            if (np.mean(seq[:, 2] == 0) > 0.005 or 
                np.mean(seq[:, 4] == 0) > 0.005):
                continue

            recent_low_price = seq[-1, 3]  # Most recent low price
            average_volume = np.mean(seq[:, 2] + seq[:, 4])  # Avg(hvol + lvol)
            if average_volume <= 96:
                continue
            if (average_volume * recent_low_price) <= 1_000_000:
                continue

            # New Check 2: Buylimit * recent low price > 20M
            # Get buylimit from Mapping table
            cur.execute("SELECT buylimit FROM Mapping WHERE typeid=?", (typeid,))
            buylimit_row = cur.fetchone()
            
            # Skip if no buylimit found or invalid value
            if not buylimit_row or buylimit_row[0] == 0:
                continue
                
            buylimit = buylimit_row[0]
            if buylimit != None:
                if (buylimit * recent_low_price) <= 1_000_000:
                    continue

            # Create sequences for this item
            sequences = create_sequences(filled_data_list)
            self.all_sequences.extend(sequences)
        
        conn.close()
    
    def __len__(self):
        return len(self.all_sequences)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.all_sequences[idx]).float()

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=130):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, emb_dim)
        return x + self.pe[:x.size(1)]
    
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention
        attn_output, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # Feed-forward
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        return x
    
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, d_ff, num_layers, dropout, output_dim):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.output_layer = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x is (batch_size, seq_len, input_dim)
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        for layer in self.encoder_layers:
            x = layer(x)
        # Take the last position's output to predict the next price
        x = x[:, -1, :]  # (batch_size, d_model)
        x = self.output_layer(x)  # (batch_size, 1)
        return x

#Hyperparameters
input_dim = 4
d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 6
dropout = 0.1
output_dim = 16

batch_size = 32  #32
num_workers = 4  #4
epochs = 7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Usage example
dataset = OSRSMarketDataset()
train_size = int(0.95 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
len_train = len(train_loader)
len_val = len(val_loader)

model = TransformerModel(
    input_dim = input_dim,
    d_model = d_model,
    num_heads = num_heads,
    d_ff = d_ff,
    num_layers = num_layers,
    dropout = dropout,
    output_dim = output_dim
).to(device)

criterion = nn.MSELoss()  # Now compares 4 output values vs 4 targets
#criterion = nn.HuberLoss(delta=1.35)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

wandb.init(project="OSRS_PricePredictor_Model")

for epoch in range(epochs):
    running_loss = 0
    train_skipped = 0
    model.train()
    train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
    for batch in train_progress:
        batch = batch.to(device)  # batch: (batch_size, seq_len, 4)
        x = batch[:, :120, :]  # (batch_size, 30, 4)
        # Target: Last 4 timesteps of [highPrice, lowPrice]
        y = batch[:, 120:, :2]  # (batch_size, 4, 2)
        
        # Flatten targets for loss calculation
        y_flat = y.reshape(-1, 16)  # (batch_size, 8)
        pred = model(x)
        loss = criterion(pred, y_flat)
        
        if loss.item() > 100:
            train_skipped += 1
            train_progress.set_postfix({'loss': f"{loss.item():.4f}"})
            continue       
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        running_loss += loss.item()
        wandb.log({"train_loss": loss.item()})
        train_progress.set_postfix({'loss': f"{loss.item():.4f}"})
    train_loss = running_loss / (len_train - train_skipped)
    
    # Validation Loop
    model.eval()  # Set model to evaluation mode (disables dropout, batchnorm, etc.)
    val_loss = 0.0
    val_skipped = 0
    val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
    with torch.no_grad():  # No gradient updates for validation
        for batch in val_progress:
            batch = batch.to(device)  # batch: (batch_size, seq_len, 4)
            x = batch[:, :120, :]  # (batch_size, 30, 4)
            # Target: Last 4 timesteps of [highPrice, lowPrice]
            y = batch[:, 120:, :2]  # (batch_size, 4, 2)
            
            # Flatten targets for loss calculation
            y_flat = y.reshape(-1, 16)  # (batch_size, 8)
            pred = model(x)
            loss = criterion(pred, y_flat)
            
            if loss.item() > 100:
                val_skipped += 1
                val_progress.set_postfix({'loss': f"{loss.item():.4f}"})
                continue              
            
            val_loss += loss.item()
            # Update validation progress bar
            val_progress.set_postfix({'val_loss': f"{loss.item():.4f}"})
    val_loss /= (len_val - val_skipped)
    wandb.log({"epoch": epoch + 1, "train_epoch_loss": train_loss, "val_epoch_loss": val_loss})
    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    print(f"Epoch {epoch+1} | Train Skipped: {train_skipped} | Val Loss: {val_skipped}")
    model_filename = f"OSRS_PricePredictorLongMSE_epoch_{epoch+1}.pth"
    torch.save(model.state_dict(), model_filename)

torch.save(model.state_dict(), "OSRS_PricePredictorLongMSE.pth")
wandb.finish()