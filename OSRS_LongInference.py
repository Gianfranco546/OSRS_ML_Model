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

#Hyperparameters
input_dim = 4
d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 12
dropout = 0.1
output_dim = 16

batch_size = 32
num_workers = 4
epochs = 7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

def analyze_market():
    # Initialize model and load weights
    model = TransformerModel(input_dim=input_dim, d_model=d_model, num_heads=num_heads, 
                            d_ff=d_ff, num_layers=num_layers, dropout=dropout, 
                            output_dim=output_dim).to(device)
    model.load_state_dict(torch.load("OSRS_PricePredictorLong1_epoch_3.pth"))
    model.eval()

    conn = sqlite3.connect('osrsmarketdata.sqlite')
    cur = conn.cursor()

    # Get all item IDs with recent data
    cur.execute("""SELECT DISTINCT typeid FROM marketdata 
                WHERE interval=21600 
                AND timestamp >= (SELECT MAX(timestamp)-21600*128 
                                FROM marketdata 
                                WHERE interval=21600)""")
    item_ids = [row[0] for row in cur.fetchall()]

    results = []

    for typeid in tqdm(item_ids, desc="Analyzing Items"):
        # Get last 128 timesteps (120 for input + 8 prediction)
        cur.execute("""SELECT timestamp, avgHighPrice, highPriceVolume, 
                      avgLowPrice, lowPriceVolume 
                      FROM marketdata 
                      WHERE typeid=? AND interval=21600 
                      ORDER BY timestamp DESC 
                      LIMIT 128""", (typeid,))
        data = cur.fetchall()[::-1]  # Reverse to chronological order

        # Skip items with insufficient data
        if len(data) < 128:
            continue

        # Forward fill and process sequence
        processed = []
        last_high = None
        last_low = None
        for row in data:
            ts, high, hvol, low, lvol = row
            high = last_high if high is None else high
            low = last_low if low is None else low
            processed.append((high, low, hvol, lvol))
            last_high, last_low = high, low

        # Convert to numpy array
        seq = np.array(processed, dtype=np.float32)

        # Skip sequences with volume issues
        if (np.mean(seq[:, 2] == 0) > 0.01 or 
            np.mean(seq[:, 3] == 0) > 0.01):
            continue

        recent_low_price = seq[-1, 1]  # Most recent low price
        average_volume = np.mean(seq[:, 2] + seq[:, 3])  # Avg(hvol + lvol)
        if (average_volume * recent_low_price) <= 20_000_000:
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
            if (buylimit * recent_low_price) <= 20_000_000:
                continue

        # Calculate features (same as dataset preprocessing)
        high_prices = seq[:, 0]
        low_prices = seq[:, 1]
        
        # Get normalization parameters from first 120 steps
        price_mean = np.mean(np.log(high_prices[1:121]/high_prices[:120]))
        price_std = np.std(np.log(high_prices[1:121]/high_prices[:120]))
        vol_mean = np.mean(np.log1p(seq[:120, 2]))
        vol_std = np.std(np.log1p(seq[:120, 2]))

        # Create input sequence
        input_seq = []
        for i in range(120, 128):
            high_return = np.log(high_prices[i]/high_prices[i-1])
            low_return = np.log(low_prices[i]/low_prices[i-1])
            hvol = np.log1p(seq[i, 2])
            lvol = np.log1p(seq[i, 3])
            
            norm_high = (high_return - price_mean) / price_std
            norm_low = (low_return - price_mean) / price_std
            norm_hvol = (hvol - vol_mean) / vol_std
            norm_lvol = (lvol - vol_mean) / vol_std
            
            input_seq.append([norm_high, norm_low, norm_hvol, norm_lvol])

        # Convert to tensor and predict
        input_tensor = torch.tensor([input_seq], dtype=torch.float32).to(device)
        with torch.no_grad():
            pred = model(input_tensor).cpu().numpy().reshape(8, 2)

        # Decode predictions
        current_high = high_prices[-1]
        current_low = low_prices[-1]
        
        # Convert returns to absolute prices
        pred_high = [current_high * np.exp((pred[i,0] * price_std) + price_mean) 
                    for i in range(8)]
        pred_low = [current_low * np.exp((pred[i,1] * price_std) + price_mean) 
                   for i in range(8)]

        max_high = max(pred_high)
        max_low = max(pred_low)
        min_high = min(pred_high)
        min_low = min(pred_low)
        
        high_pct = (max_high - current_high) / current_high * 100
        low_pct = (max_low - current_low) / current_low * 100
        
        high_pct1 = (min_high - current_high) / current_high * 100
        low_pct1 = (min_low - current_low) / current_low * 100

        # Get item name
        cur.execute("SELECT name FROM Mapping WHERE typeid=?", (typeid,))
        name = cur.fetchone()[0]
        if typeid == 29993:
            print(f"{name}: {high_pct:.2f}% "
                f"({current_high:.0f} → {max_high:.0f})")
            print(f"{name}: {low_pct:.2f}% "
                f"({current_low:.0f} → {max_low:.0f})")
        
        results.append({
            'typeid': typeid,
            'name': name,
            'current_high': current_high,
            'current_low': current_low,
            'max_high': max_high,
            'max_low': max_low,
            'high_pct': high_pct,
            'low_pct': low_pct,
            'min_high': min_high,
            'min_low': min_low,
            'high_pct1': high_pct1,
            'low_pct1': low_pct1
        })

    # Generate rankings
    high_rank = sorted(results, key=lambda x: x['high_pct'], reverse=True)[:10]
    low_rank = sorted(results, key=lambda x: x['low_pct'], reverse=True)[:10]

    # Print results
    print("\nTop 10 High Price Gainers:")
    for item in high_rank:
        print(f"{item['name']}: {item['high_pct']:.2f}% "
              f"({item['current_high']:.0f} → {item['max_high']:.0f})")

    print("\nTop 10 Low Price Gainers:")
    for item in low_rank:
        print(f"{item['name']}: {item['low_pct']:.2f}% "
              f"({item['current_low']:.0f} → {item['max_low']:.0f})")
        
    high_rank = sorted(results, key=lambda x: x['high_pct1'])[:10]
    low_rank = sorted(results, key=lambda x: x['low_pct1'])[:10]
    
    print("\nLowest 10 High Price Gainers:")
    for item in high_rank:
        print(f"{item['name']}: {item['high_pct1']:.2f}% "
            f"({item['current_high']:.0f} → {item['min_high']:.0f})")

    print("\nLowest 10 Low Price Gainers:")
    for item in low_rank:
        print(f"{item['name']}: {item['low_pct1']:.2f}% "
            f"({item['current_low']:.0f} → {item['min_low']:.0f})")
    conn.close()

# Run the analysis
analyze_market()