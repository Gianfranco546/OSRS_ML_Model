import sqlite3
import numpy as np
from collections import defaultdict
import torch
from torch.nn.functional import cosine_similarity
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
num_layers = 8
dropout = 0.1
output_dim = 16

batch_size = 32
num_workers = 4
epochs = 7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
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
        embeddings = x[:, -1, :]  # (batch_size, d_model)
        output = self.output_layer(embeddings)
        return output, embeddings  # Return both predictions and embeddings

def analyze_market():
    # Initialize model and load weights
    model = TransformerModel(input_dim=input_dim, d_model=d_model, num_heads=num_heads, 
                            d_ff=d_ff, num_layers=num_layers, dropout=dropout, 
                            output_dim=output_dim).to(device)
    model.load_state_dict(torch.load("OSRS_PricePredictorLong6hMSE_epoch_1.pth"))
    model.eval()

    conn = sqlite3.connect('osrsmarketdata.sqlite')
    cur = conn.cursor()

    # Get all item IDs with recent data
    cur.execute("""SELECT DISTINCT typeid FROM marketdata 
                WHERE interval=21600 
                AND timestamp >= (SELECT MAX(timestamp)-21600*488 
                                FROM marketdata 
                                WHERE interval=21600)""")
    item_ids = [row[0] for row in cur.fetchall()]
    
    results = []
    all_embeddings = []
    item_info = []

    for typeid in tqdm(item_ids, desc="Analyzing Items"):
        if typeid == 2434:
            print("Yes1")
        # Get last 128 timesteps (120 for input + 8 prediction)
        cur.execute("""SELECT timestamp, avgHighPrice, highPriceVolume, 
                      avgLowPrice, lowPriceVolume 
                      FROM marketdata 
                      WHERE typeid=? AND interval=21600 
                      ORDER BY timestamp DESC 
                      LIMIT 488""", (typeid,))
        data = cur.fetchall()[::-1]  # Reverse to chronological order

        # Skip items with insufficient data
        if len(data) < 488:
            continue
        if typeid == 2434:
            print("Yes2")

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
        if (average_volume * recent_low_price) <= 1_000_000:
            continue
        if typeid == 2434:
            print("Yes3")

        # New Check 2: Buylimit * recent low price > 20M
        # Get buylimit from Mapping table
        cur.execute("SELECT buylimit FROM Mapping WHERE typeid=?", (typeid,))
        buylimit_row = cur.fetchone()
        
        # Skip if no buylimit found or invalid value
        if not buylimit_row or buylimit_row[0] == 0:
            continue
        if typeid == 2434:
            print("Yes4") 
        buylimit = buylimit_row[0]
        if buylimit != None:
            if (buylimit * recent_low_price) <= 1_000_000:
                continue
            
        if typeid == 2434:
            print("Yes5")

        # Calculate features (same as dataset preprocessing)
        high_prices = seq[:, 0]
        low_prices = seq[:, 1]
        
        # Get normalization parameters from first 120 steps
        price_mean = np.mean(np.log(high_prices[1:481]/high_prices[:480]))
        price_std = np.std(np.log(high_prices[1:481]/high_prices[:480]))
        vol_mean = np.mean(np.log1p(seq[:480, 2]))
        vol_std = np.std(np.log1p(seq[:480, 2]))

        # Create input sequence
        input_seq = []
        for i in range(480, 488):
            high_return = np.log(high_prices[i]/high_prices[i-1])
            low_return = np.log(low_prices[i]/low_prices[i-1])
            hvol = np.log1p(seq[i, 2])
            lvol = np.log1p(seq[i, 3])
            
            norm_high = (high_return - price_mean) / price_std
            norm_low = (low_return - price_mean) / price_std
            norm_hvol = (hvol - vol_mean) / vol_std
            norm_lvol = (lvol - vol_mean) / vol_std
            
            input_seq.append([norm_high, norm_low, norm_hvol, norm_lvol])
        input_tensor = torch.tensor([input_seq], dtype=torch.float32).to(device)
        with torch.no_grad():
            pred, embeddings = model(input_tensor)  # Get both predictions and embeddings
            pred = pred.cpu().numpy().reshape(8, 2)
            embedding = embeddings.cpu().numpy().flatten()

        # ... [existing price decoding and result storage] ...
        cur.execute("SELECT name FROM Mapping WHERE typeid=?", (typeid,))
        name = cur.fetchone()[0]
        results.append({
            'typeid': typeid,
            'name': name,
            # ... [existing fields] ...,
            'embedding': embedding
        })
        if typeid == 2434:
            print("Yes")
        all_embeddings.append(embedding)
        item_info.append((typeid, name))

    # ... [existing ranking print code] ...

    # New similarity search functionality
    while True:
        target_name = input("\nEnter item name for similarity search (or 'exit' to quit): ").strip()
        if target_name.lower() == 'exit':
            break
            
        # Find target item
        target_item = next((item for item in results if item['name'].lower() == target_name.lower()), None)
        if not target_item:
            print(f"Item '{target_name}' not found in analyzed items")
            continue

        # Calculate cosine similarities
        target_embedding = torch.tensor(target_item['embedding'])
        embeddings_tensor = torch.tensor(np.array(all_embeddings))
        
        similarities = cosine_similarity(
            target_embedding.unsqueeze(0),
            embeddings_tensor,
            dim=1
        ).numpy().flatten()

        # Create sorted list of (similarity, name, typeid)
        ranked_items = sorted(
            zip(similarities, [i[1] for i in item_info], [i[0] for i in item_info]),
            key=lambda x: x[0], 
            reverse=True
        )

        # Exclude the target itself and show top 10
        print(f"\nItems most similar to '{target_name}':")
        for sim, name, typeid in ranked_items[1:11]:  # Skip first (self)
            print(f"- {name} (ID: {typeid})")

    conn.close()

# Run the analysis
analyze_market()