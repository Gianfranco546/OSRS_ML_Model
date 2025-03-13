import sqlite3
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import math
from tqdm import tqdm
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
        cur.execute("SELECT typeid, timestamp, avgHighPrice, highPriceVolume, avgLowPrice, lowPriceVolume FROM marketdata WHERE interval = 21600 ORDER BY typeid, timestamp;")
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
                    price_std = 1
                if vol_std == 0:
                    vol_std = 1                
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
            # Check volume criteria: 10% or more zeros in high or low volume
            zero_high_volume_count = sum(1 for entry in filled_data_list if entry[2] == 0)
            zero_low_volume_count = sum(1 for entry in filled_data_list if entry[4] == 0)
            high_volume_zero_ratio = zero_high_volume_count / total_hours
            low_volume_zero_ratio = zero_low_volume_count / total_hours
            if high_volume_zero_ratio >= 0.1 or low_volume_zero_ratio >= 0.1:
                continue  # Exclude this item

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
        self.self_attn = CustomMultiheadAttention(query_emb_dim = d_model, kv_emb_dim = d_model, hidden_dim = d_model, num_heads = num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention
        attn_output = self.self_attn(x, x)
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
class CustomMultiheadAttention(nn.Module):
    def __init__(self, query_emb_dim, kv_emb_dim, hidden_dim, num_heads, is_causal = False):
        super().__init__()
        self.query_emb_dim = query_emb_dim
        self.kv_emb_dim = kv_emb_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"

        # Projection layers for queries, keys, and values
        self.W_q = nn.Linear(query_emb_dim, hidden_dim)
        self.W_k = nn.Linear(kv_emb_dim, hidden_dim)
        self.W_v = nn.Linear(kv_emb_dim, hidden_dim)
        
        # Optional output projection back to query embedding dimension
        self.out_proj = nn.Linear(hidden_dim, query_emb_dim)
        self.is_causal = is_causal
        
    def forward(self, query_input, key_value_input, key_padding_mask=None):
        # query_input shape: (batch_size, query_seq_len, query_emb_dim)
        # key_value_input shape: (batch_size, kv_seq_len, kv_emb_dim)
        batch_size, query_seq_len, _ = query_input.size()
        _, kv_seq_len, _ = key_value_input.size()
        
        # Project inputs to Q, K, V
        Q = self.W_q(query_input)  # (batch, query_seq_len, hidden_dim)
        K = self.W_k(key_value_input)  # (batch, kv_seq_len, hidden_dim)
        V = self.W_v(key_value_input)  # (batch, kv_seq_len, hidden_dim)
        
        # Reshape to [batch, seq_len, num_heads, head_dim] and transpose to [batch, num_heads, seq_len, head_dim]
        Q = Q.view(batch_size, query_seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, kv_seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, kv_seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Compute attention scores (QK^T / sqrt(d_k))
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply padding mask if provided
        if key_padding_mask is not None:
            # Reshape mask to (batch_size, 1, 1, kv_seq_len)
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float('-inf'))
       
        # Create a causal mask for self-attention to prevent information leakage
        # Conditional causal masking
        if self.is_causal:
            mask = torch.triu(torch.ones(query_seq_len, kv_seq_len), 
                           diagonal=1).to(Q.device)
            mask = mask.masked_fill(mask == 1, float('-inf'))
            scores = scores + mask
        
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, V)  # [batch, num_heads, query_seq_len, head_dim]
        
        # Transpose and reshape back to [batch, query_seq_len, hidden_dim]
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, query_seq_len, self.hidden_dim)
        
        # Project back to original query embedding dimension
        output = self.out_proj(output)
        
        return output

#Hyperparameters
input_dim = 4
d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 6
dropout = 0.1
output_dim = 16

batch_size = 8
num_workers = 4
epochs = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Usage example
dataset = OSRSMarketDataset()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

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
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(epochs):
    running_loss = 0
    model.train()
    train_progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
    for batch in train_progress:
        batch = batch.to(device)  # batch: (batch_size, seq_len, 4)
        x = batch[:, :120, :]  # (batch_size, 30, 4)
        # Target: Last 4 timesteps of [highPrice, lowPrice]
        y = batch[:, 120:, :2]  # (batch_size, 4, 2)
        
        # Flatten targets for loss calculation
        y_flat = y.reshape(-1, 16)  # (batch_size, 8)
        pred = model(x)
        loss = criterion(pred, y_flat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        train_progress.set_postfix({'loss': f"{loss.item():.4f}"})
    train_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f}")