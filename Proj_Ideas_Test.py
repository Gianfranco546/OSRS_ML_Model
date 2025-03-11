import torch
import torch.nn as nn
import math
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler

prices = 1
volumes = 1
train_indices = 1

# Prices: Convert to log returns
price_returns = np.log(prices[1:]) - np.log(prices[:-1])

# Volumes: Log transform
volume_log = np.log1p(volumes)

# Fit scalers on training data only
price_scaler = StandardScaler().fit(price_returns[train_indices])
volume_scaler = RobustScaler().fit(volume_log[train_indices].reshape(-1, 1))

# Transform
X_price = price_scaler.transform(price_returns)
X_volume = volume_scaler.transform(volume_log.reshape(-1, 1))

# Combine into model input
X = np.concatenate([X_price, X_volume], axis=1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return x

class PricePredictor(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=6, pred_horizon=1):
        super().__init__()
        self.embedding = nn.Linear(2, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=512, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.regressor = nn.Linear(d_model, pred_horizon)

    def forward(self, src):
        # src shape: (batch_size, seq_len, 2)
        src = self.embedding(src)  # (batch_size, seq_len, d_model)
        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2)  # (seq_len, batch_size, d_model)
        output = self.transformer(src)  # (seq_len, batch_size, d_model)
        output = output[-1]  # Use last timestep for prediction
        return self.regressor(output)  # (batch_size, pred_horizon)
    
def train(model, dataloader, optimizer, criterion):
    model.train()
    for batch in dataloader:
        x, y = batch  # x: (batch_size, seq_len, 2), y: (batch_size, pred_horizon)
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
# Predict scaled returns
model = 1
inputs = 1
last_known_price = 1
pred_returns_scaled = model(inputs)

# Inverse transform
pred_returns = price_scaler.inverse_transform(pred_returns_scaled)

# Convert returns to prices
pred_prices = np.exp(np.cumsum(pred_returns) + np.log(last_known_price))