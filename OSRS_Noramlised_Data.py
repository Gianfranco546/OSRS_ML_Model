import sqlite3
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict
import torch
from sklearn.preprocessing import StandardScaler, RobustScaler
import random
torch.manual_seed(42)
#np.random.seed(42)
random.seed(42)
torch.cuda.manual_seed_all(42)


class OSRSMarketDataset(Dataset):
    def __init__(self, price_scaler=None, volume_scaler=None):
        self.all_sequences = []
        self.price_scaler = price_scaler
        self.volume_scaler = volume_scaler
        
        # Connect to database and load raw data
        conn = sqlite3.connect('osrsmarketdata.sqlite')
        cur = conn.cursor()
        cur.execute("""SELECT typeid, timestamp, avgHighPrice, highPriceVolume,
                      avgLowPrice, lowPriceVolume FROM marketdata 
                      WHERE interval = 3600 ORDER BY typeid, timestamp;""")
        all_data = cur.fetchall()
        
        # Process data by item
        item_data = defaultdict(list)
        for row in all_data:
            typeid, ts, high_p, high_v, low_p, low_v = row
            item_data[typeid].append((ts, high_p, high_v, low_p, low_v))

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
        
        # Process each item's data
        for typeid, data in item_data.items():
            # Forward fill prices
            start_index = 0
            for idx, entry in enumerate(data):
                if entry[1] is not None and entry[3] is not None:
                    start_index = idx
                    break
            filtered_data_list = data[start_index:]
            filled_data = list(forward_fill_prices(filtered_data_list))
            
            # Skip items with insufficient data
            total_hours = len(filled_data)
            if total_hours < 35:  # Need 35 points to make 34 returns
                continue
            
            zero_high_volume_count = sum(1 for entry in filled_data if entry[2] == 0)
            zero_low_volume_count = sum(1 for entry in filled_data if entry[4] == 0)
            high_volume_zero_ratio = zero_high_volume_count / total_hours
            low_volume_zero_ratio = zero_low_volume_count / total_hours
            if high_volume_zero_ratio >= 0.05 or low_volume_zero_ratio >= 0.05:
                continue  # Exclude this item
            
            # Convert to numpy arrays
            high_prices = np.array([x[1] for x in filled_data], dtype=np.float32)
            low_prices = np.array([x[3] for x in filled_data], dtype=np.float32)
            high_volumes = np.array([x[2] for x in filled_data], dtype=np.float32)
            low_volumes = np.array([x[4] for x in filled_data], dtype=np.float32)

            # Calculate log returns (skip first element)
            log_high_returns = np.log(high_prices[1:]/high_prices[:-1])
            log_low_returns = np.log(low_prices[1:]/low_prices[:-1])

            # Transform volumes (log1p and skip first element)
            log_high_vol = np.log1p(high_volumes[1:])
            log_low_vol = np.log1p(low_volumes[1:])

            # Create feature matrix
            features = np.column_stack((log_high_returns, log_low_returns,
                                      log_high_vol, log_low_vol))

            # Create sequences of 34 timesteps
            for i in range(len(features) - 33):
                seq = features[i:i+34]
                seq = seq.astype(np.float32)
                price_features = seq[:, :2]
                combined_mean = price_features.mean()    # average of all values in both columns
                combined_std = price_features.std() 
                self.all_sequences.append(seq.astype(np.float32))

        # Fit scalers if not provided
        if self.price_scaler is None or self.volume_scaler is None:
            all_features = np.concatenate(self.all_sequences)
            price_features = all_features[:, :2]
            vol_features = all_features[:, 2:]
            
            self.price_scaler = StandardScaler().fit(price_features)
            self.volume_scaler = RobustScaler().fit(vol_features)

        # Apply scaling
        self.all_sequences = [
            np.column_stack((
                self.price_scaler.transform(seq[:, :2]),
                self.volume_scaler.transform(seq[:, 2:])
            )) for seq in self.all_sequences
        ]

    def __len__(self):
        return len(self.all_sequences)

    def __getitem__(self, idx):
        return torch.from_numpy(self.all_sequences[idx]).float()

# Usage example
from torch.utils.data import DataLoader
dataset = OSRSMarketDataset()
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
count = 0
for batch in dataloader:
    print(batch)
    count+=1
    if count == 50:
        break