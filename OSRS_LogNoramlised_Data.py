import sqlite3
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict
import torch

class OSRSMarketDataset(Dataset):
    def __init__(self):
        self.all_sequences = []
        # Connect to the database
        conn = sqlite3.connect('osrsmarketdata.sqlite')
        cur = conn.cursor()
        # Fetch all relevant data for 1-hour interval
        cur.execute("SELECT typeid, timestamp, avgHighPrice, highPriceVolume, avgLowPrice, lowPriceVolume FROM marketdata WHERE interval = 3600 ORDER BY typeid, timestamp;")
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
        
        # Function to create sequences of 34 consecutive hours
        def create_sequences(item_data_list):
            sequences = []
            n = len(item_data_list)
            for i in range(n - 33):
                # Check if the first hour of the sequence has highPriceVolume or lowPriceVolume as 0
                first_hour = item_data_list[i]
                if first_hour[2] == 0 or first_hour[4] == 0:
                    continue  # Skip this sequence
                start_timestamp = item_data_list[i][0]
                end_timestamp = item_data_list[i+33][0]
                if end_timestamp == start_timestamp + 3600 * 33:
                    sequence_data = []
                    for j in range(i, i+34):
                        _, avgHighPrice, highPriceVolume, avgLowPrice, lowPriceVolume = item_data_list[j]

                        sequence_data.append([avgHighPrice, avgLowPrice, highPriceVolume, lowPriceVolume])
                    sequences.append(np.array(sequence_data, dtype=np.float32))
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
            if total_hours < 35:
                continue  # Not enough data
            # Check volume criteria: 10% or more zeros in high or low volume
            zero_high_volume_count = sum(1 for entry in filled_data_list if entry[2] == 0)
            zero_low_volume_count = sum(1 for entry in filled_data_list if entry[4] == 0)
            high_volume_zero_ratio = zero_high_volume_count / total_hours
            low_volume_zero_ratio = zero_low_volume_count / total_hours
            if high_volume_zero_ratio >= 0.1 or low_volume_zero_ratio >= 0.1:
                continue  # Exclude this item
            # Convert to numpy arrays
            high_prices = np.array([x[1] for x in filled_data_list], dtype=np.float32)
            low_prices = np.array([x[3] for x in filled_data_list], dtype=np.float32)
            high_volumes = np.array([x[2] for x in filled_data_list], dtype=np.float32)
            low_volumes = np.array([x[4] for x in filled_data_list], dtype=np.float32)

            # Calculate log returns (skip first element)
            log_high_returns = np.log(high_prices[1:]/high_prices[:-1])
            log_low_returns = np.log(low_prices[1:]/low_prices[:-1])

            # Transform volumes (log1p and skip first element)
            log_high_vol = np.log1p(high_volumes[1:])
            log_low_vol = np.log1p(low_volumes[1:])

            # Create feature matrix
            features = np.column_stack((log_high_returns, log_low_returns,
                                      log_high_vol, log_low_vol))
            # Create sequences for this item
            sequences = create_sequences(features)
            self.all_sequences.extend(sequences)
        
        conn.close()
    
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