import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from sklearn.metrics import mean_squared_error
import random
import torch
import numpy as np
import sqlite3
from collections import defaultdict
from torch.utils.data import random_split, Dataset
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Load your time series data into a pandas DataFrame
# Assume 'data.csv' contains two columns: 'date' and 'price'

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
            for i in range(n - 488):
                # Check if the first hour of the sequence has highPriceVolume or lowPriceVolume as 0
                first_hour = item_data_list[i]
                if first_hour[2] == 0 or first_hour[4] == 0:
                    continue  # Skip this sequence

                sequence_data = []
                for j in range(i, i+489):
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
                price_features = features[:480, :2]
                price_mean = price_features.mean()    # average of all values in both columns
                price_std = price_features.std()
                vol_features = features[:480, 2:]            
                vol_mean = vol_features.mean()
                vol_std = vol_features.std()
                if price_std == 0:
                    price_std = 1000
                if vol_std == 0:
                    vol_std = 1000                
                sequence = np.column_stack((
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
            if total_hours < 489:
                continue  # Not enough data

            seq = np.array(filled_data_list, dtype=np.float32)

            # Skip sequences with volume issues
            if (np.mean(seq[:, 2] == 0) > 0.005 or 
                np.mean(seq[:, 4] == 0) > 0.005):
                continue

            recent_low_price = seq[-1, 3]  # Most recent low price
            average_volume = np.mean(seq[:, 2] + seq[:, 4])  # Avg(hvol + lvol)
            if average_volume <= 24:
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
                if (buylimit * recent_low_price) <= 100_000:
                    continue

            # Create sequences for this item
            sequences = create_sequences(filled_data_list)
            self.all_sequences.extend(sequences)
        
        conn.close()
    
    def __len__(self):
        return len(self.all_sequences)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.all_sequences[idx]).float()
print("Start Loading")
dataset = OSRSMarketDataset()
train_size = int(0.95 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print("Finished Loading")
mse_list = []
count = 0

# Iterate through each sample in the validation dataset.
for sample in val_dataset:
    count += 1
    print("Sample", count)
    # Convert the torch tensor to a NumPy array.
    data_np = sample.numpy()
    
    # Split into training (first 480 points) and test data (last 8 points).
    train_data = data_np[:480]
    test_data = data_np[480:]
    
    # Convert training data to a Pandas Series.
    ts = pd.Series(train_data)
    
    # Automatically determine ARIMA parameters using auto_arima.
    stepwise_model = pm.auto_arima(ts,
                                   start_p=1, start_q=1,
                                   max_p=3, max_q=3,
                                   seasonal=False,
                                   trace=False,
                                   error_action='ignore',
                                   suppress_warnings=True)
    order = stepwise_model.order
    
    # Fit the ARIMA model on the training data.
    model = ARIMA(ts, order=order)
    model_fit = model.fit()
    
    # Forecast the next 8 data points.
    forecast = model_fit.forecast(steps=8)
    
    # Calculate the Mean Squared Error between the forecast and the actual test data.
    mse = mean_squared_error(test_data, forecast)
    mse_list.append(mse)

# Calculate and print the average MSE over the entire dataset.
avg_mse = np.mean(mse_list)
print("Average MSE over dataset:", avg_mse)
