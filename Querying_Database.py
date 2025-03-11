import sqlite3
import pandas as pd
conn = sqlite3.connect('osrsmarketdata.sqlite')
cur = conn.cursor()

typeids = (1234, 5678, 9012)
cur.execute("SELECT timestamp, typeid, avgHighPrice, highPriceVolume, avgLowPrice, lowPriceVolume "
            "FROM marketdata "
            "WHERE interval = ? AND typeid IN ({})".format(','.join(['?'] * len(typeids))),
            (300,) + tuple(typeids))


data = cur.fetchall()
df = pd.DataFrame(data, columns=['timestamp', 'typeid', 'avgHighPrice', 'highPriceVolume', 'avgLowPrice', 'lowPriceVolume'])
'''
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
df.set_index('timestamp', inplace=True)'''
print(df)