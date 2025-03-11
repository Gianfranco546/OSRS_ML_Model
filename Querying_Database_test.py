import sqlite3
import pandas as pd
conn = sqlite3.connect('osrsmarketdata.sqlite')
cur = conn.cursor()

typeids = (1, 4151)
cur.execute("SELECT timestamp, typeid, avgHighPrice, highPriceVolume, avgLowPrice, lowPriceVolume "
            "FROM marketdata "
            "WHERE interval = ? AND typeid IN ({})".format(','.join(['?'] * len(typeids))),
            (86400,) + tuple(typeids))


data = cur.fetchall()
df = pd.DataFrame(data, columns=['timestamp', 'typeid', 'avgHighPrice', 'highPriceVolume', 'avgLowPrice', 'lowPriceVolume'])

print(df)