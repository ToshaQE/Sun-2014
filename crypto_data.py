import pandas as pd


crypto_data = pd.read_pickle("btc_1d-1.pkl")
df = crypto_data[['close', 'f-138']]
print(list(crypto_data.columns))
