# 9. Stock Market
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('TSLA.csv')
df['Date'] = pd.to_datetime(df['Date'])

# 1. Summary Statistics & Sentiment Breakdown
print(df[['Open', 'High', 'Low', 'Close', 'Volume']].describe())

# 2. Highest and Lowest Closing Prices (Exact same code as Exp 6)
max_row = df.loc[df['Close'].idxmax()]
min_row = df.loc[df['Close'].idxmin()]
print(f"\nHighest Close: {max_row['Close']} on {max_row['Date'].date()}")
print(f"Lowest Close : {min_row['Close']} on {min_row['Date'].date()}")

# 3. Plot Closing Price Trend
plt.plot(df['Date'], df['Close'], label='Close Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Stock Closing Price Trend')
plt.legend()
plt.show()
