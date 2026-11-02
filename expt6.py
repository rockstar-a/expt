# 6. Tesla Share
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('TSLA.csv')
df['Date'] = pd.to_datetime(df['Date'])

# 1. Highest and Lowest closing prices & their dates
max_row = df.loc[df['Close'].idxmax()]
min_row = df.loc[df['Close'].idxmin()]
print(f"Highest Close: {max_row['Close']} on {max_row['Date'].date()}")
print(f"Lowest Close : {min_row['Close']} on {min_row['Date'].date()}")

# 2. Daily percentage change in closing price
df['Pct_Change'] = df['Close'].pct_change() * 100

# 3. Top 10 days with highest daily gain
print("\nTop 10 Highest Daily Gain Days:")
print(df[['Date', 'Close', 'Pct_Change']].nlargest(10, 'Pct_Change'))

# 4. Plot Closing-Price Trend
plt.plot(df['Date'], df['Close'], label='Close Price')
plt.xlabel('Date')
plt.ylabel('Closing Price ($)')
plt.title('Tesla (TSLA) Closing Price Trend')
plt.legend()
plt.show()
