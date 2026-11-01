# 6. Analysing Tesla stock data using pandas and matplotlib.
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("TSLA.csv")
df.columns = df.columns.str.strip()
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df = df.dropna(subset=['Date', 'Close']).sort_values('Date')

print(df.head())
print("Shape:", df.shape)

# Highest and lowest closing prices
high = df.loc[df['Close'].idxmax()]
low = df.loc[df['Close'].idxmin()]

print("\nHighest Price:", high['Close'], "on", high['Date'].date())
print("Lowest Price:", low['Close'], "on", low['Date'].date())

# Daily percentage change
df['Daily_Change'] = df['Close'].pct_change() * 100
print("\nTop 10 Daily Gains:")
print(df.nlargest(10, 'Daily_Change')[['Date', 'Close', 'Daily_Change']])

# Closing price graph
plt.plot(df['Date'], df['Close'], label='Closing Price')
plt.title("Tesla Stock Price Trend")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Daily change graph
plt.plot(df['Date'], df['Daily_Change'], label='Daily Change')
plt.axhline(0, linestyle='--')
plt.title("Daily Percentage Change")
plt.xlabel("Date")
plt.ylabel("Change (%)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Final summary
print("\nAverage Closing Price:", round(df['Close'].mean(), 2))
print("Average Daily Change:", round(df['Daily_Change'].mean(), 4), "%")
print("\nAnalysis Completed Successfully!")
