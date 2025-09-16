import yfinance as yf
import matplotlib.pyplot as plt

# Download stock data (TCS in this example)
data = yf.download("TCS.NS", start="2023-01-01", end="2023-12-31")

# Show first 5 rows
print(data.head())

# Plot closing prices
plt.figure(figsize=(10, 5))
plt.plot(data['Close'], label="TCS Closing Price")
plt.title("TCS Stock Closing Prices (2023)")
plt.xlabel("Date")
plt.ylabel("Price (INR)")
plt.legend()
plt.show()
