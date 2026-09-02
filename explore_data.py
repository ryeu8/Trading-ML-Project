import yfinance as yf
data = yf.download("AAPL", period="1y")
print(data.head())
print(data.shape)
print(data.columns)
data.columns = data.columns.droplevel(1)
data["Return"] = data["Close"].pct_change()
print(data.head())

data = data.dropna()
print(data.shape)

import matplotlib.pyplot as plt
plt.plot(data.index, data["Close"])
plt.title("AAPL Closing Price")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.savefig("aapl_price.png")