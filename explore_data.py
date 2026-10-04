import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = yf.download("AAPL", period="1y")
data.columns = data.columns.droplevel(1)
data["Return"] = data["Close"].pct_change()
data = data.dropna()

plt.plot(data.index, data["Close"])
plt.title("AAPL Closing Price")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.savefig("aapl_price.png")

data["MA20"] = data["Close"].rolling(20).mean()
data["Volatility"] = data["Return"].rolling(20).std()
data["Momentum"] = data["Close"].pct_change(10)

data["Target"] = (data["Return"].shift(-1) > 0).astype(int)
data = data.dropna()

X = data[["MA20", "Volatility", "Momentum"]]
y = data["Target"]

split_index = int(len(data) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print(X_train.shape, X_test.shape)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy after scaling:", accuracy)
print(predictions)

baseline = max(y_test.mean(), 1 - y_test.mean())
print("Baseline (always guess majority class):", baseline)