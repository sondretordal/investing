import numpy as np
from matplotlib import pyplot as plt
import yfinance as yf


btc = yf.Ticker("BTC-USD")


df = btc.history(period="max", interval="1d")

# Convet tot [1...no_days]
df["t_days"] = (df.index - df.index[0]).days.astype(float) + 1

t = df["t_days"]
y = df["Close"]

log_t = np.log(t)
log_y = np.log(y)


plt.figure()
plt.plot(log_t, log_y)
plt.show()
