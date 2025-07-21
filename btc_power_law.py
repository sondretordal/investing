import requests
import json
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import matplotlib.ticker as ticker

# Fetch historical Bitcoin price data from Blockchain.com API
url = "https://api.blockchain.info/charts/market-price?timespan=all&format=json"
response = requests.get(url)
data = json.loads(response.text)
values = data['values']

# Parse dates and prices, skipping zeros
dates = []
prices = []
for point in values:
    timestamp = point['x']
    price = point['y']
    if price > 0:
        date = datetime.fromtimestamp(timestamp)
        dates.append(date)
        prices.append(price)

# Convert prices to NumPy array for proper slicing
prices = np.array(prices)

# Use Bitcoin genesis block as origin (January 3, 2009)
genesis_date = datetime(2009, 1, 3)
# Optional: Add a negative offset (e.g., -300) to shift origin earlier for better fit
offset_days = 0  # Adjust this if needed
days = np.array([(d - genesis_date).days + 1 + offset_days for d in dates])

# Prepare log values for fitting
log_days = np.log(days)
log_prices = np.log(prices)

plt.figure()
plt.plot(log_days, log_prices)
plt.show()

# Fit a linear regression to log-log data (slope = power law exponent)
slope, intercept, r_value, p_value, std_err = linregress(log_days, log_prices)
print(f"Power law exponent (slope): {slope:.2f}")
print(f"R-squared (fit quality): {r_value**2:.2f}")

# Manual adjustments for channel: offsets in log space (adjust these to create upper/lower bounds)
# Positive offset raises the line (upper channel), negative lowers it (lower channel)
upper_offset = 2.0  # Adjust this value manually (e.g., 0.5 to 2.0)
lower_offset = -1.0  # Adjust this value manually (e.g., -0.5 to -2.0)

# Generate fitted prices for the main line
fit_log_prices = intercept + slope * log_days
fit_prices = np.exp(fit_log_prices)

# Upper channel
fit_log_prices_upper = intercept + upper_offset + slope * log_days
fit_prices_upper = np.exp(fit_log_prices_upper)

# Lower channel
fit_log_prices_lower = intercept + lower_offset + slope * log_days
fit_prices_lower = np.exp(fit_log_prices_lower)

# Extend the plot to 2049: Calculate future days
end_date = datetime(2046, 12, 31)
max_days = (end_date - genesis_date).days + 1 + offset_days
future_days = np.linspace(1, max_days, 1000)  # Smooth line for extension
log_future_days = np.log(future_days)

# Extend fits to future
ext_fit_log_prices = intercept + slope * log_future_days
ext_fit_prices = np.exp(ext_fit_log_prices)

ext_fit_log_prices_upper = intercept + upper_offset + slope * log_future_days
ext_fit_prices_upper = np.exp(ext_fit_log_prices_upper)

ext_fit_log_prices_lower = intercept + lower_offset + slope * log_future_days
ext_fit_prices_lower = np.exp(ext_fit_log_prices_lower)

# Calculate the day for July 1, 2010 to set as minimum x
# To adjust the x-axis start, change the date in start_date below (e.g., datetime(2010, 7, 1) for July 1, 2010)
# This controls where the plot begins visually; the fit lines will extend from there onward.
start_date = datetime(2010, 7, 1)  # Adjust this date to change the x-axis start
min_day = (start_date - genesis_date).days + 1 + offset_days
if min_day < 1:
    min_day = 1  # Prevent going below 1 for log scale

# Generate custom x-ticks for Jan 1 each year from 2010 to 2046
tick_days = []
tick_labels = []
current_date = datetime(2010, 1, 1)
while current_date.year <= 2046:
    tick_date = datetime(current_date.year, 1, 1)
    tick_day = (tick_date - genesis_date).days + 1 + offset_days
    if tick_day >= min_day:  # Only include ticks after the start date
        tick_days.append(tick_day)
        tick_labels.append(tick_date.strftime('%Y-%m'))
    current_date += timedelta(days=365)  # Advance to next year

# Plot on log-log scale
plt.figure(figsize=(12, 8))
plt.loglog(days[days >= min_day], prices[days >= min_day], 'bo', label='BTC Price (actual)', markersize=4, alpha=0.5)

# Plot extended fits (clipped to start from min_day)
future_days_clipped = future_days[future_days >= min_day]
plt.loglog(future_days_clipped, ext_fit_prices[future_days >= min_day], 'r-', label=f'Power Law Fit (n={slope:.2f})', linewidth=2)
plt.loglog(future_days_clipped, ext_fit_prices_upper[future_days >= min_day], 'g--', label=f'Upper Channel (offset={upper_offset})', linewidth=2)
plt.loglog(future_days_clipped, ext_fit_prices_lower[future_days >= min_day], 'm--', label=f'Lower Channel (offset={lower_offset})', linewidth=2)

plt.xlabel('Days Since Genesis Block (log scale)')
plt.ylabel('BTC Price in USD (log scale)')
plt.title('Bitcoin Price vs. Time (Power Law Log-Log Plot with Channel)')
plt.legend()
plt.grid(True, which="both", ls="--")

# Set custom ticks and labels
plt.xticks(tick_days, tick_labels, rotation=45, ha='right')
plt.xlim(min_day, max_days * 1.1)  # Start from the adjusted date, with slight padding on right

# Make y-axis more detailed with ticks at multiples of each power of 10 (e.g., 10000, 20000, ..., 100000, etc.)
ax = plt.gca()
ymin, ymax = ax.get_ylim()
min_exp = int(np.floor(np.log10(ymin)))
max_exp = int(np.ceil(np.log10(ymax)))
yticks = []
for exp in range(min_exp, max_exp + 1):
    for i in range(1, 10):
        tick = i * 10 ** exp
        if ymin <= tick <= ymax:
            yticks.append(tick) 
ax.set_yticks(yticks)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:,.0f}'))

plt.tight_layout()
plt.show()