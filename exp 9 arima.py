import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Create a sample time series
date_range = pd.date_range(start='2030-01-01', periods=20, freq='D')
sales = [120, 135, 128, 150, 160, 145, 155, 165, 180, 175,
         190, 200, 210, 205, 220, 230, 240, 250, 245, 260]

data = pd.DataFrame({'date': date_range, 'sales': sales})
data.set_index('date', inplace=True)

# Fit ARIMA model (p, d, q)
model = ARIMA(data['sales'], order=(1, 1, 1))
model_fit = model.fit()

# Forecast next 5 days
forecast = model_fit.forecast(steps=5)

# Display forecast
print("Forecasted Sales:")
print(forecast)

# Plot original and forecasted data
plt.plot(data.index, data['sales'], label='Original', marker='o')
plt.plot(pd.date_range(data.index[-1], periods=6, freq='D')[1:], forecast, label='Forecast', marker='x')
plt.title('ARIMA Forecast of Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()