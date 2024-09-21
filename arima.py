import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import pmdarima  as pm

# Sample period data: Days between periods
# data = {
#         'Start_Date': ['2023-01-18', '2023-02-16', '2023-03-19', '2023-04-17', '2023-05-17', '2023-06-17', '2023-07-19', '2023-08-14', '2023-09-12', '2023-10-12', '2023-11-9', '2023-12-12', '2024-01-06', '2024-02-03', '2024-03-03', '2024-04-04', '2024-04-30', '2024-05-31', '2024-06-28', '2024-07-29', '2024-09-01'],
#         'Days_Gap': [28, 29, 31, 29, 30, 31, 32, 26, 29, 30, 28, 33, 25, 28, 29, 32, 26, 31, 28, 31, 34]}


data = {'Start_Date': ['2022-01-01', '2022-01-29', '2022-02-24', '2022-03-26', '2022-04-22', '2022-05-24', 
                       '2022-06-25', '2022-07-21', '2022-08-19', '2022-09-15', '2022-10-16', '2022-11-15',
                       '2022-12-14', '2023-01-12', '2023-02-08', '2023-03-10', '2023-04-09', '2023-05-06'],
        'Days_Gap': [28, 26, 30, 27, 31, 32, 26, 29, 27, 31, 30, 29, 28, 27, 30, 29, 28, 27]}




# Create a DataFrame
df = pd.DataFrame(data)

# Convert 'Start_Date' to datetime and set it as the index
df['Start_Date'] = pd.to_datetime(df['Start_Date'])
df.set_index('Start_Date', inplace=True)

# Fit the ARIMA model (order = p, d, q)
model = ARIMA(df['Days_Gap'], order=(4, 0, 0))
model_fit = model.fit()

# Summary of the model
# print(model_fit.summary())


# Predict the next cycle gap (using get_forecast for version 0.12+)
forecast = model_fit.get_forecast(steps=1)
predicted_mean = forecast.predicted_mean

# Access the predicted value
print(f"Predicted gap for the next period: {predicted_mean.iloc[0]:.2f} days")


# while using auto arima for p,d,q
# model_auto = pm.auto_arima(df['Days_Gap'], seasonal=False, stepwise=True)
# print(model_auto.order)