import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample data
data = {
    'cycleNumber': [1, 2, 3, 4, 5, 6, 7, 8],
    'daysGap': [28, 29, 27, 29, 30, 27, 28, 29]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Prepare data
X = df[['cycleNumber']]  # Feature
y = df['daysGap']   # Target

# Initialize and train the model
model = LinearRegression()
model.fit(X, y)

manual_size = np.array([[9]])  # Input size must be 2D, hence the double brackets

# Make the prediction
predicted_price = model.predict(manual_size)

# Print the result
print(f"Predicted days: {predicted_price[0]}")

# # Make predictions
# y_pred = model.predict(X_test)

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# print(f"Mean Squared Error: {mse}")
# print(f"Predicted prices: {y_pred}")
# print(f"Actual prices: {y_test.values}") 

