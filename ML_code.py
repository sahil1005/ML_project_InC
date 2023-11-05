import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the data from the Excel file
df = pd.read_excel('train.xlsx')

# Convert 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'])  # Assuming 'date' is the column name for dates

# Set 'date' column as the index
df.set_index('date', inplace=True)

# Aggregate data at year-month level and sum the sales
df_monthly = df.resample('M').sum()

# Shift the target variable (Sales) to create lag features
for i in range(1, 13):  # Assuming you want to use 12 months of lag features
    df_monthly[f'Sales_Lag_{i}'] = df_monthly['sales'].shift(i)  # Assuming 'sales' is the column name for sales data

# Drop rows with NaN values due to lag features
df_monthly.dropna(inplace=True)

# Define features (X) and target variable (y)
X = df_monthly.drop(columns=['sales'])  # Assuming 'sales' is the column name for sales data
y = df_monthly['sales']  # Assuming 'sales' is the column name for sales data

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the XGBoost Regressor
xg_reg = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                          max_depth=5, alpha=10, n_estimators=100)

# Train the model
xg_reg.fit(X_train, y_train)

# Create lag features for prediction
future_data = X.iloc[-1:]  # Use the last available data point as input for future predictions
for i in range(1, 13):  # Assuming you want to use 12 months of lag features
    future_data[f'Sales_Lag_{i}'] = df_monthly['sales'][-i]  # Assuming 'sales' is the column name for sales data

# Predict future sales
future_sales = xg_reg.predict(future_data)

# Create a DataFrame with predicted future sales and format it as desired
future_sales_df = pd.DataFrame({'yearmonth': future_data.index.strftime('%Y-%m'), 'Sales': future_sales})

# Save the formatted predicted future sales to an Excel file
future_sales_df.to_excel('predicted_sales.xlsx', index=False)

print("Formatted Predicted Future Sales saved to predicted_sales.xlsx file.")
