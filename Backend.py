import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

df = pd.read_csv("Cars24_used_cars.csv")

# 2) Inspect columns 
print(df.head())
print(df.columns)

# 3) Rename columns 
df = df.rename(columns={
    'Car Name': 'car_name',
    'Year': 'year',
    'Distance': 'km_driven',
    'Owner': 'owner',
    'Fuel': 'fuel_type',
    'Drive': 'drive',
    'Type': 'car_type',
    'Price': 'price'
})

# 4) Clean missing values
df.dropna(subset=['year', 'km_driven', 'owner', 'fuel_type', 'price'], inplace=True)

# 5) Feature engineering: split brand & model from car_name
df[['brand', 'model']] = df['car_name'].str.split(' ', n=1, expand=True)

# Select features and target
features = ['brand', 'model', 'year', 'km_driven', 'fuel_type', 'owner', 'drive', 'car_type']
X = df[features]
y = df['price'].astype(float)

# 6) Preprocess: one-hot encode categorical features
cat_features = ['brand', 'model', 'fuel_type', 'owner', 'drive', 'car_type']
num_features = ['year', 'km_driven']

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown='ignore'), cat_features),
        ("numeric", "passthrough", num_features),
    ])

# 7) Build model pipeline
model = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=300, random_state=42))
])

# 8) Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 9) Train the model
model.fit(X_train, y_train)

# 10) Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Mean Absolute Error (MAE): ₹{mae:.2f}")
print(f"Root Mean Squared Error (RMSE): ₹{rmse:.2f}")

# 11) Save model
joblib.dump(model, "cars24_price_model.pkl")
print("Model saved as cars24_price_model.pkl")

import pandas as pd
import joblib

model = joblib.load("cars24_price_model.pkl")

new_car = pd.DataFrame([{
    'brand': 'Hyundai',
    'model': 'Creta',
    'year': 2019,
    'km_driven': 35000,
    'fuel_type': 'Petrol',
    'owner': 'First',
    'drive': 'FWD',
    'car_type': 'SUV'
}])

predicted_price = model.predict(new_car)[0]
print(f"Predicted Used Car Price: ₹{predicted_price:.2f}")