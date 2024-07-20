import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load and preprocess data
data = pd.read_csv("housing.csv")
data.dropna(inplace=True)

# Prepare features and target
X = data.drop(['median_house_value'], axis=1)
y = data['median_house_value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

train_data = X_train.join(y_train)
train_data['total_rooms'] = np.log(train_data['total_rooms'] + 1)
train_data['total_bedrooms'] = np.log(train_data['total_bedrooms'] + 1)
train_data['population'] = np.log(train_data['population'] + 1)
train_data['households'] = np.log(train_data['households'] + 1)
train_data = train_data.join(pd.get_dummies(train_data.ocean_proximity)).drop(['ocean_proximity'], axis=1)
train_data['bedroom_ratio'] = train_data['total_bedrooms'] / train_data['total_rooms']
train_data['household_rooms'] = train_data['total_rooms'] / train_data['households']

scaler = StandardScaler()
X_train_s = scaler.fit_transform(train_data.drop(['median_house_value'], axis=1))
y_train = train_data['median_house_value']

reg = LinearRegression()
reg.fit(X_train_s, y_train)

test_data = X_test.join(y_test)
test_data['total_rooms'] = np.log(test_data['total_rooms'] + 1)
test_data['total_bedrooms'] = np.log(test_data['total_bedrooms'] + 1)
test_data['population'] = np.log(test_data['population'] + 1)
test_data['households'] = np.log(test_data['households'] + 1)
test_data = test_data.join(pd.get_dummies(test_data.ocean_proximity)).drop(['ocean_proximity'], axis=1)
test_data['bedroom_ratio'] = test_data['total_bedrooms'] / test_data['total_rooms']
test_data['household_rooms'] = test_data['total_rooms'] / test_data['households']

missing_cols = set(train_data.columns) - set(test_data.columns)
for col in missing_cols:
    test_data[col] = 0

test_data = test_data[train_data.columns]
X_test_s = scaler.transform(test_data.drop(['median_house_value'], axis=1))
y_test = test_data['median_house_value']

# Streamlit app
st.title("Housing Data Analysis and Prediction")

# Display data
if st.checkbox("Show raw data"):
    st.write(data)

# Show descriptive statistics
if st.checkbox("Show data statistics"):
    st.write(data.describe())

# Correlation heatmap
if st.checkbox("Show correlation heatmap"):
    temp_train_data = train_data.drop(['median_house_value'], axis=1)
    plt.figure(figsize=(15, 8))
    sns.heatmap(temp_train_data.corr(), annot=True, cmap="RdPu")
    st.pyplot(plt)

# Histogram
if st.checkbox("Show histograms"):
    plt.figure(figsize=(15, 8))
    train_data.hist(figsize=(15, 8), color="purple", edgecolor="black")
    st.pyplot(plt)

# Model performance
st.subheader("Model Performance")
score = reg.score(X_test_s, y_test)
st.write(f"R-squared score: {score:.4f}")

# Scatter plot
if st.checkbox("Show scatter plot of latitude and longitude"):
    plt.figure(figsize=(15, 8))
    sns.scatterplot(x='latitude', y='longitude', data=train_data, hue='median_house_value', palette='RdPu')
    st.pyplot(plt)

# User input for prediction
st.subheader("Predict House Price")

# Define input fields
def user_input_features():
    latitude = st.number_input("Latitude", -90.0, 90.0, 37.0)
    longitude = st.number_input("Longitude", -180.0, 180.0, -122.0)
    housing_median_age = st.slider("Median Age", 1, 100, 20)
    total_rooms = st.number_input("Total Rooms", 0, 10000, 500)
    total_bedrooms = st.number_input("Total Bedrooms", 0, 10000, 100)
    population = st.number_input("Population", 0, 100000, 1000)
    households = st.number_input("Households", 0, 10000, 400)
    ocean_proximity = st.selectbox("Ocean Proximity", options=["NEAR BAY", "NEAR OCEAN", "ISLAND", "INLAND", "<1H OCEAN"])

    data = {
        'latitude': latitude,
        'longitude': longitude,
        'housing_median_age': housing_median_age,
        'total_rooms': total_rooms,
        'total_bedrooms': total_bedrooms,
        'population': population,
        'households': households,
        'ocean_proximity': ocean_proximity
    }
    return pd.DataFrame(data, index=[0])

user_input = user_input_features()

# Transform user input
user_input_log = user_input.copy()
user_input_log['total_rooms'] = np.log(user_input_log['total_rooms'] + 1)
user_input_log['total_bedrooms'] = np.log(user_input_log['total_bedrooms'] + 1)
user_input_log['population'] = np.log(user_input_log['population'] + 1)
user_input_log['households'] = np.log(user_input_log['households'] + 1)

user_input_encoded = pd.get_dummies(user_input_log, columns=['ocean_proximity'])
missing_cols = set(train_data.columns) - set(user_input_encoded.columns)
for col in missing_cols:
    user_input_encoded[col] = 0
user_input_encoded = user_input_encoded[train_data.columns]

# Ensure all columns are in the same order as during training
user_input_encoded = user_input_encoded[train_data.drop(['median_house_value'], axis=1).columns]

# Scale user input
user_input_scaled = scaler.transform(user_input_encoded)

# Predict and display result
prediction = reg.predict(user_input_scaled)
st.write(f"Predicted House Price: ${prediction[0] * 1000:.2f}")
