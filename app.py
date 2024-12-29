import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso,ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# load the dataset
cal= fetch_california_housing()
df= pd.DataFrame(data=cal.data, columns=cal.feature_names)
df['price']=cal.target
df.head()

# title of the app
st.title("California House Price Prediction for XYZ Brokerage Company") 
st.subheader("Data Overview")
st.dataframe(df.head(10))

# split the data into input and target variables
X=df.drop('price', axis=1) # input variables
y=df['price'] # target variable

# split the input data and target data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


selected_model = st.selectbox("Select a model", ["Linear Regression", "Ridge", "Lasso", "ElasticNet"])

# intiate the model
models={
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "ElasticNet": ElasticNet()}

# train the selected model
final_model = models[selected_model] # from dictionary whatever model user has selected 
# intialize that model only


# train the model
final_model.fit(X_train, y_train)

# make predictions
y_pred = final_model.predict(X_test)

# EVALUATE THE MODEL
# calculate the metrics
r2=r2_score(y_test, y_pred)
mae=mean_absolute_error(y_test, y_pred)
mse=mean_squared_error(y_test, y_pred)
rmse=np.sqrt(mse)


# display the metrics for the model
st.subheader("Model Evaluation")
st.write("R2 Score:", r2)
st.write("Mean Absolute Error:", mae)
st.write("Mean Squared Error:", mse)
st.write("Root Mean Squared Error:", rmse)

# prompt the user to enter the input values
st.write("Enter the input values and predict the price of the house:")

user_input={}

for col in X.columns:
    user_input[col]=st.number_input(col)

# convert the user input into a dataframe
user_input_df= pd.DataFrame(user_input, index=[0])

# standardize the user input
user_input_df_sc= scaler.transform(user_input_df)

# predict the price of the house
price = final_model.predict(user_input_df_sc)

st.write(f"Predicted Pricefor this particular house is ${np.round(price[0]*100000), 2} (USD)")