import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


st.title("Wine Quality Prediction using ElasticNet")
st.markdown("Train and evaluate a regression model on the Wine Quality dataset.")

alpha = st.slider("Select Alpha (Regularization Strength)", 0.0, 1.0, 0.5)
l1_ratio = st.slider("Select L1 Ratio (ElasticNet Mixing)", 0.0, 1.0, 0.5)

csv_url = "https://raw.githubusercontent.com/plotly/datasets/master/winequality-red.csv"

@st.cache_data          
def load_data():
    data = pd.read_csv(csv_url)
    return data

data = load_data()
st.write("### Sample Data", data.head())


train, test = train_test_split(data)
train_x = train.drop("quality", axis=1)
test_x = test.drop("quality", axis=1)
train_y = train[["quality"]]
test_y = test[["quality"]]

with mlflow.start_run():
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    model.fit(train_x, train_y)

    predictions = model.predict(test_x)

    rmse, mae, r2 = eval_metrics(test_y, predictions)


    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    if tracking_url_type_store != "file":
        mlflow.sklearn.log_model(model, "model", registered_model_name="ElasticNetWineModel")
    else:
        mlflow.sklearn.log_model(model, "model")


st.subheader("Model Evaluation Metrics:")
st.write(f"**RMSE**: {rmse:.2f}")
st.write(f"**MAE**: {mae:.2f}")
st.write(f"**R²**: {r2:.2f}")
