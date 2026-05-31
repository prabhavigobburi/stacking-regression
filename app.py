import streamlit as st, pandas as pd, numpy as np, joblib, plotly.express as px
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Insurance Stacking Regression",layout="wide")
st.title("Insurance Cost Prediction using Stacking Regressor")

df=pd.read_csv("data/insurance.csv")
st.subheader("Dataset")
st.dataframe(df.head())

st.write("Shape:",df.shape)
st.write("Missing Values")
st.dataframe(df.isnull().sum().reset_index())

X=df.drop("charges",axis=1)
y=df["charges"]

for c in X.select_dtypes(include="object").columns:
    X[c]=LabelEncoder().fit_transform(X[c].astype(str))

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=joblib.load("models/stacking_regressor.pkl")
pred=model.predict(X_test)

c1,c2,c3,c4=st.columns(4)
c1.metric("R²",f"{r2_score(y_test,pred):.4f}")
c2.metric("MAE",f"{mean_absolute_error(y_test,pred):.2f}")
c3.metric("MSE",f"{mean_squared_error(y_test,pred):.2f}")
c4.metric("RMSE",f"{mean_squared_error(y_test,pred):.2f}")

st.plotly_chart(px.scatter(x=y_test,y=pred,title="Actual vs Predicted"),use_container_width=True)

rf=joblib.load("models/tuned_random_forest.pkl")
imp=pd.DataFrame({"Feature":X.columns,"Importance":rf.feature_importances_})
st.plotly_chart(px.bar(imp,x="Feature",y="Importance",title="Feature Importance"),use_container_width=True)
