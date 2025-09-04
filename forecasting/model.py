import pandas as pd
from prophet import Prophet #forecating model by meta 

def forecast_sales(df, store, product, periods=7):
    # Filter data for the given store and product
    data = df[(df['store'] == store) & (df['product'] == product)].copy()
    data = data[['date', 'sales']]
    data = data.rename(columns={'date': 'ds', 'sales': 'y'})
    # Fit Prophet model
    model = Prophet()
    model.fit(data)
    # Make future dataframe
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']] 