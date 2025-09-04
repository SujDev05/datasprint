import pandas as pd

def recommend_inventory(forecast_df, safety_stock_ratio=0.2):
    forecast_df = forecast_df.copy()
    forecast_df['recommended_inventory'] = (forecast_df['yhat'] * (1 + safety_stock_ratio)).round().astype(int)
    return forecast_df[['ds', 'recommended_inventory']] 