import pandas as pd
import numpy as np

def detect_anomalies(df, store, product, threshold=2.0):
    # Filter data for the given store and product
    data = df[(df['store'] == store) & (df['product'] == product)].copy()
    sales = data['sales']
    mean = sales.mean()
    std = sales.std()
    data['z_score'] = (sales - mean) / std
    data['anomaly'] = data['z_score'].abs() > threshold
    return data[['date', 'store', 'product', 'sales', 'z_score', 'anomaly']] 