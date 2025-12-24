import pandas as pd
import numpy as np
from prophet import Prophet
from io import BytesIO
from utils.azure_storage import download_blob_to_bytes


def forecast_demand(blob_name: str, periods: int = 30) -> dict:
    """Forecast future sales using Facebook Prophet time series model from Azure Blob Storage"""
    # Download blob content
    content = download_blob_to_bytes(blob_name)
    
    # Load dataframe based on file extension
    if blob_name.endswith('.csv'):
        df = pd.read_csv(BytesIO(content))
    else:
        df = pd.read_excel(BytesIO(content))

    date_col = next((c for c in df.columns if 'date' in c.lower()), None)
    sales_col = next((c for c in df.columns if 'sales' in c.lower() or 'amount' in c.lower() or 'revenue' in c.lower()),
                     None)

    if not date_col or not sales_col:
        raise ValueError("Could not detect date or sales column")

    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col, sales_col])

    prophet_df = df.groupby(date_col)[sales_col].sum().reset_index()
    prophet_df.columns = ['ds', 'y']

    if len(prophet_df) < 10:
        raise ValueError("Not enough data points for forecasting (need at least 10)")

    model = Prophet(
        daily_seasonality=True if len(prophet_df) > 30 else False,
        weekly_seasonality=True if len(prophet_df) > 14 else False,
        yearly_seasonality=False
    )
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    future_forecast = forecast.tail(periods)

    forecast_data = []
    for _, row in future_forecast.iterrows():
        forecast_data.append({
            "date": row['ds'].strftime('%Y-%m-%d'),
            "predicted_sales": float(row['yhat']),
            "lower_bound": float(row['yhat_lower']),
            "upper_bound": float(row['yhat_upper'])
        })

    avg_current = prophet_df['y'].tail(30).mean()
    avg_forecast = future_forecast['yhat'].mean()
    trend = "increasing" if avg_forecast > avg_current else "decreasing"

    insights = [
        f"Forecast shows {trend} trend over next {periods} days",
        f"Expected average daily sales: {avg_forecast:.2f}",
        f"Current average: {avg_current:.2f}"
    ]

    return {
        "forecast": forecast_data,
        "insights": insights,
        "model_params": {
            "data_points_used": len(prophet_df),
            "forecast_periods": periods
        }
    }
