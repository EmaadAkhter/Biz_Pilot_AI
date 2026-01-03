import pandas as pd
import logging
from typing import Optional
from prophet import Prophet
from utils.redis_cache import get_cached_forecast, cache_forecast

logger = logging.getLogger(__name__)


def forecast_demand(df: pd.DataFrame, periods: int = 30, user_id: str = None,
                    blob_name: str = None, use_cache: bool = True) -> dict:
    """Forecast future sales using Facebook Prophet time series model

    Args:
        df: DataFrame to forecast
        periods: Number of days to forecast (1-365)
        user_id: User ID for caching
        blob_name: File identifier for caching
        use_cache: Whether to use Redis cache

    Returns:
        Dictionary with forecast data and insights
    """

    # Try to get from cache if enabled
    if use_cache and user_id and blob_name:
        cached = get_cached_forecast(user_id, blob_name, periods)
        if cached:
            logger.info(f"✓ Forecast cache hit for {blob_name} (periods={periods})")
            return cached

    logger.info(f"Computing forecast for {blob_name or 'dataframe'} (periods={periods})")

    # Auto-detect columns
    date_col = next((c for c in df.columns if 'date' in c.lower()), None)
    sales_col = next((c for c in df.columns if 'sales' in c.lower() or 'amount' in c.lower() or 'revenue' in c.lower()),
                     None)

    if not date_col or not sales_col:
        raise ValueError("Could not detect date or sales column")

    # Prepare data
    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
    df_copy = df_copy.dropna(subset=[date_col, sales_col])

    if len(df_copy) < 10:
        raise ValueError("Not enough data points for forecasting (need at least 10)")

    # Aggregate by date
    prophet_df = df_copy.groupby(date_col)[sales_col].sum().reset_index()
    prophet_df.columns = ['ds', 'y']

    # Train model
    model = Prophet(
        daily_seasonality=True if len(prophet_df) > 30 else False,
        weekly_seasonality=True if len(prophet_df) > 14 else False,
        yearly_seasonality=False,
        interval_width=0.95
    )
    model.fit(prophet_df)

    # Generate forecast
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    future_forecast = forecast.tail(periods)

    # Format forecast data
    forecast_data = []
    for _, row in future_forecast.iterrows():
        forecast_data.append({
            "date": row['ds'].strftime('%Y-%m-%d'),
            "predicted_sales": float(row['yhat']),
            "lower_bound": float(row['yhat_lower']),
            "upper_bound": float(row['yhat_upper'])
        })

    # Calculate trend
    avg_current = prophet_df['y'].tail(30).mean()
    avg_forecast = future_forecast['yhat'].mean()
    trend_direction = "increasing" if avg_forecast > avg_current else "decreasing"
    trend_percent = ((avg_forecast - avg_current) / avg_current * 100) if avg_current != 0 else 0

    insights = [
        f"Forecast shows {trend_direction} trend over next {periods} days ({trend_percent:.1f}%)",
        f"Expected average daily sales: ${avg_forecast:.2f}",
        f"Current average: ${avg_current:.2f}"
    ]

    result = {
        "forecast": forecast_data,
        "insights": insights,
        "summary": {
            "trend": trend_direction,
            "trend_percent": round(trend_percent, 2),
            "current_avg": round(avg_current, 2),
            "forecast_avg": round(avg_forecast, 2),
            "data_points_used": len(prophet_df),
            "forecast_periods": periods
        }
    }

    # Cache the results if enabled
    if use_cache and user_id and blob_name:
        if cache_forecast(user_id, blob_name, periods, result):
            logger.info(f"✓ Forecast cached for {blob_name} (periods={periods})")
        else:
            logger.warning(f"Failed to cache forecast for {blob_name}")

    return result