import pandas as pd


def analyze_sales_data(filepath: str) -> dict:
    """Analyze sales data and return comprehensive statistics for visualization"""
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)

    # Auto-detect columns
    date_col = next((c for c in df.columns if 'date' in c.lower()), None)
    product_col = next((c for c in df.columns if 'product' in c.lower() or 'item' in c.lower()), None)
    sales_col = next((c for c in df.columns if 'sales' in c.lower() or 'amount' in c.lower() or 'revenue' in c.lower()),
                     None)
    quantity_col = next((c for c in df.columns if 'quantity' in c.lower() or 'qty' in c.lower()), None)
    category_col = next((c for c in df.columns if 'category' in c.lower()), None)
    region_col = next((c for c in df.columns if 'region' in c.lower() or 'location' in c.lower()), None)

    analytics = {
        "total_rows": len(df),
        "columns": list(df.columns),
        "detected_columns": {
            "date": date_col,
            "product": product_col,
            "sales": sales_col,
            "quantity": quantity_col,
            "category": category_col,
            "region": region_col
        }
    }

    # Sales statistics
    if sales_col:
        analytics["sales_statistics"] = {
            "total_sales": float(df[sales_col].sum()),
            "average_sales": float(df[sales_col].mean()),
            "max_sales": float(df[sales_col].max()),
            "min_sales": float(df[sales_col].min()),
            "median_sales": float(df[sales_col].median()),
            "std_dev": float(df[sales_col].std())
        }

    # Quantity statistics
    if quantity_col:
        analytics["quantity_statistics"] = {
            "total_quantity": float(df[quantity_col].sum()),
            "average_quantity": float(df[quantity_col].mean())
        }

    # Top products
    if product_col and sales_col:
        top_products = df.groupby(product_col)[sales_col].sum().sort_values(ascending=False).head(10)
        analytics["top_products"] = [
            {"name": str(k), "sales": float(v)}
            for k, v in top_products.items()
        ]

        bottom_products = df.groupby(product_col)[sales_col].sum().sort_values(ascending=True).head(5)
        analytics["bottom_products"] = [
            {"name": str(k), "sales": float(v)}
            for k, v in bottom_products.items()
        ]

    # Category breakdown
    if category_col and sales_col:
        category_sales = df.groupby(category_col)[sales_col].sum().sort_values(ascending=False)
        analytics["sales_by_category"] = [
            {"category": str(k), "sales": float(v)}
            for k, v in category_sales.items()
        ]

    # Region breakdown
    if region_col and sales_col:
        region_sales = df.groupby(region_col)[sales_col].sum().sort_values(ascending=False)
        analytics["sales_by_region"] = [
            {"region": str(k), "sales": float(v)}
            for k, v in region_sales.items()
        ]

    # Time series data
    if date_col and sales_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df_clean = df.dropna(subset=[date_col])

        if len(df_clean) > 0:
            # Daily sales
            daily_sales = df_clean.groupby(df_clean[date_col].dt.date)[sales_col].sum()
            analytics["daily_sales"] = [
                {"date": str(k), "sales": float(v)}
                for k, v in daily_sales.items()
            ]

            # Monthly sales
            monthly_sales = df_clean.groupby(df_clean[date_col].dt.to_period('M'))[sales_col].sum()
            analytics["monthly_sales"] = [
                {"month": str(k), "sales": float(v)}
                for k, v in monthly_sales.items()
            ]

            # Weekly sales
            weekly_sales = df_clean.groupby(df_clean[date_col].dt.to_period('W'))[sales_col].sum()
            analytics["weekly_sales"] = [
                {"week": str(k), "sales": float(v)}
                for k, v in weekly_sales.items()
            ]

            analytics["time_range"] = {
                "start": str(df_clean[date_col].min().date()),
                "end": str(df_clean[date_col].max().date()),
                "total_days": len(daily_sales)
            }

    # Product-Category matrix (if both exist)
    if product_col and category_col and sales_col:
        product_category = df.groupby([product_col, category_col])[sales_col].sum().reset_index()
        analytics["product_category_breakdown"] = [
            {
                "product": str(row[product_col]),
                "category": str(row[category_col]),
                "sales": float(row[sales_col])
            }
            for _, row in product_category.iterrows()
        ]

    return analytics