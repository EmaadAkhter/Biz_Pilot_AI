from fastmcp import FastMCP
from dotenv import load_dotenv
import pandas as pd

from utils.analytics import analyze_sales_data
from utils.forecast import forecast_demand
from utils.file_handler import get_user_files, load_dataframe, get_file_path

load_dotenv()

mcp = FastMCP("bizpilot-analytics")


@mcp.tool()
def list_available_files(user_id: str) -> dict:
    """List all available sales data files for a user.

    Args:
        user_id: The user's ID

    Returns:
        Dictionary with list of available files
    """
    try:
        files = get_user_files(user_id)
        return {
            "status": "success",
            "total_files": len(files),
            "files": files
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def analyze_sales_file(filename: str, user_id: str) -> dict:
    """Analyze sales data from an uploaded file.

    Returns:
    - Total sales, averages, max/min values
    - Top and bottom products
    - Daily/weekly/monthly trends
    - Sales by category and region

    Args:
        filename: Name or blob_name of the file
        user_id: The user's ID

    Returns:
        Dictionary with comprehensive sales analytics
    """
    try:
        blob_name = get_file_path(filename, user_id)
        df = load_dataframe(blob_name, user_id)
        analytics = analyze_sales_data(df)
        return analytics
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def query_sales_data(filename: str, user_id: str, question: str) -> dict:
    """Answer specific questions about sales data.

    Args:
        filename: Name or blob_name of the file
        user_id: The user's ID
        question: The user's question about the data

    Returns:
        Data summary with instructions for analysis
    """
    try:
        blob_name = get_file_path(filename, user_id)
        df = load_dataframe(blob_name, user_id)
        analytics = analyze_sales_data(df)

        return {
            "question": question,
            "data_summary": analytics,
            "instruction": "Analyze the data_summary and provide a clear answer to the user's question with specific numbers and insights."
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def forecast_sales_demand(filename: str, user_id: str, periods: int = 30) -> dict:
    """Forecast future sales demand using Facebook Prophet.

    Returns predictions with confidence intervals and trend analysis.

    Args:
        filename: Name or blob_name of the file
        user_id: The user's ID
        periods: Number of days to forecast (1-365, default: 30)

    Returns:
        Forecast data with trend analysis
    """
    try:
        if periods < 1 or periods > 365:
            return {"status": "error", "message": "Periods must be between 1 and 365"}

        blob_name = get_file_path(filename, user_id)
        df = load_dataframe(blob_name, user_id)
        result = forecast_demand(df, periods)
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def conduct_market_research(
        idea: str,
        target_customer: str,
        geography: str,
        depth_level: int = 2
) -> dict:
    """Conduct comprehensive market research analysis.

    Args:
        idea: Business idea or product description
        target_customer: Target customer segment
        geography: Geographic market (e.g., 'USA', 'India', 'Global')
        depth_level: Research depth (1=Quick, 2=Medium, 3=Deep)

    Returns:
        Research framework with areas to investigate
    """
    try:
        if depth_level not in [1, 2, 3]:
            return {"status": "error", "message": "depth_level must be 1, 2, or 3"}

        research_framework = {
            "idea": idea,
            "target_customer": target_customer,
            "geography": geography,
            "depth_level": depth_level,
            "research_areas": {}
        }

        if depth_level == 1:
            research_framework["research_areas"] = {
                "market_size": "Rough market size estimate",
                "competitors": "Top 3-5 competitors",
                "swot": "Brief SWOT analysis",
                "opportunities": "2-3 key opportunity areas"
            }
        elif depth_level == 2:
            research_framework["research_areas"] = {
                "market_size": "Market size with growth trends",
                "competitors": "Detailed competitor analysis",
                "customer_pain_points": "Key pain points",
                "existing_solutions": "Current solutions and gaps",
                "demand_signals": "Market demand indicators",
                "opportunities": "Strategic opportunities"
            }
        else:
            research_framework["research_areas"] = {
                "executive_summary": "High-level overview",
                "market_size_and_growth": "Comprehensive market analysis",
                "competitive_intelligence": "Full competitive landscape",
                "feature_gap_analysis": "Market gaps",
                "market_segmentation": "Customer segments and sizing",
                "geographic_trends": "Regional variations",
                "customer_sentiment": "Customer feedback analysis",
                "porters_five_forces": "Industry structure",
                "marketing_mix": "Product, Price, Place, Promotion, People, Process, Physical Evidence",
                "strategic_recommendations": "Actionable recommendations"
            }

        research_framework["instruction"] = f"Use web search to research each area. Provide data-driven insights for depth level {depth_level}."

        return research_framework

    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    mcp.run()