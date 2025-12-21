from fastmcp import FastMCP
from dotenv import load_dotenv

# Import utility functions
from utils.analytics import analyze_sales_data
from utils.forecast import forecast_demand
from utils.file_handler import list_files, get_file_path

load_dotenv()

# Create FastMCP server instance
mcp = FastMCP("bizpilot-analytics")


@mcp.tool()
def list_available_files(user_hash: str = None) -> dict:
    """List all available sales data files that can be analyzed.

    Args:
        user_hash: Optional user hash prefix to filter files

    Returns:
        Dictionary with status and list of available files
    """
    try:
        files = list_files(user_hash)
        return {
            "status": "success",
            "total_files": len(files),
            "files": files
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@mcp.tool()
def analyze_sales_file(filename: str) -> dict:
    """Analyze sales data from an uploaded file and return comprehensive statistics.

    Returns statistics including:
    - Total sales, averages, max/min values
    - Top and bottom products
    - Daily/weekly/monthly trends
    - Sales by category and region (if available)

    Args:
        filename: Name of the uploaded sales data file

    Returns:
        Dictionary with comprehensive sales analytics
    """
    try:
        filepath = get_file_path(filename)
        analytics = analyze_sales_data(filepath)
        return analytics
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@mcp.tool()
def query_sales_data(filename: str, question: str) -> dict:
    """Get detailed sales data to answer specific questions about the data.

    This returns raw statistics that you (the LLM) should interpret and explain.

    Args:
        filename: Name of the uploaded sales data file
        question: The user's question about the sales data

    Returns:
        Dictionary with the question and comprehensive data summary to analyze
    """
    try:
        filepath = get_file_path(filename)
        analytics = analyze_sales_data(filepath)

        return {
            "question": question,
            "data_summary": analytics,
            "instruction": "Analyze the data_summary above and provide a clear, specific answer to the user's question. Include relevant numbers, trends, and actionable insights."
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@mcp.tool()
def forecast_sales_demand(filename: str, periods: int = 30) -> dict:
    """Forecast future sales demand using Facebook Prophet time series model.

    Returns predictions with confidence intervals and trend analysis.

    Args:
        filename: Name of the uploaded sales data file with historical sales
        periods: Number of future periods (days) to forecast (default: 30, max: 365)

    Returns:
        Dictionary with forecast data, trend analysis, and model information
    """
    try:
        if periods < 1 or periods > 365:
            return {
                "status": "error",
                "message": "Periods must be between 1 and 365"
            }

        filepath = get_file_path(filename)
        result = forecast_demand(filepath, periods)
        return result
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@mcp.tool()
def conduct_market_research(
        idea: str,
        target_customer: str,
        geography: str,
        depth_level: int = 2
) -> dict:
    """Conduct comprehensive market research analysis.

    Returns a structured framework of research areas that you should investigate
    using web search and your knowledge.

    Args:
        idea: Business idea or product description
        target_customer: Target customer segment or persona
        geography: Geographic market (e.g., 'USA', 'India', 'Global')
        depth_level: Research depth (1=Quick scan, 2=Medium depth, 3=Deep analysis)

    Returns:
        Dictionary with research framework and areas to investigate
    """
    try:
        if depth_level not in [1, 2, 3]:
            return {
                "status": "error",
                "message": "depth_level must be 1, 2, or 3"
            }

        research_framework = {
            "idea": idea,
            "target_customer": target_customer,
            "geography": geography,
            "depth_level": depth_level,
            "research_areas": {}
        }

        if depth_level == 1:
            research_framework["research_areas"] = {
                "market_size": "Provide rough market size estimate",
                "competitors": "List top 3-5 competitors",
                "swot": "Brief SWOT analysis",
                "opportunities": "2-3 key opportunity areas"
            }
        elif depth_level == 2:
            research_framework["research_areas"] = {
                "market_size": "Market size with growth trends",
                "competitors": "Detailed competitor analysis (products, pricing, positioning)",
                "customer_pain_points": "Key pain points of target customers",
                "existing_solutions": "Current solutions and their limitations",
                "demand_signals": "Market demand indicators and trends",
                "opportunities": "Strategic opportunities with reasoning"
            }
        else:  # depth_level == 3
            research_framework["research_areas"] = {
                "executive_summary": "High-level overview",
                "market_size_and_growth": "Comprehensive market analysis",
                "competitive_intelligence": "Full competitive landscape",
                "feature_gap_analysis": "What's missing in the market",
                "market_segmentation": "Customer segments and sizing",
                "geographic_trends": "Regional variations and opportunities",
                "customer_sentiment": "Customer feedback and sentiment analysis",
                "porters_five_forces": "Industry structure analysis",
                "marketing_mix_7ps": "Product, Price, Place, Promotion, People, Process, Physical Evidence",
                "strategic_recommendations": "Actionable recommendations"
            }

        research_framework[
            "instruction"] = f"Use web search and your knowledge to research each area listed above. Provide detailed, data-driven insights for a depth level {depth_level} analysis."

        return research_framework

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


if __name__ == "__main__":
    # Run the MCP server
    mcp.run()