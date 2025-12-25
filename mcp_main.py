from fastmcp import FastMCP
from dotenv import load_dotenv
import logging
import os

from utils.analytics import analyze_sales_data
from utils.forecast import forecast_demand
from utils.azure_storage import get_user_files, load_dataframe, get_file_path, health_check

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("bizpilot-analytics")


@mcp.tool()
def check_system_health() -> dict:
    """Check system health and Azure Blob Storage connectivity.
    
    Returns:
        Health status including storage connectivity
    """
    try:
        health_status = health_check()
        return {
            "status": "success",
            "health": health_status
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@mcp.tool()
def list_available_files(user_id: str) -> dict:
    """List all available sales data files for a user.

    Args:
        user_id: The user's ID

    Returns:
        Dictionary with list of available files and metadata
    """
    try:
        files = get_user_files(user_id)
        return {
            "status": "success",
            "total_files": len(files),
            "files": files,
            "message": f"Found {len(files)} file(s)" if files else "No files found. Upload a file to get started."
        }
    except Exception as e:
        logger.error(f"Error listing files for user {user_id}: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to list files: {str(e)}"
        }


@mcp.tool()
def analyze_sales_file(filename: str, user_id: str) -> dict:
    """Analyze sales data from an uploaded file.
    Args:
        filename: Name or blob_name of the file to analyze
        user_id: The user's ID
    Returns:
        Dictionary with comprehensive sales analytics
    """
    try:
        # Get the actual blob name
        blob_name = get_file_path(filename, user_id)
        logger.info(f"Analyzing file {blob_name} for user {user_id}")
        
        # Load the dataframe - now with correct signature
        df = load_dataframe(blob_name)
        
        # Perform analytics
        analytics = analyze_sales_data(df)
        
        return {
            "status": "success",
            "filename": filename,
            "blob_name": blob_name,
            "analytics": analytics
        }
    except Exception as e:
        logger.error(f"Error analyzing file for user {user_id}: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to analyze file: {str(e)}",
            "filename": filename
        }


@mcp.tool()
def query_sales_data(filename: str, user_id: str, question: str) -> dict:
    """Answer specific questions about sales data.
    Args:
        filename: Name or blob_name of the file
        user_id: The user's ID
        question: The user's specific question about the data

    Returns:
        Data summary with analytics to answer the question
    """
    try:
        # Get the actual blob name
        blob_name = get_file_path(filename, user_id)
        logger.info(f"Querying file {blob_name} for user {user_id}: {question}")
        
        # Load the dataframe - now with correct signature
        df = load_dataframe(blob_name)
        
        # Get full analytics
        analytics = analyze_sales_data(df)

        return {
            "status": "success",
            "filename": filename,
            "blob_name": blob_name,
            "question": question,
            "data_summary": analytics,
            "instruction": "Use the data_summary to provide a clear, specific answer to the user's question with relevant numbers and insights."
        }
    except Exception as e:
        logger.error(f"Error querying file for user {user_id}: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to query file: {str(e)}",
            "question": question,
            "filename": filename
        }


@mcp.tool()
def forecast_sales_demand(filename: str, user_id: str, periods: int = 30) -> dict:
    """Forecast future sales demand using Facebook Prophet time series analysis.

    Args:
        filename: Name or blob_name of the file containing historical sales data
        user_id: The user's ID
        periods: Number of days to forecast (1-365, default: 30)

    Returns:
        Forecast data with predictions, confidence intervals, and trend analysis
    """
    try:
        # Validate periods
        if periods < 1 or periods > 365:
            return {
                "status": "error",
                "message": "Periods must be between 1 and 365 days"
            }

        # Get the actual blob name
        blob_name = get_file_path(filename, user_id)
        logger.info(f"Forecasting {periods} periods for file {blob_name} (user {user_id})")
        
        # Load the dataframe - now with correct signature
        df = load_dataframe(blob_name)
        
        # Generate forecast
        result = forecast_demand(df, periods)
        
        return {
            "status": "success",
            "filename": filename,
            "blob_name": blob_name,
            "forecast": result
        }
    except Exception as e:
        logger.error(f"Error forecasting for user {user_id}: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to generate forecast: {str(e)}",
            "filename": filename
        }


@mcp.tool()
def conduct_market_research(
    idea: str,
    target_customer: str,
    geography: str,
    depth_level: int = 2
) -> dict:
    """Conduct comprehensive market research analysis for a business idea.
    Args:
        idea: Business idea or product description
        target_customer: Target customer segment (e.g., "SMBs in tech", "millennials")
        geography: Geographic market (e.g., 'USA', 'India', 'Europe', 'Global')
        depth_level: Research depth - 1 (Quick), 2 (Medium), or 3 (Deep)

    Returns:
        Structured research framework with areas to investigate
    """
    try:
        # Validate inputs
        if not idea or len(idea.strip()) < 10:
            return {
                "status": "error",
                "message": "Business idea must be at least 10 characters"
            }
        
        if not target_customer or len(target_customer.strip()) < 3:
            return {
                "status": "error",
                "message": "Target customer must be specified (at least 3 characters)"
            }
        
        if not geography or len(geography.strip()) < 2:
            return {
                "status": "error",
                "message": "Geography must be specified (at least 2 characters)"
            }
        
        if depth_level not in [1, 2, 3]:
            return {
                "status": "error",
                "message": "depth_level must be 1 (Quick), 2 (Medium), or 3 (Deep)"
            }

        research_framework = {
            "status": "success",
            "idea": idea,
            "target_customer": target_customer,
            "geography": geography,
            "depth_level": depth_level,
            "research_areas": {}
        }

        if depth_level == 1:
            # Quick research - basic overview
            research_framework["research_areas"] = {
                "market_size": {
                    "description": "Rough market size estimate and growth rate",
                    "search_queries": [
                        f"{idea} market size {geography}",
                        f"{idea} industry growth rate"
                    ]
                },
                "competitors": {
                    "description": "Top 3-5 direct competitors",
                    "search_queries": [
                        f"top {idea} companies {geography}",
                        f"{idea} competitors analysis"
                    ]
                },
                "swot": {
                    "description": "Brief SWOT analysis",
                    "areas": ["Strengths", "Weaknesses", "Opportunities", "Threats"]
                },
                "opportunities": {
                    "description": "2-3 key opportunity areas",
                    "focus": "Market gaps and underserved segments"
                }
            }
            research_framework["instruction"] = "Conduct quick research for each area. Focus on high-level insights and key numbers."

        elif depth_level == 2:
            # Medium research - detailed analysis
            research_framework["research_areas"] = {
                "market_size": {
                    "description": "Market size with historical growth trends and projections",
                    "search_queries": [
                        f"{idea} market size {geography} 2024",
                        f"{idea} market forecast 2025-2030",
                        f"{idea} TAM SAM SOM"
                    ]
                },
                "competitors": {
                    "description": "Detailed competitor landscape analysis",
                    "search_queries": [
                        f"{idea} competitive landscape {geography}",
                        f"{idea} market leaders",
                        f"{idea} startup competitors"
                    ],
                    "include": ["Market share", "Pricing", "Key features", "Funding"]
                },
                "customer_pain_points": {
                    "description": "Key customer problems and unmet needs",
                    "search_queries": [
                        f"{target_customer} problems with {idea}",
                        f"{idea} customer complaints",
                        f"{target_customer} needs {geography}"
                    ]
                },
                "existing_solutions": {
                    "description": "Current solutions and their gaps",
                    "focus": "What exists, what's missing, differentiation opportunities"
                },
                "demand_signals": {
                    "description": "Market demand indicators",
                    "search_queries": [
                        f"{idea} search trends",
                        f"{idea} recent funding {geography}",
                        f"{target_customer} adoption rate {idea}"
                    ]
                },
                "opportunities": {
                    "description": "Strategic market opportunities",
                    "focus": "Whitespace, emerging trends, underserved niches"
                }
            }
            research_framework["instruction"] = "Conduct thorough research for each area. Provide specific data, numbers, and actionable insights."

        else:  # depth_level == 3
            # Deep research - comprehensive strategic analysis
            research_framework["research_areas"] = {
                "executive_summary": {
                    "description": "High-level market opportunity overview",
                    "include": ["Market attractiveness", "Key findings", "Recommendation"]
                },
                "market_size_and_growth": {
                    "description": "Comprehensive market analysis",
                    "search_queries": [
                        f"{idea} global market size",
                        f"{idea} market size {geography}",
                        f"{idea} CAGR forecast",
                        f"{idea} market drivers and restraints"
                    ],
                    "include": ["TAM/SAM/SOM", "Growth drivers", "Market maturity"]
                },
                "competitive_intelligence": {
                    "description": "Full competitive landscape mapping",
                    "search_queries": [
                        f"{idea} competitive analysis {geography}",
                        f"{idea} market share by company",
                        f"{idea} competitor strategies",
                        f"{idea} recent M&A activity"
                    ],
                    "include": ["Market leaders", "Challenger brands", "Niche players", "New entrants"]
                },
                "feature_gap_analysis": {
                    "description": "Market gaps and differentiation opportunities",
                    "focus": "Feature comparison matrix, unmet needs, innovation areas"
                },
                "market_segmentation": {
                    "description": "Customer segments and addressable markets",
                    "search_queries": [
                        f"{target_customer} demographics {geography}",
                        f"{idea} customer segments",
                        f"{target_customer} buying behavior"
                    ],
                    "include": ["Segment size", "Willingness to pay", "Acquisition cost"]
                },
                "geographic_trends": {
                    "description": "Regional market variations and opportunities",
                    "focus": "Geographic differences, emerging markets, expansion potential"
                },
                "customer_sentiment": {
                    "description": "Customer feedback and satisfaction analysis",
                    "search_queries": [
                        f"{idea} customer reviews",
                        f"{target_customer} satisfaction {idea}",
                        f"{idea} NPS score"
                    ]
                },
                "porters_five_forces": {
                    "description": "Industry structure analysis",
                    "forces": [
                        "Threat of new entrants",
                        "Bargaining power of suppliers",
                        "Bargaining power of buyers",
                        "Threat of substitutes",
                        "Competitive rivalry"
                    ]
                },
                "marketing_mix_7p": {
                    "description": "Strategic marketing analysis",
                    "elements": {
                        "Product": "Features, quality, differentiation",
                        "Price": "Pricing strategy, competitor pricing",
                        "Place": "Distribution channels, market access",
                        "Promotion": "Marketing channels, messaging",
                        "People": "Target audience, personas",
                        "Process": "Customer journey, user experience",
                        "Physical Evidence": "Brand perception, trust signals"
                    }
                },
                "strategic_recommendations": {
                    "description": "Actionable go-to-market recommendations",
                    "include": [
                        "Market entry strategy",
                        "Positioning and differentiation",
                        "Pricing strategy",
                        "Channel strategy",
                        "Key success factors",
                        "Risks and mitigation"
                    ]
                }
            }
            research_framework["instruction"] = "Conduct comprehensive research for each area. Use web search extensively. Provide detailed, data-driven analysis with specific numbers, trends, and strategic recommendations. This is a deep dive - be thorough."

        return research_framework

    except Exception as e:
        logger.error(f"Error creating research framework: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to create research framework: {str(e)}"
        }


if __name__ == "__main__":
    mcp.run()
