from utils.llm import call_llm


def do_market_research(idea: str, customer: str, geography: str, level: int) -> dict:
    """Perform market research at specified depth level (1=quick, 2=medium, 3=deep)"""

    if level == 1:
        prompt = f"""
You are a market research analyst. Provide a QUICK market scan for:

Idea: {idea}
Target Customer: {customer}
Geography: {geography}

Provide:
1. Estimated market size (rough)
2. Top 3-5 competitors
3. SWOT summary (brief)
4. 2-3 opportunity areas

Keep it concise and actionable.
"""
    elif level == 2:
        prompt = f"""
You are a market research analyst. Provide MEDIUM depth research for:

Idea: {idea}
Target Customer: {customer}
Geography: {geography}

Provide:
1. Market size estimates with sources
2. Detailed competitor analysis (products, pricing, positioning)
3. Customer pain points
4. Existing solutions breakdown
5. Demand signals and trends

Be detailed but stay under 1000 words.
"""
    else:
        prompt = f"""
You are a senior market research analyst. Provide DEEP research for:

Idea: {idea}
Target Customer: {customer}
Geography: {geography}

Provide a comprehensive report including:
1. Full competitive intelligence
2. Feature gap analysis
3. Market segmentation
4. Geographic trends
5. Customer sentiment analysis
6. Porter's 5 Forces
7. Marketing mix (7Ps)
8. Executive summary

Make it detailed and professional.
"""

    research_text = call_llm(prompt)

    return {
        "level": level,
        "idea": idea,
        "research": research_text
    }