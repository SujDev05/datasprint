from langchain.prompts import PromptTemplate
from langchain.prompts import FewShotPromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector

# Base system prompt for retail analytics
SYSTEM_PROMPT = """You are an expert retail analytics AI assistant. You help retailers understand their sales data, 
forecasts, anomalies, and inventory recommendations. Always provide actionable insights and explain complex 
concepts in simple terms."""

# Enhanced prompt templates using LangChain
FORECAST_EXPLANATION_TEMPLATE = PromptTemplate(
    input_variables=["product", "store", "forecast_data", "historical_data"],
    template="""
    {system_prompt}
    
    Based on the following data for {product} at {store}:
    
    Historical Sales Data:
    {historical_data}
    
    Forecast Data:
    {forecast_data}
    
    Please explain:
    1. The sales forecast trends for the next period
    2. Key patterns in the historical data
    3. Factors that might influence future sales
    4. Recommendations for the retailer
    
    Provide your analysis in a clear, business-friendly manner.
    """
)

ANOMALY_EXPLANATION_TEMPLATE = PromptTemplate(
    input_variables=["product", "store", "anomaly_data", "date"],
    template="""
    {system_prompt}
    
    Anomaly detected for {product} at {store} on {date}:
    
    Anomaly Details:
    {anomaly_data}
    
    Please analyze:
    1. Possible causes for this anomaly
    2. Whether this is a positive or negative trend
    3. Recommended actions for the retailer
    4. How to prevent similar issues in the future
    
    Focus on practical business insights.
    """
)

INVENTORY_RECOMMENDATION_TEMPLATE = PromptTemplate(
    input_variables=["product", "store", "forecast", "current_inventory", "safety_stock"],
    template="""
    {system_prompt}
    
    Inventory Analysis for {product} at {store}:
    
    Forecasted Demand: {forecast}
    Current Inventory: {current_inventory}
    Safety Stock Level: {safety_stock}
    
    Please provide:
    1. Optimal inventory level recommendation
    2. Reorder point calculation
    3. Risk assessment (overstock/understock)
    4. Cost implications
    5. Action items for inventory management
    
    Be specific with numbers and timelines.
    """
)

# Few-shot examples for better responses
FORECAST_EXAMPLES = [
    {
        "product": "Laptop",
        "store": "Electronics Store",
        "forecast": "Increasing trend, 15% growth expected",
        "explanation": "The laptop sales show a strong upward trend due to back-to-school season. Recommend increasing inventory by 20% to meet demand."
    },
    {
        "product": "Winter Jacket", 
        "store": "Clothing Store",
        "forecast": "Seasonal decline, 30% decrease expected",
        "explanation": "Winter jacket sales are declining as spring approaches. Recommend reducing inventory and offering discounts to clear stock."
    }
]

# Create example prompt template
EXAMPLE_PROMPT = PromptTemplate(
    input_variables=["product", "store", "forecast", "explanation"],
    template="Product: {product}\nStore: {store}\nForecast: {forecast}\nExplanation: {explanation}\n"
)

# Create few-shot prompt template
FORECAST_FEW_SHOT_TEMPLATE = FewShotPromptTemplate(
    example_selector=LengthBasedExampleSelector(
        examples=FORECAST_EXAMPLES,
        max_length=200,
        example_prompt=EXAMPLE_PROMPT
    ),
    example_prompt=EXAMPLE_PROMPT,
    prefix="Here are examples of sales forecast explanations:",
    suffix="Now explain the forecast for {product} at {store}:",
    input_variables=["product", "store"],
    example_separator="\n\n"
)

# Business intelligence prompt
BUSINESS_INSIGHTS_TEMPLATE = PromptTemplate(
    input_variables=["data_summary", "user_question"],
    template="""
    {system_prompt}
    
    Retail Data Summary:
    {data_summary}
    
    User Question: {user_question}
    
    Provide a comprehensive business analysis including:
    1. Data interpretation
    2. Key insights
    3. Business implications
    4. Recommended actions
    5. Risk factors to consider
    
    Format your response with clear sections and bullet points.
    """
)

# Quick response templates for common queries
QUICK_RESPONSES = {
    "trend": "Based on the data, {product} at {store} shows a {trend_direction} trend. {explanation}",
    "comparison": "Comparing {product} across stores: {store_a} has {metric_a} while {store_b} has {metric_b}. {insight}",
    "recommendation": "For {product} at {store}, I recommend {action} because {reasoning}."
}

# Export all templates
__all__ = [
    'SYSTEM_PROMPT',
    'FORECAST_EXPLANATION_TEMPLATE', 
    'ANOMALY_EXPLANATION_TEMPLATE',
    'INVENTORY_RECOMMENDATION_TEMPLATE',
    'FORECAST_FEW_SHOT_TEMPLATE',
    'BUSINESS_INSIGHTS_TEMPLATE',
    'QUICK_RESPONSES'
] 