from langchain.chains import LLMChain, SequentialChain
from langchain.chains.base import Chain
from langchain.prompts import PromptTemplate
from typing import Dict, List, Any
import pandas as pd

class RetailAnalysisChain(Chain):
    """Custom LangChain for comprehensive retail analysis."""
    
    def __init__(self, llm, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm
        
    @property
    def input_keys(self) -> List[str]:
        return ["product", "store", "sales_data"]
    
    @property
    def output_keys(self) -> List[str]:
        return ["forecast_analysis", "anomaly_analysis", "inventory_recommendation", "business_insights"]
    
    def _call(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        product = inputs["product"]
        store = inputs["store"]
        sales_data = inputs["sales_data"]
        
        # Create analysis chains
        forecast_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["product", "store", "data"],
                template="Analyze the sales forecast for {product} at {store} based on this data: {data}"
            )
        )
        
        anomaly_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["product", "store", "data"],
                template="Identify and explain any anomalies in sales data for {product} at {store}: {data}"
            )
        )
        
        inventory_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["product", "store", "forecast"],
                template="Based on the forecast analysis: {forecast}, provide inventory recommendations for {product} at {store}"
            )
        )
        
        insights_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["product", "store", "forecast", "anomaly", "inventory"],
                template="""Provide business insights for {product} at {store}:
                Forecast: {forecast}
                Anomalies: {anomaly}
                Inventory: {inventory}
                
                Give actionable business recommendations."""
            )
        )
        
        # Execute chains sequentially
        forecast_result = forecast_chain.run(product=product, store=store, data=sales_data)
        anomaly_result = anomaly_chain.run(product=product, store=store, data=sales_data)
        inventory_result = inventory_chain.run(product=product, store=store, forecast=forecast_result)
        insights_result = insights_chain.run(
            product=product, store=store, 
            forecast=forecast_result, anomaly=anomaly_result, inventory=inventory_result
        )
        
        return {
            "forecast_analysis": forecast_result,
            "anomaly_analysis": anomaly_result,
            "inventory_recommendation": inventory_result,
            "business_insights": insights_result
        }

class SalesComparisonChain(Chain):
    """Chain for comparing sales across stores/products."""
    
    def __init__(self, llm, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm
        
    @property
    def input_keys(self) -> List[str]:
        return ["store_a", "store_b", "product", "sales_data_a", "sales_data_b"]
    
    @property
    def output_keys(self) -> List[str]:
        return ["comparison_analysis", "performance_insights", "recommendations"]
    
    def _call(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        store_a = inputs["store_a"]
        store_b = inputs["store_b"]
        product = inputs["product"]
        sales_a = inputs["sales_data_a"]
        sales_b = inputs["sales_data_b"]
        
        comparison_prompt = PromptTemplate(
            input_variables=["store_a", "store_b", "product", "sales_a", "sales_b"],
            template="""
            Compare sales performance for {product} between {store_a} and {store_b}:
            
            {store_a} sales: {sales_a}
            {store_b} sales: {sales_b}
            
            Provide:
            1. Performance comparison
            2. Key differences
            3. Recommendations for improvement
            """
        )
        
        comparison_chain = LLMChain(llm=self.llm, prompt=comparison_prompt)
        result = comparison_chain.run(
            store_a=store_a, store_b=store_b, product=product,
            sales_a=sales_a, sales_b=sales_b
        )
        
        return {
            "comparison_analysis": result,
            "performance_insights": "Analysis complete",
            "recommendations": "See comparison analysis"
        }

def create_retail_workflow(llm):
    """Create a comprehensive retail analysis workflow using LangChain."""
    
    # Define individual analysis steps
    data_analysis_prompt = PromptTemplate(
        input_variables=["data"],
        template="Analyze this retail sales data and extract key metrics: {data}"
    )
    
    trend_analysis_prompt = PromptTemplate(
        input_variables=["metrics"],
        template="Based on these metrics: {metrics}, identify sales trends and patterns."
    )
    
    recommendation_prompt = PromptTemplate(
        input_variables=["trends"],
        template="Given these trends: {trends}, provide actionable business recommendations."
    )
    
    # Create chains
    data_chain = LLMChain(llm=llm, prompt=data_analysis_prompt, output_key="metrics")
    trend_chain = LLMChain(llm=llm, prompt=trend_analysis_prompt, output_key="trends")
    recommendation_chain = LLMChain(llm=llm, prompt=recommendation_prompt, output_key="recommendations")
    
    # Combine into sequential workflow
    workflow = SequentialChain(
        chains=[data_chain, trend_chain, recommendation_chain],
        input_variables=["data"],
        output_variables=["metrics", "trends", "recommendations"],
        verbose=True
    )
    
    return workflow 