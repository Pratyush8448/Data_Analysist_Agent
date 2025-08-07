import os
from openai import OpenAI
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self):
        self.api_key = os.getenv("AI_PROXY_API_KEY")
        self.base_url = os.getenv("AI_PROXY_BASE_URL")
        self.model_name = os.getenv("AI_MODEL_NAME", "gpt-4o-mini")
        
        if not self.api_key or not self.base_url:
            logger.warning("AI_PROXY_API_KEY or AI_PROXY_BASE_URL not set. LLM features will be disabled.")
            self.client = None
        else:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
    
    def is_available(self) -> bool:
        """Check if LLM client is properly configured"""
        return self.client is not None
    
    async def analyze_data(self, question: str, data_summary: str = "") -> Optional[str]:
        """Use LLM to analyze data and answer questions"""
        if not self.is_available():
            return None
        
        try:
            prompt = f"""
            You are a data analyst. Analyze the following data and answer the question.
            
            Question: {question}
            
            Data Summary: {data_summary}
            
            Please provide your answer in a structured format:
            
            1. Count: [number] (if asking for count)
            2. Name: [movie/court name] (if asking for specific item)
            3. Correlation/Slope: [numeric value] (if asking for correlation or regression)
            4. Additional insights: [brief explanation]
            
            Be specific and provide exact numbers when possible. If you cannot answer from the given data, say so.
            """
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful data analyst assistant. Provide precise, structured answers with specific numbers and names."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.1
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return None
    
    async def generate_insights(self, data_description: str, analysis_results: List[Any]) -> Optional[str]:
        """Generate insights from analysis results"""
        if not self.is_available():
            return None
        
        try:
            prompt = f"""
            Based on the following data analysis results, provide key insights:
            
            Data: {data_description}
            Results: {analysis_results}
            
            Provide 2-3 key insights in bullet points.
            """
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a data analyst providing insights."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.2
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"LLM insight generation failed: {e}")
            return None

# Global LLM client instance
llm_client = LLMClient() 