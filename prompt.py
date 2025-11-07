from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough

class SoilAnalysis(BaseModel):
    quality: str = Field(..., description="Brief summary of soil quality")
    recommended_crops: list[str] = Field(..., description="List of suitable crops")
    suggestions: str = Field(..., description="Detailed suggestions to improve soil")

parser = PydanticOutputParser(pydantic_object=SoilAnalysis)

template = PromptTemplate(
    template="""
You are an expert soil analyst and agronomist. Analyze the following soil report.
Use the retrieved context if helpful, but do not assume values not present in the report.

Retrieved context (may include related pages / prior reports):
{context}

Soil Report (to analyze now):
{soil_text}

{format_instructions}
""",
    input_variables=["soil_text", "context"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

def get_chain(llm):
    # We'll supply both soil_text and context at call time
    # and keep your same llm -> parser pipeline.
    return {"soil_text": RunnablePassthrough(), "context": RunnablePassthrough()} | template | llm | parser
