from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableMap, RunnablePassthrough

# Define output structure
class SoilAnalysis(BaseModel):
    quality: str = Field(..., description="Brief summary of soil quality")
    recommended_crops: list[str] = Field(..., description="List of suitable crops")
    suggestions: str = Field(..., description="Detailed suggestions to improve soil")

# Setup parser
parser = PydanticOutputParser(pydantic_object=SoilAnalysis)

# Prompt template
template = PromptTemplate(
    template="""
You are an expert soil analyst and agronomist. Analyze the following soil report and respond in the specified format.

Soil Report:
{soil_text}

{format_instructions}
""",
    input_variables=["soil_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Runnable chain
def get_chain(llm):
    return {"soil_text": RunnablePassthrough()} | template | llm | parser
