from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0, max_tokens=1024)
parser = JsonOutputParser()

prompt = """
You are a strategic Project Router.
Analyze these tasks and group them into 1 to 4 distinct "Projects" based on their department or focus area (e.g., Marketing, Logistics, Finance).
Return ONLY this strict JSON format:
{{
    "projects": [
        {{
            "project_name": "Name of the distinct project (e.g. Maison Trade Show: Marketing)",
            "tasks": ["task 1", "task 2"]
        }}
    ]
}}

TASKS:
{tasks}
"""
chain = ChatPromptTemplate.from_template(prompt) | llm | parser

import asyncio
async def main():
    res = await chain.ainvoke({"tasks": ["buy wood", "hammer nails", "buy ads", "post on facebook"]})
    print(res)

asyncio.run(main())
