from pathlib import Path
from textwrap import dedent

from agno.agent import Agent

from agno.models.google import Gemini
from agno.embedder.google import GeminiEmbedder

from agno.storage.sqlite import SqliteStorage

from agno.memory.v2.db.sqlite import SqliteMemoryDb
from agno.memory.v2.memory import Memory

from agno.vectordb.qdrant import Qdrant
from agno.vectordb.search import SearchType

from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.tavily import TavilyTools
from agno.tools.wikipedia import WikipediaTools

from agno.tools.website import WebsiteTools
from agno.tools.firecrawl import FirecrawlTools

from agno.tools.csv_toolkit import CsvTools
from agno.tools.pandas import PandasTools

from agno.tools.file import FileTools
from agno.tools.python import PythonTools
from agno.tools.shell import ShellTools

from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.document.chunking.agentic import AgenticChunking

import os

from dotenv import load_dotenv
load_dotenv()

# ************* Paths *************
cwd = Path(__file__).parent
knowledge_dir = cwd.joinpath("knowledge/storybrand/")
output_dir = cwd.joinpath("output")

# # Create the output directory if it does not exist
# output_dir.mkdir(parents=True, exist_ok=True)
# *******************************
agent_storage_file: str = "tmp/agents.db"
memory_storage_file: str = "tmp/memory.db"

memory_db = SqliteMemoryDb(table_name="brainspark_architect_memory", 
                           db_file=memory_storage_file)

# Create Memory object that wraps the database
memory = Memory(db=memory_db)

# Storage
agent_storage = SqliteStorage(table_name="brainspark_architect_agent", db_file=agent_storage_file)


# Vectordb - Fix URL format and add embedder here
api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
collection_name = "brainspark_architect_knowledge"


vector_db = Qdrant(
    collection=collection_name,
    url=qdrant_url,
    api_key=api_key,
    embedder=GeminiEmbedder(id="text-embedding-004", 
                            dimensions=768, 
                            api_key=os.getenv("1DCNNGEMINI")),
)

# Knowledge Base - Remove embedder from here
knowledge_base = PDFKnowledgeBase(
    path=knowledge_dir,
    vector_db=vector_db,
    reader=PDFReader(chunk=True, chunk_size=5000),
    chunking_strategy=AgenticChunking()
)


# Agents
brandscript_architect = Agent(
    name="BrandScript Architect",
    agent_id="brandscript_architect",
    model=Gemini(id="gemini-2.0-flash", temperature=0.2, api_key=os.getenv("1DCNNGEMINI")),
    description="""
    You are an expert StoryBrand (SB7) Guide. Your purpose is to construct a clear and compelling Master BrandScript. Identify the Character (the customer), their Problem (External, Internal, Philosophical, 
    and the Villain), position BrainSpark Digital as the Guide (with Empathy and Authority), define a clear Plan, craft strong Calls to Action (Direct and Transitional), articulate what Failure is avoided, 
    and paint a vivid picture of Success. Define the Character Transformation and the overarching Controlling Idea. Ensure all 7 SB7 elements are robustly addressed. Your output must be in Markdown format.
    Always ask multiple Questions to the Knowledge Base to get the information you need.
    
""",
    instructions="""
    1. Analyze all provided input materials to accurately identify the 'Character'—the primary customer or client segment. Define their fundamental desires and aspirations as they relate to the services offered.
    2. Articulate the multifaceted 'Problem' that the Character encounters, distinguishing between External problems (tangible challenges), Internal problems (frustrations and self-doubts), and Philosophical problems (the inherent 'wrongness' of the situation). Clearly identify the 'Villain' that personifies these problems.
    3. Position BrainSpark Digital as the empathetic and competent 'Guide.' Emphasize genuine understanding of the Character's frustrations (Empathy) and showcase the expertise, credibility, and proven ability to solve their problems (Competency/Authority).
    4. Develop a clear, simple, and actionable 3 to 4-step 'Plan' that the Character can follow to engage with the Guide and overcome their problems. This plan should demystify the process and reduce perceived risk.
    5. Craft compelling 'Calls to Action' (CTAs) that prompt the Character to take the next step. Differentiate between Direct CTAs and Transitional CTAs.
    6. Clearly define what specific 'Failure' or negative consequences the Character will avoid by taking the proposed action and engaging with the Guide. This creates urgency and highlights the stakes.
    7. Paint a vivid and aspirational picture of 'Success.' Describe the positive outcomes and the desirable transformation the Character will experience after successfully navigating their problems with the Guide's help.
    8. Synthesize all elements into a cohesive Master BrandScript. Distill a concise and impactful One-Liner and a memorable Controlling Idea that encapsulates the core message.
    9. Generate initial drafts for BrandScript elements. Review, refine, and customize these drafts to ensure authenticity and alignment with the brand's unique value proposition.
""",
    memory=memory,
    enable_user_memories=True,
    knowledge=knowledge_base,
    search_knowledge=True,
    add_references=True,
    enable_agentic_knowledge_filters=True,
    storage=agent_storage,
    add_history_to_messages=True,
    add_datetime_to_instructions=True,
    user_id="z4hid",
    debug_mode=True,
    markdown=True,
    role="brandscript_architect",
)

if __name__ == "__main__":
    try:
        # Comment out after first run
        # knowledge_base.load(recreate=False)
        
        brandscript_architect.print_response("Develop a comprehensive Brandscript for our new AI-driven analytics service. We are brainspark digital. We are targeting small businesses.", stream=True)
    except Exception as e:
        print(f"Error: {e}")
