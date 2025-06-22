import chainlit as cl
from agents.storybrand import brandscript_architect
from agents.seo import seo_specialist
from agents.product_manager import product_manager
from agents.growth_hacker import growth_hacker
from agents.social_media_manager import social_media_manager
from agents.script_writer import script_writer
from agents.content_creator import content_creator
import asyncio
from typing import Dict, Any, Optional
import logging
import re
import csv
import os
from datetime import datetime
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

# Import data layer requirements from Chainlit. This helps with saving chat data.
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer

# This sets up a way to record messages or warnings from the app while it's running.
# It's like a diary for the program to write down what it's doing, which is helpful for fixing problems.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# This is the name of the file where we'll save conversations for later review.
# A CSV file is a simple type of spreadsheet.
EVALUATION_CSV_FILE = "evaluation.csv"

# This is the web address (URL) for our online database.
# It's where all the chat history and user information will be stored so it's not lost.
DB_CONNINFO = "postgresql+asyncpg://chainlit_mwqo_user:LlQUvFvkpvGLiJSlr9597eDafKOT9Cxj@dpg-d1c766idbo4c73ckr5i0-a.oregon-postgres.render.com/chainlit_mwqo"

async def create_db_and_tables():
    """
    This function connects to our database and creates all the necessary tables.
    Tables are like spreadsheets inside a database to organize information.
    We need tables for users, conversations (threads), messages (steps), etc.
    It only creates them if they don't already exist, so we don't erase data by accident.
    """
    # Create a connection "engine" to talk to our database.
    engine = create_async_engine(DB_CONNINFO)
    # Start a connection to the database. The 'async with' part makes sure it closes properly.
    async with engine.begin() as conn:
        # The following commands create each table we need in the database.

        # Table to store user information like their username and when they were created.
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                "id" UUID PRIMARY KEY,
                "identifier" TEXT NOT NULL UNIQUE,
                "metadata" JSONB NOT NULL,
                "createdAt" TEXT
            );
        """))
        # Table to store conversation threads. Each separate chat is a "thread".
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS threads (
                "id" UUID PRIMARY KEY,
                "createdAt" TEXT,
                "name" TEXT,
                "userId" UUID,
                "userIdentifier" TEXT,
                "tags" TEXT[],
                "metadata" JSONB,
                FOREIGN KEY ("userId") REFERENCES users("id") ON DELETE CASCADE
            );
        """))
        # Table to store each individual message or "step" in a conversation.
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS steps (
                "id" UUID PRIMARY KEY,
                "name" TEXT NOT NULL,
                "type" TEXT NOT NULL,
                "threadId" UUID NOT NULL,
                "parentId" UUID,
                "streaming" BOOLEAN NOT NULL,
                "waitForAnswer" BOOLEAN,
                "isError" BOOLEAN,
                "metadata" JSONB,
                "tags" TEXT[],
                "input" TEXT,
                "output" TEXT,
                "createdAt" TEXT,
                "command" TEXT,
                "start" TEXT,
                "end" TEXT,
                "generation" JSONB,
                "showInput" TEXT,
                "language" TEXT,
                "indent" INT,
                "defaultOpen" BOOLEAN,
                FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
            );
        """))
        # Table to store special items in a chat, like images, files, or videos.
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS elements (
                "id" UUID PRIMARY KEY,
                "threadId" UUID,
                "type" TEXT,
                "url" TEXT,
                "chainlitKey" TEXT,
                "name" TEXT NOT NULL,
                "display" TEXT,
                "objectKey" TEXT,
                "size" TEXT,
                "page" INT,
                "language" TEXT,
                "forId" UUID,
                "mime" TEXT,
                "props" JSONB,
                FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
            );
        """))
        # Table to store user feedback (e.g., if a user "liked" or "disliked" a message).
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS feedbacks (
                "id" UUID PRIMARY KEY,
                "forId" UUID NOT NULL,
                "threadId" UUID NOT NULL,
                "value" INT NOT NULL,
                "comment" TEXT,
                FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
            );
        """))
    # Close the connection engine to free up resources.
    await engine.dispose()
    logger.info("Database tables checked/created.")

# This line runs the function to create the database tables right when the app starts.
# `asyncio.run()` is how we run a special `async` function from regular Python code.
asyncio.run(create_db_and_tables())


def init_evaluation_csv():
    """
    This function creates the CSV file for evaluations if it's not already there.
    We add the column titles (headers) to it so we know what each column means.
    """
    if not os.path.exists(EVALUATION_CSV_FILE):
        with open(EVALUATION_CSV_FILE, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['timestamp', 'user', 'agent', 'input', 'output'])

def save_to_evaluation(user_input: str, ai_output: str, user_id: str = "unknown", agent_name: str = "unknown"):
    """
    This function saves a piece of conversation into our evaluation CSV file.
    This helps the developers see how well the AI agents are performing so they can be improved.
    """
    try:
        # First, make sure the CSV file exists and has headers.
        init_evaluation_csv()
        # Open the file in "append" mode, which means we add new rows without deleting old ones.
        with open(EVALUATION_CSV_FILE, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            timestamp = datetime.now().isoformat()
            # Write the new row with all the conversation details.
            writer.writerow([timestamp, user_id, agent_name, user_input, ai_output])
        return True
    except Exception as e:
        # If something goes wrong, log the error.
        logger.error(f"Error saving to evaluation CSV: {e}")
        return False

# This is a special Chainlit "decorator". It tells Chainlit how to save and load all chat data.
# We are telling it to use the SQLAlchemyDataLayer, which knows how to talk to our PostgreSQL database.
@cl.data_layer
def get_data_layer():
    """
    This function provides the "data layer" that Chainlit uses for persistence.
    "Persistence" means saving data so it's not lost when the app closes or the user leaves.
    """
    return SQLAlchemyDataLayer(
        conninfo=DB_CONNINFO,
    )

# This dictionary holds all our different AI "agents".
# Think of it as a team of specialists. Each agent has a name, a job, an icon, and other info.
# This makes it easy to manage them and switch between them.
AGENTS = {
    "BrandScript Architect": {
        "agent": brandscript_architect, # The actual agent code that does the thinking.
        "icon": "🎯",                   # An emoji to show in the user interface.
        "color": "#8B5CF6",            # A color for styling things in the UI.
        "expertise": "Brand Strategy & StoryBrand Framework", # A short description of what it's good at.
        "keywords": ["brand", "storybrand", "sb7", "strategy", "story", "narrative", "customer", "journey"], # Words to identify this agent.
        "alias": ["brand", "storybrand", "sb7", "architect"] # Nicknames for this agent to make it easier to call.
    },
    "SEO Specialist": {
        "agent": seo_specialist,
        "icon": "🔍", 
        "color": "#10B981",
        "expertise": "Search Engine Optimization & Lean SEO",
        "keywords": ["seo", "search", "optimization", "keywords", "ranking", "traffic", "organic"],
        "alias": ["seo", "search", "specialist"]
    },
    "Product Manager": {
        "agent": product_manager,
        "icon": "📊",
        "color": "#3B82F6", 
        "expertise": "Product Strategy & Market Analysis",
        "keywords": ["product", "strategy", "market", "analysis", "roadmap", "features", "pricing"],
        "alias": ["product", "pm", "manager"]
    },
    "Growth Hacker": {
        "agent": growth_hacker,
        "icon": "📈",
        "color": "#F59E0B",
        "expertise": "Growth Strategy & AARRR Funnel",
        "keywords": ["growth", "aarrr", "funnel", "acquisition", "retention", "experiments", "metrics"],
        "alias": ["growth", "hacker", "gh"]
    },
    "Social Media Manager": {
        "agent": social_media_manager,
        "icon": "📱",
        "color": "#EF4444",
        "expertise": "Social Media Strategy & Engagement",
        "keywords": ["social", "media", "engagement", "content", "posts", "community", "viral"],
        "alias": ["social", "media", "smm"]
    },
    "Script Writer": {
        "agent": script_writer,
        "icon": "🎬",
        "color": "#8B5CF6",
        "expertise": "Video Scripts & Content Creation",
        "keywords": ["script", "video", "content", "writing", "storytelling", "narrative", "dialogue"],
        "alias": ["script", "writer", "video"]
    },
    "Content Creator": {
        "agent": content_creator,
        "icon": "✍️",
        "color": "#06B6D4",
        "expertise": "Blog Posts & Marketing Content",
        "keywords": ["content", "blog", "articles", "writing", "marketing", "copy", "editorial"],
        "alias": ["content", "creator", "writer", "blog"]
    }
}

def find_agent_by_command(command: str) -> Optional[str]:
    """
    This function helps find the right agent based on what the user types.
    For example, if the user types "seo", this function will find the "SEO Specialist".
    It checks agent names, aliases (nicknames), and keywords to find a match.
    """
    command_lower = command.lower().strip()
    
    # First, check if the command matches an agent's full name exactly.
    for agent_name in AGENTS.keys():
        if agent_name.lower() == command_lower:
            return agent_name
    
    # If not, check if it matches an agent's nickname (alias).
    for agent_name, agent_info in AGENTS.items():
        if command_lower in [alias.lower() for alias in agent_info.get("alias", [])]:
            return agent_name
    
    # Finally, check if it matches any of the agent's special keywords.
    for agent_name, agent_info in AGENTS.items():
        if command_lower in [keyword.lower() for keyword in agent_info.get("keywords", [])]:
            return agent_name
    
    # If no agent is found after all checks, return nothing (None).
    return None

def parse_agent_command(message: str) -> tuple[Optional[str], str]:
    """
    This function checks if a user's message is a command to switch agents.
    For example, it looks for things like "/switch seo" or "@growth tell me about funnels".
    It uses "regular expressions" (the `re` library), which are a powerful way to find patterns in text.
    """
    # Pattern 1: Looks for commands starting with /switch, /agent, or /use, followed by an agent name.
    switch_pattern = r'^/(switch|agent|use)\s+(.+)$'
    match = re.match(switch_pattern, message.strip(), re.IGNORECASE)
    
    if match:
        # If a match is found, find the agent and return it. The message part is empty because it's just a command.
        agent_name = find_agent_by_command(match.group(2))
        return agent_name, ""
    
    # Pattern 2: Looks for commands starting with @ followed by an agent name and a message.
    at_pattern = r'^@(\w+)\s+(.+)$'
    match = re.match(at_pattern, message.strip(), re.IGNORECASE)
    
    if match:
        # If a match is found, find the agent and return it along with the rest of the message.
        agent_name = find_agent_by_command(match.group(1))
        if agent_name:
            return agent_name, match.group(2)
    
    # If no command patterns are found, it's just a regular message.
    return None, message

async def switch_agent(new_agent_name: str, user_message: str = "") -> bool:
    """
    This function handles the logic for switching to a different agent.
    It updates the user's session to use the new agent and sends a confirmation message.
    """
    try:
        if new_agent_name not in AGENTS:
            # If the agent name doesn't exist, we can't switch.
            return False
        
        # Get all the information about the new agent from our AGENTS dictionary.
        new_agent_info = AGENTS[new_agent_name]
        new_agent = new_agent_info["agent"]
        
        # The "user session" is a place to store information about the current user's chat.
        # We update it to remember which agent is now active.
        cl.user_session.set("agent", new_agent)
        cl.user_session.set("agent_info", new_agent_info)
        cl.user_session.set("chat_profile", new_agent_name)
        
        # Create a friendly message to tell the user that the switch was successful.
        switch_msg = f"""
🔄 **Agent Switched Successfully!**

**Now chatting with:** {new_agent_info['icon']} **{new_agent_name}**

**Expertise:** {new_agent_info['expertise']}

**What I can help you with:**
{new_agent.description[:300]}...

---

💡 **Quick Commands:**
- Type `/switch [agent]` to switch agents
- Type `@[agent] [message]` to send a message to a specific agent
- Type `/agents` to see all available agents
- Type `/help` for more commands

{new_agent_info['icon']} **{new_agent_name}** is ready to help!
        """
        
        # Send the confirmation message to the user interface.
        await cl.Message(
            content=switch_msg,
            author=f"{new_agent_info['icon']} {new_agent_name}"
        ).send()
        
        return True
        
    except Exception as e:
        # If something goes wrong, log the error and return False.
        logger.error(f"Error switching agent: {e}")
        return False

async def show_available_agents():
    """This function creates and sends a message that lists all the available AI agents."""
    agents_list = "## 🤖 Available Agents\n\n"
    
    # Go through each agent in our AGENTS dictionary and add its details to the list.
    for name, info in AGENTS.items():
        agents_list += f"### {info['icon']} **{name}**\n"
        agents_list += f"**Expertise:** {info['expertise']}\n"
        agents_list += f"**Keywords:** {', '.join(info['keywords'][:5])}\n"
        agents_list += f"**Aliases:** {', '.join(info['alias'])}\n\n"
    
    # Add a helpful section explaining how to use the commands.
    agents_list += """
## 🔄 How to Switch Agents

**Command Methods:**
- `/switch [agent_name]` - Switch to a specific agent
- `/agent [agent_name]` - Switch to a specific agent  
- `/use [agent_name]` - Switch to a specific agent

**Quick Chat Method:**
- `@[agent_name] [your_message]` - Send message to specific agent

**Examples:**
- `/switch seo` - Switch to SEO Specialist
- `/agent brand` - Switch to BrandScript Architect
- `@growth How can I improve my funnel?` - Ask Growth Hacker directly

**Other Commands:**
- `/agents` - Show this list
- `/current` - Show current agent
- `/help` - Show all commands
    """
    
    # Send the final list to the user as a message.
    await cl.Message(
        content=agents_list,
        author="🤖 Agent Directory"
    ).send()

async def show_current_agent():
    """This function shows details about the agent that is currently active."""
    current_agent_name = cl.user_session.get("chat_profile", "BrandScript Architect")
    agent_info = AGENTS.get(current_agent_name, AGENTS["BrandScript Architect"])
    
    # Create a message with the current agent's information.
    current_info = f"""
## {agent_info['icon']} **Current Agent: {current_agent_name}**

**Expertise:** {agent_info['expertise']}

**What I can help you with:**
{agent_info['agent'].description[:400]}...

**Keywords:** {', '.join(agent_info['keywords'])}
**Aliases:** {', '.join(agent_info['alias'])}

---

💡 **Want to switch?** Use `/switch [agent_name]` or `/agents` to see all options.
    """
    
    # Send the message to the user.
    await cl.Message(
        content=current_info,
        author=f"{agent_info['icon']} {current_agent_name}"
    ).send()

async def show_help():
    """This function shows a help message with all the available commands."""
    help_text = """## 🆘 Quick Help

**Agent Commands:**
- `/switch [agent]` - Switch to a specific agent
- `@[agent] [message]` - Send message directly to an agent
- `/agents` - Show all available agents

**Features:**
- 📊 **Add to Evaluation** button saves conversations to CSV
- 📋 **Copy Response** button for easy sharing

**Agent Shortcuts:**
Brand • SEO • Product • Growth • Social • Script • Content

**Examples:**
- `/switch seo` - Switch to SEO Specialist  
- `@growth How can I improve my funnel?` - Ask Growth Hacker
    """
    
    # Send the help message to the user.
    await cl.Message(
        content=help_text,
        author="🆘 Help System"
    ).send()

# This decorator tells Chainlit how to handle users logging in.
# This specific one allows any user to log in if they provide a username in the "header" of the request.
# It's often used for simple authentication.
@cl.header_auth_callback
def header_auth_callback(headers: Dict) -> Optional[cl.User]:
    # It gets the username from a header called 'x-username'. If that doesn't exist, it defaults to 'guest'.
    username = headers.get("x-username", "guest")
    return cl.User(identifier=username, metadata={"role": "user", "provider": "header"})

# This decorator is for another type of login: using a username and password.
@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    # For this example, it accepts any username and password and creates a new user.
    # In a real app, you would check the password against a database here.
    return cl.User(
        identifier=username,
        metadata={
            "role": "user",
            "provider": "credentials"
        }
    )

# This decorator sets up the different "chat profiles" that a user can choose from when they start a chat.
# Each profile corresponds to one of our AI agents.
@cl.set_chat_profiles
async def chat_profile():
    """Set up different agent profiles for the UI with enhanced styling and user-based filtering."""
    profiles = []
    
    # Create a chat profile for each agent in our AGENTS dictionary.
    for name, agent_info in AGENTS.items():
        profiles.append(
            cl.ChatProfile(
                name=name,
                markdown_description=f"**{agent_info['expertise']}**\n\n{agent_info['agent'].description[:200]}..." if len(agent_info['agent'].description) > 200 else agent_info['agent'].description,
                icon=agent_info['icon']
            )
        )
    
    return profiles

# This decorator sets up the "starter" prompts.
# These are suggestion buttons that appear at the beginning of a chat to help the user get started.
@cl.set_starters
async def set_starters():
    """Set up contextual starter prompts."""
    return [
        cl.Starter(
            label="🤖 Show All Agents",
            message="/agents",
            icon="🤖"
        ),
        cl.Starter(
            label="🎯 Create Master BrandScript",
            message="Help me create a comprehensive StoryBrand (SB7) Master BrandScript for my business",
            icon="🎯"
        ),
        cl.Starter(
            label="🔍 SEO Website Audit",
            message="Analyze my website's SEO performance and provide optimization recommendations",
            icon="🔍"
        ),
        cl.Starter(
            label="📈 AARRR Funnel Analysis",
            message="Analyze my customer acquisition funnel and identify growth opportunities",
            icon="📈"
        ),
        cl.Starter(
            label="🆘 Help & Commands",
            message="/help",
            icon="🆘"
        )
    ]

# This decorator specifies what should happen when a user starts a new chat.
@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session with the selected agent and authenticated user."""
    # Get the user who just logged in.
    user = cl.user_session.get("user")
    # Find out which chat profile (agent) the user selected. Default to BrandScript Architect.
    chat_profile = cl.user_session.get("chat_profile", "BrandScript Architect")
    
    # Get the information for the selected agent.
    agent_info = AGENTS.get(chat_profile)
    if not agent_info:
        agent_info = AGENTS["BrandScript Architect"]  # Use a default if something goes wrong.
    
    agent = agent_info["agent"]
    
    # Store the active agent and its info in the user's session so we can remember it.
    cl.user_session.set("agent", agent)
    cl.user_session.set("agent_info", agent_info)
    cl.user_session.set("chat_profile", chat_profile)
    
    is_first_login = user.metadata.get("first_login", False) if user else False
    
    # Create a nice, clean welcome message.
    welcome_msg = f"""# {agent_info['icon']} Welcome to BrainSpark Digital!

**Currently active:** {chat_profile} - {agent_info['expertise']}

{agent.description[:200]}...

**Quick commands:** `/switch [agent]` • `/agents` • `/help`
    """
    
    # If it's the user's first time, add a special welcome note.
    if is_first_login:
        welcome_msg += f"""
        
🎉 **Welcome to BrainSpark Digital!** This appears to be your first time here. 
Your account has been created and all your conversations will be saved for future reference.

🚀 **Pro Tip:** Try typing `/agents` to see all available AI specialists!
        """
    
    # Send the final welcome message to the user.
    welcome_element = cl.Message(
        content=welcome_msg,
        author=f"{chat_profile} Assistant",
    )
    await welcome_element.send()

# This decorator is for resuming a chat that was saved in the database.
@cl.on_chat_resume
async def on_chat_resume(thread: Dict[str, Any]):
    """
    This function runs when a user continues a conversation from their chat history.
    It makes sure to load the correct agent that was being used in that conversation.
    """
    # Get the chat profile (agent name) that was saved with the conversation.
    chat_profile = cl.user_session.get("chat_profile", "BrandScript Architect")
    
    # Find the agent's info.
    agent_info = AGENTS.get(chat_profile)
    if not agent_info:
        agent_info = AGENTS["BrandScript Architect"]  # Default if needed.
    
    agent = agent_info["agent"]
    
    # Put the correct agent back into the user's session so the conversation can continue smoothly.
    cl.user_session.set("agent", agent)
    cl.user_session.set("agent_info", agent_info)
    
    logger.info(f"Chat resumed for thread ID {thread.get('id')}. Restored agent: {chat_profile}")

# This decorator is the most important one. It runs every time the user sends a message.
@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages with enhanced agent switching and data persistence."""
    
    # Get the current agent and user info from the session.
    agent = cl.user_session.get("agent")
    agent_info = cl.user_session.get("agent_info")
    chat_profile = cl.user_session.get("chat_profile", "BrandScript Architect")
    user = cl.user_session.get("user")
    
    # Make sure an agent and user are properly loaded before continuing.
    if not agent or not agent_info:
        await cl.Message(content="❌ Agent not initialized. Please refresh the page and select an agent.", author="System").send()
        return
    if not user:
        await cl.Message(content="❌ User not authenticated. Please refresh the page and log in.", author="System").send()
        return
    
    # Get the text content of the user's message.
    user_message = message.content.strip()
    
    # Check if the message is a command to switch agents.
    target_agent, processed_message = parse_agent_command(user_message)
    
    if target_agent:
        # If it is a command, switch to the new agent.
        success = await switch_agent(target_agent, processed_message)
        if success and processed_message:
            # If the switch was successful and there was a message included (like "@agent message"),
            # then we process that message with the new agent.
            agent = cl.user_session.get("agent")
            agent_info = cl.user_session.get("agent_info")
            chat_profile = cl.user_session.get("chat_profile")
            user_message = processed_message
        elif success:
            # If it was just a switch command, we're done here.
            return
        else:
            # If the agent name was invalid, tell the user.
            await cl.Message(content=f"❌ Could not find agent matching '{target_agent}'. Type `/agents` to see available agents.", author="System").send()
            return
    
    # Check for other commands like /agents, /current, or /help.
    if user_message.lower() == "/agents":
        await show_available_agents()
        return
    elif user_message.lower() == "/current":
        await show_current_agent()
        return
    elif user_message.lower() == "/help":
        await show_help()
        return
    
    # Log the interaction for analytics.
    logger.info(f"User {user.identifier} ({user.metadata.get('role', 'user')}) sent message to {chat_profile}")
    
    # Create action buttons that will appear with the AI's response.
    actions = [
        cl.Action(
            name="add_to_evaluation",
            label="📊 Add to Evaluation",
            description="Save this conversation to evaluation dataset"
        ),
        cl.Action(
            name="copy_response",
            label="📋 Copy Response",
            description="Copy AI response for easy sharing"
        )
    ]
    
    # Create the message bubble for the AI's response, but leave it empty for now.
    response_msg = cl.Message(
        content="",
        author=f"{agent_info['icon']} {chat_profile}",
        actions=actions
    )
    
    # Show a "thinking" indicator in the UI so the user knows the app is working.
    step_name = f"{agent_info['icon']} {chat_profile} Processing"
    async with cl.Step(name=step_name, type="llm") as status_step:
        status_step.input = user_message
        
        try:
            # Tell the agent to process the user's message and stream the response back.
            # "Streaming" means the response appears word by word, like someone typing.
            response_stream = await cl.make_async(agent.run)(user_message, stream=True)

            content = ""
            for event in response_stream:
                if event.event == "RunResponseContent":
                    # As each piece of the response arrives, add it to the message bubble.
                    await response_msg.stream_token(event.content)
                    content += event.content
                
            await status_step.stream_token("✅ **Analysis complete!**\n")
                    
        except Exception as e:
            # If an error happens, log it and show a helpful error message to the user.
            logger.error(f"Error during agent execution for user {user.identifier}: {e}")
            await status_step.stream_token(f"❌ **Error: {str(e)}**\n")
            error_msg = f"""
❌ **Something went wrong!** 

**Error Details:** {str(e)[:100]}...

**Try this:**
- Refresh the page and try again
- Simplify your question
- Switch to a different agent using `/switch [agent]`
- Check your internet connection

**Need help?** The issue has been logged for our team to review.
            """
            await cl.Message(content=error_msg, author="System").send()
            return
        
        status_step.output = f"✅ Successfully processed query with {chat_profile} for user {user.identifier}"
    
    # Save the last conversation turn in the session so our action buttons can use it.
    if content:
        cl.user_session.set("last_user_input", user_message)
        cl.user_session.set("last_ai_output", content)
        cl.user_session.set("last_agent_name", chat_profile)
    
    # Send the final, complete message to the UI.
    if response_msg.content or content:
        await response_msg.send()

# This decorator specifies what to do when a chat session ends.
@cl.on_chat_end
async def on_chat_end():
    """Clean up when chat ends with enhanced logging."""
    user = cl.user_session.get("user")
    chat_profile = cl.user_session.get("chat_profile", "Unknown")
    
    # Log that the session has ended.
    if user:
        logger.info(f"Chat session ended for user {user.identifier} using {chat_profile}")
    else:
        logger.info("Chat session ended for unauthenticated user")

# This is an "action callback". It runs when a user clicks on a button with the name "add_to_evaluation".
@cl.action_callback("add_to_evaluation")
async def on_add_to_evaluation(action):
    """Handle adding conversation to evaluation CSV."""
    user = cl.user_session.get("user")
    # Get the last conversation turn from the session.
    user_input = cl.user_session.get("last_user_input", "")
    ai_output = cl.user_session.get("last_ai_output", "")
    agent_name = cl.user_session.get("last_agent_name", "unknown")
    
    user_id = user.identifier if user else "unknown"
    
    # Save the data to the CSV file and notify the user.
    if save_to_evaluation(user_input, ai_output, user_id, agent_name):
        await cl.Message(content="✅ **Added to evaluation dataset!** Your conversation has been saved to `evaluation.csv` for model improvement.", author="System").send()
    else:
        await cl.Message(content="❌ **Failed to save.** Please try again.", author="System").send()

# This is another action callback, this time for the "copy_response" button.
@cl.action_callback("copy_response")
async def on_copy_response(action):
    """Handle copying AI response to clipboard."""
    ai_output = cl.user_session.get("last_ai_output", "")
    
    if ai_output:
        # To make copying easy, we put the text inside a special "Text" element.
        text_element = cl.Text(name="copy_text", content=ai_output, display="inline")
        await cl.Message(content="📋 **Response copied below** - Select and copy the text:", author="System", elements=[text_element]).send()
    else:
        await cl.Message(content="❌ No response to copy.", author="System").send()

# This runs if the user clicks the "Stop" button in the UI.
@cl.on_stop
async def on_stop():
    """Handle when user stops the agent with user context."""
    agent_info = cl.user_session.get("agent_info", {"icon": "🤖"})
    user = cl.user_session.get("user")
    
    if user:
        logger.info(f"Task stopped by user {user.identifier}")
    
    await cl.Message(content=f"⏹️ **Task stopped by user.** {agent_info['icon']} Ready for your next question!", author="Assistant").send()

# This runs if the user changes settings in the UI.
@cl.on_settings_update
async def setup_agent_settings(settings):
    """Handle settings updates for enhanced user control with user context."""
    user = cl.user_session.get("user")
    cl.user_session.set("settings", settings)
    
    agent = cl.user_session.get("agent")
    if agent and settings:
        if user:
            logger.info(f"Settings updated for user {user.identifier}: {settings}")
        else:
            logger.info(f"Settings updated: {settings}")

# This part of the code allows the script to be run directly from the command line.
if __name__ == "__main__":
    import subprocess
    import sys
    
    # This checks if the user typed "run" after the script name (e.g., "python app.py run").
    # If so, it starts the Chainlit server.
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        subprocess.run(["chainlit", "run", __file__, "-w"])
    else:
        # Otherwise, it just prints a helpful message.
        print("To run the app, use: chainlit run app.py")