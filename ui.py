import chainlit as cl
from agents.storybrand import brandscript_architect
from agents.seo import seo_specialist
import asyncio
from typing import Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Store agent instances
AGENTS = {
    "BrandScript Architect": brandscript_architect,
    "SEO Specialist": seo_specialist
}

@cl.set_chat_profiles
async def chat_profile():
    """Set up different agent profiles for the UI."""
    return [
        cl.ChatProfile(
            name="BrandScript Architect",
            markdown_description="Expert StoryBrand (SB7) Guide for creating compelling Master BrandScripts",
            icon="💎"
        ),
        cl.ChatProfile(
            name="SEO Specialist", 
            markdown_description="SEO optimization expert for improving search rankings",
            icon="🔍"
        )
    ]

@cl.set_starters
async def set_starters():
    """Set up starter prompts to help users begin conversations."""
    return [
        cl.Starter(
            label="Create BrandScript",
            message="Help me create a comprehensive BrandScript for my business",
            icon="📝"
        ),
        cl.Starter(
            label="Analyze SEO Strategy",
            message="Analyze and improve my website's SEO strategy",
            icon="📊"
        ),
        cl.Starter(
            label="Brand Messaging",
            message="Develop clear and compelling brand messaging",
            icon="💬"
        ),
        cl.Starter(
            label="Content Strategy",
            message="Create a content strategy for my business",
            icon="📖"
        )
    ]

@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session with the selected agent."""
    chat_profile = cl.user_session.get("chat_profile", "BrandScript Architect")
    
    # Get the selected agent
    agent = AGENTS.get(chat_profile)
    if not agent:
        agent = brandscript_architect  # Default fallback
    
    # Store agent in session
    cl.user_session.set("agent", agent)
    cl.user_session.set("chat_profile", chat_profile)
    
    # Send welcome message
    welcome_msg = f"""
# Welcome to BrainSpark Digital! 🧠✨

You're now chatting with the **{chat_profile}**. 

I'm here to help you create compelling brand stories and optimize your digital presence. Ask me anything about:
- StoryBrand Framework (SB7)
- Brand messaging and positioning  
- Content strategy
- SEO optimization
- Digital marketing

How can I help you today?
    """
    
    await cl.Message(content=welcome_msg).send()

@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages and stream agent responses with status updates."""
    
    agent = cl.user_session.get("agent")
    chat_profile = cl.user_session.get("chat_profile", "BrandScript Architect")
    
    if not agent:
        await cl.Message(content="❌ Agent not initialized. Please start a new chat.").send()
        return
    
    # Create main response message
    response_msg = cl.Message(content="")
    
    # Create status step to show what the agent is doing
    async with cl.Step(name="🤖 Agent Processing", type="llm") as status_step:
        status_step.input = message.content
        
        try:
            # Set initial status
            await status_step.stream_token("🔄 **Initializing agent...**\n")
            await asyncio.sleep(0.5)
            
            # Show search activities based on agent capabilities
            await status_step.stream_token("🚀 **Starting analysis...**\n")
            await asyncio.sleep(0.5)
            
            await status_step.stream_token("📚 **Searching knowledge base...**\n")
            await asyncio.sleep(1)
            
            await status_step.stream_token("🤔 **Processing your request...**\n")
            await asyncio.sleep(0.5)
            
            # Use the simple agent.run() method that works
            response = agent.run(message.content)
            
            await status_step.stream_token("✅ **Analysis complete!**\n")
            
            # Handle the response content
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            # Stream the response content in chunks for better UX
            chunk_size = 100
            for i in range(0, len(content), chunk_size):
                chunk = content[i:i+chunk_size]
                await response_msg.stream_token(chunk)
                await asyncio.sleep(0.05)  # Small delay for streaming effect
                    
        except Exception as e:
            logger.error(f"Error during agent execution: {e}")
            await status_step.stream_token(f"❌ **Error: {str(e)}**\n")
            await cl.Message(content="❌ An error occurred while processing your request. Please try again.").send()
            return
        
        # Finalize the status step
        status_step.output = f"✅ Successfully processed query with {chat_profile}"
    
    # Send the final response
    await response_msg.send()

@cl.on_chat_end
def on_chat_end():
    """Clean up when chat ends."""
    logger.info("Chat session ended")
    cl.user_session.clear()

@cl.on_stop
async def on_stop():
    """Handle when user stops the agent."""
    await cl.Message(content="⏹️ **Task stopped by user.** Feel free to ask me anything else!").send()

if __name__ == "__main__":
    # This allows running the UI directly
    import os
    import subprocess
    
    # Run chainlit
    subprocess.run(["chainlit", "run", __file__, "-w"]) 