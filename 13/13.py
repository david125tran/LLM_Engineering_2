# ---------------------------------- Libraries ----------------------------------
import autogen
from autogen import GroupChat, GroupChatManager
from dotenv import load_dotenv
import google.generativeai as genai
import gradio as gr
from IPython.display import display, Markdown
import json
from openai import OpenAI as OpenAIClient
import os
from pydantic import BaseModel
import random
from typing import List, Dict, Union, Optional, Any
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")



# ---------------------------------- Variables ----------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)



# ---------------------------------- Functions ----------------------------------
def print_banner(text: str) -> None:
    """
    Create a banner for easier visualiziation of what's going on 

    Ex.
    Input:  "Device"
    Output:
            *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
            *                          Device                           *
            *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
    """
    banner_len = len(text)
    mid = 49 - banner_len // 2

    print("\n\n\n")
    print("*" + "-*" * 50)
    if (banner_len % 2 != 0):
        print("*"  + " " * mid + text + " " * mid + "*")
    else:
        print("*"  + " " * mid + text + " " + " " * mid + "*")
    print("*" + "-*" * 50)



def save_chat_history_to_md(chat_result, part_number: int, script_dir: str) -> str:
    """
    Save an AutoGen chat_result's conversation history to a markdown file.
    
    Args:
        chat_result: The chat result object returned by agent.initiate_chat().
        part_number (int): The part number (e.g. 1 or 2).
        script_dir (str): The directory of the running script (used as base path).
    
    Returns:
        str: The path of the saved markdown file.
    """
    # Build file path like "<script_dir>\\1_conversation_history.md"
    file_path = os.path.join(script_dir, f"{part_number}_conversation_history.md")

    # Build markdown text
    md_lines = [f"# Part {part_number} Conversation History\n"]
    for msg in chat_result.chat_history:
        speaker = msg["name"]
        content = msg["content"].strip()
        md_lines.append(f"## {speaker}\n")
        md_lines.append(f"{content}\n")
        md_lines.append("---\n")

    # Write to markdown file (UTF-8 for safety)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"✅ Chat part {part_number} history saved to: {file_path}")
    return file_path



# ---------------------------------- Load Environment Variables ----------------------------------
print_banner("Load Environment Variables")

# Load environment variables and create OpenAI client
load_dotenv(dotenv_path=parent_dir + r"\.env", override=True)

openai_api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

# View the first few characters in the key
print(f"OpenAI API Key: {openai_api_key[:15]}...")
print(f"Google API Key: {google_api_key[:15]}...")

# Configure APIs
openai_client = OpenAIClient(api_key = openai_api_key)
genai.configure(api_key = google_api_key)

# Initialize gemini model
gemini_model = genai.GenerativeModel("gemini-2.0-flash")



# ---------------------------------- Part 1: AI Agents Config (Single-Model: OpenAI) ----------------------------------
print_banner("Part 1: AI Agents Config (Single-Model: OpenAI)")

# Configuration for OpenAI Agent
config_list_openai = [
    {
        "model": "gpt-4o-mini",
        "api_key": openai_api_key,
    }
]

llm_config_openai = {
    "config_list": config_list_openai,
    "temperature": 0.7,  # Use a slightly higher temp for creative marketing ideas
    "timeout": 120,
}

print("AI Agents Configured.")



# ---------------------------------- Part 1: Configure AI Agents Prompts (Single-Model: OpenAI) ----------------------------------
print_banner("Part 1: Configure AI Agents Prompts (Single-Model: OpenAI)")

# Chief Marketing Officer (CMO) Prompt
cmo_prompt = """You are the Chief Marketing Officer (CMO) of a new shoe brand (sustainable).
Provide high-level strategy, define target audiences, and guide the Marketer. Focus on the 
big picture. Be concise.  Keep your responses to less than one paragraph."""

# Brand Marketer Prompt
brand_marketer_prompt = """You are the Brand Marketer for the shoe brand. Brainstorm creative, 
specific campaign ideas (digital, content, experiences). Focus on tactics and details. Suggest 
KPIs for your ideas.  Keep your responses to less than one paragraph."""

# System Media Strategist Prompt
system_media_strategist_prompt = """You are the Media Strategist for the shoe brand. Analyze 
campaign ideas from the Brand Marketer and recommend optimal media channels (social, search, 
display, influencer) to reach target audiences effectively. Provide budget allocation suggestions 
and expected ROI for each channel.  Keep your responses to less than one paragraph."""

print("OpenAI Agent System Prompts Configured.")



# ---------------------------------- Part 1: Initialize AI Agents (Single-Model: OpenAI) ----------------------------------
print_banner("Part 1: Initialize AI Agents (Single-Model: OpenAI)")

# The 'human_input_mode' parameter is set to "NEVER" to ensure agents interact autonomously.

# Create Chief Marketing Officer (CMO) Agent with OpenAI
cmo_agent_openai = autogen.ConversableAgent(
    name = "Chief_Marketing_Officer_OpenAI",
    system_message = cmo_prompt,
    llm_config = llm_config_openai,  # Assign the OpenAI config
    human_input_mode = "NEVER")

print(f"Agent '{cmo_agent_openai.name}' created (using OpenAI).")

# Create Brand Marketer Agent with OpenAI
brand_marketer_agent_openai = autogen.ConversableAgent(
    name = "Brand_Marketer_OpenAI",
    system_message = brand_marketer_prompt,
    llm_config = llm_config_openai,  # Assign the same OpenAI config
    human_input_mode = "NEVER")

print(f"Agent '{brand_marketer_agent_openai.name}' created (using OpenAI).")

# Create System Media Strategist Agent with OpenAI
media_strategist_agent_openai = autogen.ConversableAgent(
    name = "Media_Strategist_OpenAI",
    system_message = system_media_strategist_prompt,
    llm_config = llm_config_openai,  # Assign the same OpenAI config
    human_input_mode = "NEVER")
print(f"Agent '{media_strategist_agent_openai.name}' created (using OpenAI).")

print("AI Agents Initialized.")



# ---------------------------------- Part 1: Multi-Agent Conversation (Single-Model: OpenAI) ----------------------------------
print_banner(" Part 1: Multi-Agent Conversation (Single-Model: OpenAI)")
print("Invoking a multi-agent conversation between CMO and Brand Marketer (OpenAI Only).")

# Cheif Marketing Officer initiates the chat with Brand Marketer
initial_task_message = """
Context: We're launching a new sustainable shoe line and need campaign ideas
Instruction: Brainstorm a campaign concept with specific elements
Input: Our sustainable, futuristic shoe brand needs marketing direction
Output: A concise campaign concept with the following structure:
Brand Marketer, let's brainstorm initial campaign ideas for our new sustainable shoe line.
Give me a distinct campaign concept. Outline: core idea, target audience, primary channels, 
and 1-2 KPIs. Keep it concise. Try to arrive at a final answer in 2-3 turns.
"""

print("--- Starting Agent Conversation (OpenAI Only) ---")
print("Chief Marketing Officer (OpenAI) initiating chat with Brand Marketer (OpenAI). Max Turns = 4")
print("--------------------------------------------------")

# Chief Marketing Officer (OpenAI) initiates the chat with Brand Marketer (OpenAI)
chat_result_openai_only = cmo_agent_openai.initiate_chat(
    # Define the recipient as the Brand Marketer Agent
    recipient = brand_marketer_agent_openai, 
    # Invoke the initial task message
    message = initial_task_message, 
    # Set maximum number of turns for the conversation
    max_turns = 4
)

print("--------------------------------------------------")
print("--- Conversation Ended (OpenAI Only) ---")


def print_chat_history(chat_result):
    """Any chat result object has a chat_history attribute that contains the conversation history.
    This function prints the conversation history in a readable format.
    """
    for i in chat_result.chat_history:  
        print(i['name'])
        print("_"*100)
        print(i['content'])
        print("_"*100)

# Print the chat history
print_chat_history(chat_result_openai_only)

# Save the chat history to a markdown file
save_chat_history_to_md(chat_result_openai_only, part_number=1, script_dir=script_dir)

# ---------------------------------- Part 2: AI Agents Config (Multi-Model: Gemini & OpenAI) ----------------------------------
print_banner("Part 2: AI Agents Config (Multi-Model: Gemini & OpenAI)")

# AutoGen - An open-source framework to build multi-agent applications with LLMs.
# It helps you orchestrate conversations between multiple AI agents powered by different LLMs.

# Configuration for Gemini Agent
config_list_gemini = [
    {
        "model": "gemini-2.0-flash",   
        "api_key": google_api_key,
        "api_type": "google",           # Specify the API type for Autogen's Google integration
    }
]

# Gemini LLM Config
llm_config_gemini = {
    "config_list": config_list_gemini,
    "temperature": 0.6,                 # Maybe slightly less randomness for strategic Chief Marketing Officer
    "timeout": 120,                     # Set a timeout of 120 seconds
}

print("Multi-Model Agents Configured.")



# ---------------------------------- Part 2: Configure AI Agents Prompts (Multi-Model: Gemini & OpenAI) ----------------------------------
print_banner("Part 2: Configure AI Agents Prompts (Multi-Model: Gemini & OpenAI)")

# (Reusing the same prompts from Part 1 for consistency)


# ---------------------------------- Part 2: Initialize AI Agents (Multi-Model: Gemini & OpenAI) ----------------------------------
print_banner("Part 2: Initialize AI Agents (Multi-Model: Gemini & OpenAI)")

# Create the Chief Marketing Officer Agent using Google Gemini
cmo_agent_gemini = autogen.ConversableAgent(
    name = "Chief_Marketing_Officer_Gemini",
    system_message = cmo_prompt,
    llm_config = llm_config_gemini,  # Assign the Gemini config!
    human_input_mode = "NEVER")

# Create the Brand Marketer Agent using OpenAI GPT 
brand_marketer_agent_openai_mixed = autogen.ConversableAgent(
    name = "Brand_Marketer_OpenAI", 
    system_message = brand_marketer_prompt,
    llm_config = llm_config_openai,  # Assign the OpenAI config!
    human_input_mode = "NEVER")

# Create the System Media Strategist Agent using OpenAI GPT
system_media_strategist_agent_openai_mixed = autogen.ConversableAgent(
    name = "Media_Strategist_OpenAI",
    system_message = system_media_strategist_prompt,
    llm_config = llm_config_openai,  # Assign the OpenAI config!
    human_input_mode = "NEVER")

# Print confirmation of agent creation
print(f"Agent '{cmo_agent_gemini.name}' created (using Google Gemini).")
print(f"Agent '{brand_marketer_agent_openai_mixed.name}' created (using OpenAI).")
print(f"Agent '{system_media_strategist_agent_openai_mixed.name}' created (using OpenAI).")



# ---------------------------------- Part 2: Multi-Agent Conversation (Multi-Model: Gemini & OpenAI) ----------------------------------
print_banner("Part 2: Multi-Agent Conversation (Multi-Model: Gemini & OpenAI)")
print("Invoking a multi-agent conversation between CMO (Gemini) and Brand Marketer (OpenAI).")

# Cheif Marketing Officer initiates the chat with Brand Marketer
initial_task_message = """
Context: We're launching a new sustainable shoe line and need campaign ideas
Instruction: Brainstorm a campaign concept with specific elements
Input: Our sustainable, futuristic shoe brand needs marketing direction
Output: A concise campaign concept with the following structure:
Brand Marketer, let's brainstorm initial campaign ideas for our new sustainable shoe line.
Give me a distinct campaign concept. Outline: core idea, target audience, primary channels, 
and 1-2 KPIs. Keep it concise. Try to arrive at a final answer in 2-3 turns.
"""

print("--- Starting Agent Conversation (Multi-Model: Gemini + OpenAI) ---")
print("Chief Marketing Officer (Gemini) initiating chat with Brand Marketer (OpenAI). Max Turns = 4")
print("------------------------------------------------------------------")

# Chief Marketing Officer (Gemini) initiates the chat with the Brand Marketer (OpenAI)
chat_result_multi_model = cmo_agent_gemini.initiate_chat(
    # Define the recipient as the Brand Marketer Agent
    recipient = brand_marketer_agent_openai_mixed,
    # Invoke the initial task message
    message = initial_task_message,
    # Set maximum number of turns for the conversation
    max_turns = 4)

print("------------------------------------------------------------------")
print("--- Conversation Ended (Multi-Model) ---")

# Print the chat history
print_chat_history(chat_result_multi_model)

# Save the chat history to a markdown file
save_chat_history_to_md(chat_result_multi_model, part_number=2, script_dir=script_dir)



# ---------------------------------- Part 3: Adding Human Guidance (User Proxy Agent) & Leveraging Group Chat ----------------------------------
print_banner("Part 3: Adding Human Guidance (User Proxy Agent) & Leveraging Group Chat")
print("Creating a User Proxy Agent to represent human input in the multi-agent conversation.")

# Create the User Proxy Agent (Represents You)
user_proxy_agent = autogen.UserProxyAgent(
    # Name of the agent
    name = "Human_User_Proxy",
    # Prompt user for input until 'exit'
    human_input_mode = "ALWAYS",
    # Limit consecutive auto-replies to avoid long monologues
    max_consecutive_auto_reply = 1,
    # Create exit condition for human user
    is_termination_msg = lambda x: x.get("content", "").rstrip().lower() in ["exit", "quit", "terminate"],
    # Configure to not execute code
    code_execution_config = False,
    # Provide context for the human user
    system_message = "You are the human user interacting with a multi-model AI team (Gemini CMO, OpenAI Marketer). Guide the brainstorm. Type 'exit' to end.",
)

print(f"Agent '{user_proxy_agent.name}' created for HIL with multi-model team.")

# **Initiating HIL Chat with the Multi-Model Team**
print("--- Starting Human-in-the-Loop (HIL) Conversation (Multi-Model) ---")
print("You will interact with Gemini CMO and OpenAI Marketer. Type 'exit' to end.")
print("---------------------------------------------------------------------")

# Reset agents for a clean new session
cmo_agent_gemini.reset()                    # Reset Gemini CMO
brand_marketer_agent_openai_mixed.reset()   # Reset OpenAI Marketer
user_proxy_agent.reset()                    # Reset User Proxy Agent

# Create a GroupChat with multiple agents
# This sets up a collaborative chat environment where multiple agents can interact
groupchat = GroupChat(
    agents = [user_proxy_agent, cmo_agent_gemini, brand_marketer_agent_openai],  # List of agents participating in the group chat
    messages = [ ],  # Initialize with empty message history
    max_round = 20,  # Optional: Limits how many conversation rounds can occur before terminating
)

# Create a manager for the group chat
# The GroupChatManager orchestrates the conversation flow between agents
# It determines which agent should speak next and handles the overall conversation logic
group_manager = GroupChatManager(groupchat = groupchat, llm_config = llm_config_openai)  # Uses OpenAI's LLM to manage the conversation

# User Proxy initiates the chat - Let's give a new task
group_chat_result = group_manager.initiate_chat(
    recipient = user_proxy_agent,  # Start by talking to the Gemini CMO
    message = """Hello team!!""",
)

print("---------------------------------------------------------------------")
print("--- Conversation Ended (Human terminated or Max Turns) ---")

# Print the chat history
print_chat_history(group_chat_result)
# Save the chat history to a markdown file
save_chat_history_to_md(group_chat_result, part_number=3, script_dir=script_dir)

