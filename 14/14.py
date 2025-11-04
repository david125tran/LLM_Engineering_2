# ---------------------------------- Libraries ----------------------------------
from amadeus import Client, ResponseError
from datetime import date
from dotenv import load_dotenv
import google.generativeai as genai
import gradio as gr
from graphviz import Source
from IPython.display import display, Markdown, Image
import json
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from openai import OpenAI as OpenAIClient
import operator
import os
from pydantic import BaseModel
import random
from typing import TypedDict, Annotated, Sequence, List, Tuple, Optional, Any, Union, Literal,  Tuple
import uuid
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")



# ---------------------------------- Variables ----------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

# Define which parts to run
run_part_1 = False
run_part_2 = False
run_part_3 = True



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



# ---------------------------------- Load Environment Variables ----------------------------------
print_banner("Load Environment Variables")

# Load environment variables and create OpenAI client
load_dotenv(dotenv_path=parent_dir + r"\.env", override=True)

openai_api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
amadeus_api_key = os.environ["AMADEUS_CLIENT_ID"]
amadeus_api_secret = os.environ["AMADEUS_CLIENT_SECRET"]

# View the first few characters in the key
print(f"OpenAI API Key: {openai_api_key[:15]}...")
print(f"Google API Key: {google_api_key[:15]}...")
print(f"Tavily API Key: {tavily_api_key[:15]}...")
print(f"Amadeus Key starts with: {amadeus_api_key[:5]}...")
print(f"Amadeus Secret starts with: {amadeus_api_secret[:5]}...")

# Configure APIs
openai_client = OpenAIClient(api_key = openai_api_key)
genai.configure(api_key = google_api_key)

# Initialize gemini model
gemini_model = genai.GenerativeModel("gemini-2.0-flash")


if (run_part_1):
    # ---------------------------------- Part 1: Workflow Outlined ----------------------------------
    # User Script Execution Flow
    # │
    # ├── 1️⃣ Define State
    # │      class AgentState(TypedDict):
    # │          input_text: str
    # │          summary: str
    # │
    # ├── 2️⃣ Define Node Function
    # │      def summarize_step(state: AgentState) -> AgentState:
    # │          llm = ChatOpenAI(model="gpt-3.5-turbo")
    # │          prompt = f"Please summarize: {state['input_text']}"
    # │          result = llm.invoke([prompt])
    # │          return {"input_text": state["input_text"],
    # │                  "summary": result.content}
    # │
    # ├── 3️⃣ Build Graph
    # │      workflow = StateGraph(AgentState)
    # │      workflow.add_node("summarize", summarize_step)
    # │      workflow.add_edge("summarize", END)
    # │      workflow.set_entry_point("summarize")
    # │
    # │      ┌──────────────┐
    # │      │   START      │
    # │      └──────┬───────┘
    # │             │
    # │             ▼
    # │      ┌─────────────────────────────┐
    # │      │   Node: summarize_step()    │
    # │      └──────────────┬──────────────┘
    # │                     │
    # │                     ▼
    # │                ┌───────────┐
    # │                │   END     │
    # │                └───────────┘
    # │
    # ├── 4️⃣ Compile Graph
    # │      graph = workflow.compile()
    # │
    # ├── 5️⃣ Initialize State
    # │      initial_state = {
    # │          "input_text": <your text>,
    # │          "summary": ""
    # │      }
    # │
    # ├── 6️⃣ Invoke Graph (run node)
    # │      result = graph.invoke(initial_state)
    # │
    # └── 7️⃣ Retrieve Output
    #        summary = result["summary"]
    #        print(summary)



    #       ┌──────────────┐
    #       │   START      │   (entry point)
    #       └──────┬───────┘
    #              │
    #              ▼
    #       ┌─────────────────────────────────────────────────────────────┐
    #       │                 Node: "summarize"                           │
    #       │-------------------------------------------------------------│
    #       │ Function: summarize_step(state: AgentState)                 │
    #       │                                                             │
    #       │ Inputs (from state):                                        │
    #       │   • input_text : str                                        │
    #       │   • summary    : str (ignored on input)                     │
    #       │                                                             │
    #       │ Action:                                                     │
    #       │   • Build prompt → call ChatOpenAI("gpt-3.5-turbo").invoke  │
    #       │   • Produce a one-sentence summary                          │
    #       │                                                             │
    #       │ Outputs (to state):                                         │
    #       │   • input_text : (passes through unchanged)                 │
    #       │   • summary    : result.content (LLM-generated)             │
    #       └───────────────┬─────────────────────────────────────────────┘
    #                       │
    #                       ▼
    #                 ┌───────────┐
    #                 │   END     │
    #                 └───────────┘



    # ---------------------------------- Part 1: Define State (Small Workflow) ----------------------------------
    print_banner("Part 1: Define State (Small Workflow)")
    # Define a state that includes: 
    # 1. input_text: The original text to be summarized
    # 2. summary: The generated summary of the input text

    # The state is like a container that stores and passes data between different parts of our workflow
    # Each node receives and returns a state object, and the State can include messages, variables, memory, etc.
    class AgentState(TypedDict):
        """
        State container for the text summarization workflow.
        Holds the original input text and the generated summary.
        """
        input_text: str
        summary: str

    print("Defined AgentState")


    # ---------------------------------- Part 1: Define Node Function(s) (Small Workflow) ----------------------------------
    print_banner("Part 1: Define Node Function(s) (Small Workflow)")

    # Define the key node, which represents the functions that perform specific tasks in your graph
    # They receive the current state and return a modified state
    # Note that a node can be simple functions, LLM calls, or complex agents

    def summarize_step(state: AgentState) -> AgentState:
        """
        Create a concise summary of the input text by:
        1. Receiving the current state with the input text.
        2. Instantiating an LLM to generate the summary.
        3. Calls the model via `invoke` with a prompt to summarize the text.
        4. Returns the original input_text plus summary = result.content
        """
        
        # Initialize the OpenAI model and define the prompt
        llm = ChatOpenAI(model = "gpt-3.5-turbo")
        prompt = f"Please summarize the following text in one sentence that captures the main points: {state['input_text']}"
        
        # Get the summary directly from the model
        result = llm.invoke([prompt])
        
        # Update the state with our summary
        return {
            "input_text": state["input_text"],  # Keep the original text
            "summary": result.content  # Add the summary
        }

    print("Defined summarize_step() node function.")



    # ---------------------------------- Part 1: Build Graph (Small Workflow) ----------------------------------
    print_banner("Part 1: Build Graph (Small Workflow)")

    # Define the StateGraph, which is the fundamental building block of LangGraph
    # It manages the flow of information between different components
    # It maintains state throughout the execution of your workflow

    # Define the stategraph with the "AgentState" class
    workflow = StateGraph(AgentState)

    # Add a node, which is the summarize function
    workflow.add_node("summarize", summarize_step)

    # Define Edges, which define how data flows between nodes
    workflow.add_edge("summarize", END) 
    workflow.set_entry_point("summarize")
    workflow.compile()

    print("Workflow built")


    # ---------------------------------- Part 1: Compile Graph (Small Workflow) ----------------------------------
    print_banner("Part 1: Compile Graph (Small Workflow)")    

    # Compile and execute the graph with sample text
    # After defining your graph, you need to compile it to create an executable workflow
    # Invoke it with an initial state to run the entire process
        
    # Compile the graph
    graph = workflow.compile() 

    print("Workflow built")



    # ---------------------------------- Part 1: Initialize State (Small Workflow) ---------------------------------
    print("Part 1: Initialize State (Small Workflow)")

    # Example text to summarize
    sample_text = """
        Electric cars work by using electricity stored in a battery pack to power an electric motor, which drives the wheels. 
        Unlike gasoline-powered vehicles that rely on internal combustion engines, electric vehicles (EVS) use electric motors 
        that are more efficient and produce zero emissions during operation. When you press the accelerator, the battery sends 
        power to the motor, which instantly provides torque to move the car. The battery is recharged by plugging the car into 
        an external power source, such as a home charger or public charging station. Some electric cars also feature regenerative 
        braking, which captures energy during braking and feeds it back into the battery to improve efficiency.
        """


    # Set up the initial state with the input text
    initial_state = {
            "input_text": sample_text,
            "summary": ""}

    print("State initialized...")


    # ---------------------------------- Part 1: Invoke the Graph (Small Workflow) ---------------------------------
    print_banner("Part 1: Invoke the Graph (Small Workflow)")

    # Run the graph
    result = graph.invoke(initial_state)

    print("Graph invoked...")



    # ---------------------------------- Part 1: Retreive Output (Small Workflow) ---------------------------------
    print_banner("Part 1: Retreive Output (Small Workflow)")
    # Get the summary from the result
    summary = result["summary"]
        
    # Print the result
    print(summary)



if (run_part_2):
    # ---------------------------------- Part 2: Define State (Full Agentic Workflow) ----------------------------------
    print_banner("Part 2: Define State (Full Agentic Workflow)")

    class AgentState(TypedDict):
        input_text: str
        summary: str
        translated_summary: str
        sentiment: str

    print("Defined AgentState")



    # ---------------------------------- Part 2: Define Node Function(s) (Full Agentic Workflow) ----------------------------------
    print_banner("Part 2: Define Node Function(s) (Full Agentic Workflow)")

    def summarize_step(state: AgentState) -> AgentState:
        """
        Create a concise summary of the input text by:
        1. Receiving the current state with the input text.
        2. Instantiating an LLM to generate the summary.
        3. Calls the model via `invoke` with a prompt to summarize the text.
        4. Returns the original input_text plus summary = result.content
        """
        
        # Initialize the OpenAI model and define the prompt
        llm = ChatOpenAI(model = "gpt-3.5-turbo")
        prompt = f"Please summarize the following text in one sentence that captures the main points: {state['input_text']}"
        
        # Get the summary directly from the model
        result = llm.invoke([prompt])
        
        # Update the state with our summary
        return {
            # This '**state' syntax, means unpack the entire Python dictionary.  Take everything from state, like "input_text", "summary", etc., and then overwrite it with the new value
            **state,                    
            "summary": result.content  # Add the summary
        }

    print("Defined summarize_step() node function.")

    def translate_step(state: AgentState) -> AgentState:
        """Translate text"""
        
        # Initialize the OpenAI model and define the prompt
        llm = ChatOpenAI(model = "gpt-4o-mini")
        prompt = f"Please translate from english to spanish: {state['summary']}"
        
        # Get the summary directly from the model
        result = llm.invoke([prompt])
        
        # Update the state with our summary
        return {
            # This '**state' syntax, means unpack the entire Python dictionary.  Take everything from state, like "input_text", "summary", etc., and then overwrite it with the new value
            **state,                    
            "translated_summary": result.content
        }

    print("Defined translate_step() node function.")

    def sentiment_step(state: AgentState) -> AgentState:
        """
        Generate the sentiment of a given text as either 'positive', 'negative', or 'neutral'.
        """

        # Initialize the OpenAI model and define the prompt
        llm = ChatOpenAI(model = "gpt-4o-mini")
        prompt = f"""Please give the sentiment of the given text and classify as,
        'positive', 'negative', or 'neutral'.  Only respond with one of those three
        things.  If you aren't sure, respond with 'neutral'.
        
        Text to classify:
        {state['input_text']}"""
        
        # Get the summary directly from the model
        result = llm.invoke([prompt])
        
        # Update the state with our summary
        return {
            # This '**state' syntax, means unpack the entire Python dictionary.  Take everything from state, like "input_text", "summary", etc., and then overwrite it with the new value
            **state,                    
            "sentiment": result.content.strip()
        }

    print("Defined sentiment_step() node function.")

    # ---------------------------------- Part 2: Build Graph (Full Agentic Workflow) ----------------------------------
    print_banner("Part 2: Build Graph (Full Agentic Workflow)")

    # Define a stategraph with the "AgentState"
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("summarize",          summarize_step)
    workflow.add_node("translate",          translate_step)
    workflow.add_node("analyze_sentiment",  sentiment_step)

    # Define edges
    workflow.add_edge("summarize", "translate") 
    workflow.add_edge("translate", "analyze_sentiment")

    # Set the entry point
    workflow.set_entry_point("summarize")

    print("Workflow built")



    # ---------------------------------- Part 2: Compile Graph (Full Agentic Workflow) ----------------------------------
    print_banner("Part 2: Compile Graph (Full Agentic Workflow)")

    workflow.compile()

    print("Workflow built")



    # ---------------------------------- Part 2: Initialize State (Full Agentic Workflow) ---------------------------------
    print_banner("Part 2: Initialize State (Full Agentic Workflow)")

    # Example text to summarize
    sample_text = """
        Electric cars are awesome! I love them so much! They work by using electricity stored in a battery pack to power an electric motor, which drives the wheels. 
        Unlike gasoline-powered vehicles that rely on internal combustion engines, electric vehicles (EVS) use electric motors that are more efficient and produce zero emissions during operation. 
        When you press the accelerator, the battery sends power to the motor, which instantly provides torque to move the car. 
        The battery is recharged by plugging the car into an external power source, such as a home charger or public charging station. 
        Some electric cars also feature regenerative braking, which captures energy during braking and feeds it back into the battery to improve efficiency.
        """
        
    # Let's compile the graph
    graph = workflow.compile() 

    # Set up the initial state with the input text
    initial_state = {
            "input_text": sample_text,
            "summary":              "",
            "translated_summary":   "",
            "sentiment":            ""
            }

    print("State initialized...")



    # ---------------------------------- Part 2: Invoke the Graph (Full Agentic Workflow) ---------------------------------
    print_banner("Part 2: Invoke the Graph (Full Agentic Workflow)")

    # Run the graph
    result = graph.invoke(initial_state)

    print("Graph invoked...")



    # ---------------------------------- Part 2: Retreive Output (Full Agentic Workflow) ---------------------------------
    print_banner("Part 2: Retreive Output (Full Agentic Workflow)")

    # Get the summary from the result
    summary = result["summary"]
    translation = result["translated_summary"]
    sentiment = result["sentiment"]

    # Print the results with clear labels and spacing
    print("=== Generated Summary ===")
    print(summary)

    print("\n=== Translated Summary ===")
    print(translation)

    print("\n=== Sentiment Classification ===")
    print(sentiment)













if (run_part_3):
    # ---------------------------------- Part 3: Setup (Full Agentic Workflow w/Tooling) ----------------------------------
    print_banner("Part 3: Setup (Full Agentic Workflow w/Tooling)")
    # Configure the OpenAI Client (using LangChain's wrapper)
    # We use GPT-4o as it's generally better with tool calling
    llm = ChatOpenAI(model = "gpt-4.1-mini", temperature = 0, streaming = True)
    print("LangChain OpenAI Chat Model configured.")



    # ---------------------------------- Part 3: Define State (Full Agentic Workflow w/Tooling) ----------------------------------
    print_banner("Part 3: Define State (Full Agentic Workflow w/Tooling)")

    class AgentState(TypedDict):
        # AgentState is the name of the dictionary (used to represent the agent's state in the workflow).
        # It has one key: "messages", which holds a list of messages (e.g., from the user, model, or tools).
        # BaseMessage is the type used to represent each message in that list.
        # operator.add tells LangGraph to append new messages to the list during execution.
        messages: Annotated[Sequence[BaseMessage], operator.add]

    print("Defined AgentState")



    # ---------------------------------- Part 3: Define Node Function(s) (Full Agentic Workflow w/Tooling) ----------------------------------
    print_banner("Part 3: Define Node Function(s) (Full Agentic Workflow w/Tooling)")

    # Set up tavily search tool to equip LLM with tooling
    tavily_search_tool = TavilySearchResults(max_results = 3)

    # List of tools for this step
    tools_list_single = [tavily_search_tool]

    def make_call_model_with_tools(tools: list):
        """
        This function binds tool(s) to the LLM.  This format is useful when you want to
        reuse the same model logic with different tools.  
        """
        def call_model_with_tools(state: AgentState):
            """
            This is the execution function.  It knows how to use the current state (conversation
            history) and actually runs the model with the tools that were set up by the outer
            function.
            """

            print("DEBUG: Entering call_model_with_tools node")
            messages = state["messages"]
            
            # Binds the tools to the language model 
            model_with_tools = llm.bind_tools(tools)

            # Feeds the conversation history (messages) into the model
            response = model_with_tools.invoke(messages)

            # Return the model response as a new message
            return {"messages": [response]}

        return call_model_with_tools
    
    print("Defined make_call_model_with_tools() node function.")


    def should_continue(state: AgentState) -> Literal["action", "__end__"]:
        """
        This function checks the most recent message in the state and decides whether to route the
        'action' node (ToolNode) or end.  This function is used to control the flow of your agent,
        it's like a traffic signal deciding where to send the agent next.  
        """

        print("DEBUG: Entering should_continue node")

        # Look at the last message in the state 
        last_message = state["messages"][-1]
        
        # Check if the last message is an AIMessage with tool_calls and we need to do something
        if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls") and last_message.tool_calls:
            print("DEBUG: Decision: continue (route to action)")
            # Route the action
            return "action"  # Route to the node named "action"
        else:
            print("DEBUG: Decision: end (route to END)")
            return END  # Special value indicating the end of the graph

    print("Defined should_continue() node function.")


    def build_graph_one_tool(tools_list):
        # ---------------------------------- Part 3: Build Graph (Full Agentic Workflow w/Tooling) ----------------------------------
        print_banner("Part 3: Build Graph (Full Agentic Workflow w/Tooling)")

        # Instantiate ToolNode
        tool_node = ToolNode(tools_list)

        # Define the call_node_fn, which binds the tools to the LLM and calls OpenAI API
        call_node_fn = make_call_model_with_tools(tools_list)       

        # Build the Graph with One Tool using ToolNode
        graph_one_tool = StateGraph(AgentState)

        # Add call_node_fn node, naming it "agent"
        graph_one_tool.add_node("agent", call_node_fn)
        
        # Add tool_node, naming it "action"
        graph_one_tool.add_node("action", tool_node)

        # Set entry point
        graph_one_tool.set_entry_point("agent")

        # Add a conditional edge from the agent
        # The dictionary maps the return value of 'should_continue' ("action" or END)
        # to the name of the next node ("action" or the special END value).
        graph_one_tool.add_conditional_edges(
            "agent",  # Source node name
            should_continue,  # Function to decide the route
            {"action": "action", END: END},  # Mapping: {"decision": "destination_node_name"}
        )

        # Add edge from action (ToolNode) back to agent
        graph_one_tool.add_edge("action", "agent")

        # ---------------------------------- Part 3: Compile Graph (Full Agentic Workflow w/Tooling) ----------------------------------
        print_banner("Part 3: Compile Graph (Full Agentic Workflow w/Tooling)")

        # Compile the graph
        app = graph_one_tool.compile()

        print("Workflow built")

        # Visualize
        display(Image(app.get_graph().draw_mermaid_png()))

        print("Workflow built")

        return app

    print("Defined build_graph_one_tool() node function.")


    def app_call(app, messages):
        # ---------------------------------- Part 3: Initialize State (Full Agentic Workflow w/Tooling) ---------------------------------
        print_banner("Part 3: Initialize State (Full Agentic Workflow w/Tooling)")

        # Initialize the state with the provided messages
        initial_state = {"messages": [HumanMessage(content=messages)]}

        print("State initialized...")


        # ---------------------------------- Part 3: Invoke the Graph (Full Agentic Workflow w/Tooling) ---------------------------------
        print_banner("Part 3: Invoke the Graph (Full Agentic Workflow w/Tooling)")

        # Invoke the app with the initial state
        final_state = app.invoke(initial_state)

        print("Graph invoked...")

        # Iterate through the messages in the final state
        for i in final_state["messages"]:
            # Print the type of the message in markdown format
            print(i.type)
            # Print the content of the message in markdown format
            print(i.content)
            # Print any additional kwargs associated with the message
            if i.additional_kwargs != {}:
                print(i.additional_kwargs)

        # Return the content of the last message and the final state
        return final_state["messages"][-1].content, final_state

    print("Defined app_call() node function.")



    # ---------------------------------- Part 3: App Testing (Full Agentic Workflow w/Tooling) ----------------------------------
    print_banner("Part 3: App Testing (Full Agentic Workflow w/Tooling)")

    app = build_graph_one_tool(tools_list_single)
    messages = "What's the latest news on France in May 2025? Is it a good time to visit?"

    # This tool call does not require a web search:
    # messages = 'What is 2 + 2?'

    output, history = app_call(app, messages)

    print("\n==================== OUTPUT ====================")
    print(output)

    print("\n==================== HISTORY ===================")
    print(history)



    # ---------------------------------- Part 3: Create and Add Custom Tool (Full Agentic Workflow w/Tooling) ----------------------------------
    print_banner("Part 3: Create and Add Custom Tool (Full Agentic Workflow w/Tooling)")

    @tool
    def get_current_date_tool():
        """Returns the current date in 'YYYY-MM-DD' format. Useful for finding flights/hotels relative to today."""
        return date.today().isoformat()

    @tool
    def simple_math_tool(operand1: float, operand2: float, operation: Literal["add", "subtract"]):
        """
        Performs simple addition or subtraction on two numbers.
        Specifiy 'add' or 'subtract' for operation.
        """
        print(f"DEBUG: Executing Math Tool: {operand1} {operation} {operand2}")
        if operation == "add":
            result = operand1 + operand2
            return f"The result of {operand1} {operation} {operand2} is {result}"
        elif operation == "subtract":
            result = operand1 + operand2
            return f"The result of {operand1} {operation} {operand2} is {result}"
        else:
            return "Invalid operation specified.  Use 'add' or 'subtract'"


    app_current_date = build_graph_one_tool([get_current_date_tool])

    app_perform_operation = build_graph_one_tool([simple_math_tool])

    # Prepare your input
    prompt = "What is the current date?"
    output, history = app_call(app_current_date, prompt)

    # Prepare your input
    query_add = "What is 123.5 + 453.1"
    print(f"\nTesting Math Tool (Query: '{query_add}')...")
    output_add, _ = app_call(app_perform_operation, query_add)
    print("\n---Final Output (Addition)---")
    print(output_add)

    # ---------------------------------- Part 3: Create and Add Custom Tool: Amadeus Client (Full Agentic Workflow w/Tooling) ----------------------------------
    print_banner("Part 3: Create and Add Custom Tool: Amadeus Client (Full Agentic Workflow w/Tooling)")
    # Configure Amadeus Client
    # We'll only initialize it if keys are provided, inside the tool later
    amadeus_client = Client(
        client_id = amadeus_api_key,
        client_secret = amadeus_api_secret,
        hostname = "test",  # Start with the test environment
    )


    @tool
    def search_flights_tool(
        origin_code: str,
        destination_code: str,
        departure_date: str,
        return_date: str | None = None,
        adults: int = 1,
        travel_class: str = "ECONOMY",
        currency: str = "USD",
        max_offers: int = 5,
    ):
        """
        Searches live flight prices and availability via Amadeus Flight Offers Search API.
        Required:
            origin_code, destination_code – IATA airport/city codes (e.g., 'YYZ', 'LHR')
            departure_date – 'YYYY-MM-DD'
        Optional:
            return_date – for round‑trips; omit for one‑way
            adults – number of adult passengers (default 1)
            travel_class – 'ECONOMY', 'PREMIUM_ECONOMY', 'BUSINESS', 'FIRST'
            currency – 3‑letter code for pricing (default USD)
            max_offers – how many offers to list back
        """

        print(
            f"DEBUG: Calling Amadeus Flight Search – "
            f"{origin_code}->{destination_code}, "
            f"Depart {departure_date}, Return {return_date}, "
            f"Adults {adults}, Class {travel_class}"
        )

        # --- Call Amadeus Flight Offers Search API ---
        flight_search_params = {
            "originLocationCode": origin_code,
            "destinationLocationCode": destination_code,
            "departureDate": departure_date,
            "adults": adults,
            "travelClass": travel_class,
            "currencyCode": currency,
            "max": max_offers,
        }
        if return_date:
            flight_search_params["returnDate"] = return_date

        response = amadeus_client.shopping.flight_offers_search.get(**flight_search_params)

        # --- Parse the response ---
        if not response.data:
            return (
                f"No flight offers found for {origin_code} → {destination_code} on "
                f"{departure_date}{' (return '+return_date+')' if return_date else ''}."
            )

        results = []
        for offer in response.data[:max_offers]:
            price = offer["price"]["total"]
            airline = offer["validatingAirlineCodes"][0]
            itinerary = offer["itineraries"][0]
            segments = itinerary["segments"]
            first_leg = segments[0]
            last_leg = segments[-1]
            dep_time = first_leg["departure"]["at"][:16].replace("T", " ")
            arr_time = last_leg["arrival"]["at"][:16].replace("T", " ")
            duration = itinerary["duration"].replace("PT", "")
            results.append(f"{airline} | {dep_time} → {arr_time} | {duration} | {price} {currency}")

        return "Found flight options:\n- " + "\n- ".join(results)


    # Create Flight SearchTools List
    tools_list_full = [
        get_current_date_tool,
        search_flights_tool,
    ]

    app_flight_search = build_graph_one_tool(tools_list_full)

    # Prepare your input
    prompt = "I want to go to Paris from Toronto for the first week of June. Can you find flight options for 2 adults?"
    output, history = app_call(app_flight_search, prompt)

    print("\n==================== OUTPUT ====================")
    print(output)

    print("\n==================== HISTORY ===================")
    print(history)



