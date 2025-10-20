# ---------------------------------- Libraries ----------------------------------
import base64
from dotenv import load_dotenv
import gradio as gr
import io 
from IPython.display import display, Markdown  
from openai import OpenAI  
import os
from PIL import Image  


# ---------------------------------- Load Env ----------------------------------
# Load environment variables and create OpenAI client
load_dotenv(dotenv_path="C:\\Users\\Laptop\\Desktop\\Coding\\LLM Engineering - Ryan Ahmed\\LLM and Agentic AI Bootcamp Materials\\.env", override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key = openai_api_key)

# Let's view the first few characters in the key
print(openai_api_key[:15])


# ---------------------------------- Functions ----------------------------------
# Define a helper function to display markdown nicely in Jupyter notebooks
# def print_markdown(text):
#     """Displays text as Markdown in Jupyter."""
#     display(Markdown(text))


def get_ai_tutor_response(user_question):
    # Define the system prompt - instructions for the AI's personality and role
    system_prompt = "You are a tutor, explain concepts in a clear and concise manner suitable for beginners.  But don't be overly wordy."

    try:
        # Make the API call to OpenAI
        response = openai_client.chat.completions.create(
            model = "gpt-4o-mini",
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_question}],
            temperature = 0.7,
        )
        # Extract the answer content
        ai_response = response.choices[0].message.content
        return ai_response

    except Exception as e:
        # Server side error handling
        print(f"An error occurred: {e}")
        # Client side error message
        return f"Sorry, I encountered an error trying to get an answer: {e}"
    

def stream_ai_tutor_response(user_question):
    system_prompt = "You are a helpful and patient AI Tutor. Explain concepts clearly and concisely."

    try:
        # Note: stream = True is the key change here!
        stream = openai_client.chat.completions.create(
            model = "gpt-4o-mini",
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_question}],
            temperature = 0.7,
            stream = True,  # Enable streaming (magic happens here)
        )

        # Iterate through the response chunks
        full_response = ""  # Keep track of the full response if needed later

        # Loop through each chunk of the response as it arrives
        for chunk in stream:
            # Check if this chunk contains actual text content
            if chunk.choices[0].delta and chunk.choices[0].delta.content:
                # Extract the text from this chunk
                text_chunk = chunk.choices[0].delta.content
                # Add this chunk to our growing response
                full_response += text_chunk
                # 'yield' is special - it sends the current state of the response to Gradio
                # This makes the text appear to be typing in real-time
                yield full_response

    except Exception as e:
        # Server side error handling
        print(f"An error occurred during streaming: {e}")
        # Client side error message
        yield f"Sorry, I encountered an error: {e}"


def stream_ai_tutor_response_with_level(user_question, explanation_level_value):
    # Get the descriptive text for the chosen level
    level_description = explanation_levels.get(
        explanation_level_value, "clearly and concisely"
    )  # Default if level not found

    # Construct the system prompt dynamically based on the level
    system_prompt = f"You are a helpful AI Tutor. Explain the following concept {level_description}."

    print(f"DEBUG: Using System Prompt: '{system_prompt}'")  # For checking

    try:
        stream = openai_client.chat.completions.create(
            model = "gpt-4o-mini",
            messages = [{"role": "system", "content": system_prompt}, 
                        {"role": "user", "content": user_question}],
            temperature = 0.7,
            stream = True,
        )

        # Iterate through the response chunks
        full_response = ""  # Keep track of the full response if needed later

        # Loop through each chunk of the response as it arrives
        for chunk in stream:
            # Check if this chunk contains actual text content
            if chunk.choices[0].delta and chunk.choices[0].delta.content:
                # Extract the text from this chunk
                text_chunk = chunk.choices[0].delta.content
                # Add this chunk to our growing response
                full_response += text_chunk
                # 'yield' is special - it sends the current state of the response to Gradio
                # This makes the text appear to be typing in real-time
                yield full_response

    except Exception as e:
        # Server side error handling
        print(f"An error occurred during streaming: {e}")
        # Client side error message
        yield f"Sorry, I encountered an error: {e}"


# ---------------------------------- Main ----------------------------------
# # Let's test our function with a sample question
# test_question = "Could you explain the concept of functions in Python and their purpose in programming?"
# print(f"\n💬 User Question:\n{test_question}")

# # Call the function and store the response
# tutor_answer = get_ai_tutor_response(test_question)

# # Print the AI's response
# print(f"\n🤖 AI Tutor's Response:\n{tutor_answer}")



# ---------------------------------- Gradio Interface ----------------------------------
# ai_tutor_interface_simple = gr.Interface(
#     fn = get_ai_tutor_response,
#     inputs = gr.Textbox(lines = 2, placeholder = "Ask the AI Tutor anything...", label = "Your Question"),
#     outputs = gr.Textbox(label = "AI Tutor's Answer"),
#     title = "🤖 Simple AI Tutor",
#     description = "Enter your question below and the AI Tutor will provide an explanation. Powered by OpenAI.",
#     allow_flagging = "never",  # Disables the flagging feature for simplicity
# )

# # Launch the Gradio interface
# ai_tutor_interface_simple.launch()


# ---------------------------------- Gradio Interface w/Streaming----------------------------------
# ai_tutor_interface_streaming = gr.Interface(
#     fn = stream_ai_tutor_response,  # Use the generator function
#     inputs = gr.Textbox(lines = 2, placeholder = "Ask the AI Tutor a question...", label = "Your Question"),
#     outputs = gr.Markdown(
#         label = "AI Tutor's Answer (Streaming)", container = True, height = 250
#     ),  # Output is still a Markdown (it renders as HTML), container lets it be scrollable and height is set to 250px ( for better visibility)
#     title = "🤖 AI Tutor with Streaming",
#     description = "Enter your question. The answer will appear word-by-word!",
#     allow_flagging = "never",
# )

# # Launch the streaming interface
# print("Launching Streaming Gradio Interface...")
# ai_tutor_interface_streaming.launch()


# ---------------------------------- Gradio Interface w/Streaming and Explanation Level ----------------------------------
# Define the mapping for explanation levels
explanation_levels = {
    1: "like I'm 5 years old",
    2: "like I'm 10 years old",
    3: "like I'm a high school student",
    4: "like I'm a college student",
    5: "like I'm an expert in the field",
}

# Define the Gradio interface with both Textbox and slider inputs
ai_tutor_interface_slider = gr.Interface(fn = stream_ai_tutor_response_with_level,  # Function now takes 2 args
    inputs=[
        gr.Textbox(lines = 3, placeholder = "Ask the AI Tutor a question...", label = "Your Question"),
        gr.Slider(
            minimum = 1,
            maximum = 5,
            step = 1,  # Only allow whole numbers
            value = 3,  # Default level (high school)
            label = "Explanation Level",  # Label for the slider
        ),
    ],
    outputs = gr.Markdown(label = "AI Tutor's Explanation (Streaming)", container = True, height = 250),
    title = "🎓 Advanced AI Tutor",
    description = "Ask a question and select the desired level of explanation using the slider.",
    allow_flagging = "never",
)

# Launch the advanced interface
print("Launching Advanced Gradio Interface with Slider...")
ai_tutor_interface_slider.launch()


