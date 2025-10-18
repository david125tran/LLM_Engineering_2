# ---------------------------------- Libraries ----------------------------------
from dotenv import load_dotenv
from openai import OpenAI  
import os


# ---------------------------------- Load Env ----------------------------------
# Load environment variables and create OpenAI client
load_dotenv(dotenv_path="C:\\Users\\Laptop\\Desktop\\Coding\\LLM Engineering - Ryan Ahmed\\LLM and Agentic AI Bootcamp Materials\\.env", override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key = openai_api_key)


# ---------------------------------- Testing the API Call to OpenAI ----------------------------------
my_message = "Write a Poem describing the current weather in Durham, NC"
print(f"Sending message to OpenAI: '{my_message}'")

response = openai_client.chat.completions.create(model = "gpt-4o-mini",
                                                 messages = [
                                                     {"role": "user",
                                                     "content": my_message}])
ai_reply_content = response.choices[0].message.content

# Print the reply
print("\n🤖 AI's Reply: \n")
print(f"{ai_reply_content}")

# 🤖 AI's Reply: 

# In Durham's embrace, the skies painted gray,  
# A tapestry woven where clouds softly sway.  
# The whispers of autumn weave through the trees,  
# As crisp, gentle breezes dance through the leaves.  

# With hints of the past, the warmth fades away,  
# While shadows grow longer at the end of the day.  
# A sprinkle of rain, like a soft, tender kiss,  
# Caresses the earth, a moment of bliss.

# Golden-hued pumpkins adorn every porch,
# As the scent of spiced cider begins to encroach.
# The laughter of children rings clear in the air,
# While thick sweaters wrap us, a cozy affair.

# The streets of this town, with their stories untold,
# Bask in a beauty that never grows old.
# For as seasons do change and the weather does shift,
# Durham remains, a cherished, warm gift.

# So here's to the winds that cradle our dreams,
# To the sun that breaks through with golden gleams,
# In the heart of this city, come rain or come shine,
# Durham, dear Durham, forever you'll shine.

# Explore meta data to the response
response_dict = response.model_dump()

for key in response_dict:
    print(f"\n{key}:\n{response_dict[key]}")

# id:
# chatcmpl-CPV5JwC4FCntB7nGRAPMFEdaKBQkB

# choices:
# [{'finish_reason': 'stop', 'index': 0, 'logprobs': None, 'message': {'content': "In Durham's embrace, the skies painted gray,  \nA tapestry woven where clouds softly sway.  \nThe whispers of autumn weave through the trees,  \nAs crisp, gentle breezes dance through the leaves.  \n\nWith hints of the past, the warmth fades away,  \nWhile shadows grow longer at the end of the day.  \nA sprinkle of rain, like a soft, tender kiss,  \nCaresses the earth, a moment of bliss.  \n\nGolden-hued pumpkins adorn every porch,  \nAs the scent of spiced cider begins to encroach.  \nThe laughter of children rings clear in the air,  \nWhile thick sweaters wrap us, a cozy affair.  \n\nThe streets of this town, with their stories untold,  \nBask in a beauty that never grows old.  \nFor as seasons do change and the weather does shift,  \nDurham remains, a cherished, warm gift.  \n\nSo here's to the winds that cradle our dreams,  \nTo the sun that breaks through with golden gleams,  \nIn the heart of this city, come rain or come shine,  \nDurham, dear Durham, forever you'll shine.  ", 'refusal': None, 'role': 'assistant', 'annotations': [], 'audio': None, 'function_call': None, 'tool_calls': None}}]

# created:
# 1760193725

# model:
# gpt-4o-mini-2024-07-18

# object:
# chat.completion

# service_tier:
# default

# system_fingerprint:
# fp_560af6e559

# usage:
# {'completion_tokens': 229, 'prompt_tokens': 19, 'total_tokens': 248, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}


# -------------------------------- Controlling System Instructions ----------------------------------
character_personalities = {
    "David Tran": "You are David Tran, a software engineer with a life sciences background.  You have extensive LLM engineering experience and are an expert in the field of science.  You will ask if they need help developing scientific software.",
    "Shakespeare": "You are William Shakespeare, the famous playwright and poet.  You speak in iambic pentameter and rhyme all of your responses.  You will ask if they want a poem written about their day.",
}

chosen_character = "David Tran"

system_instructions = character_personalities[chosen_character]

# Define user message
user_first_message = "What are you up to today?"

# Make OpenAi API Call with System Instructions
response = openai_client.chat.completions.create(model = "gpt-4o-mini",
                                                 messages = [
                                                     # The system prompt goes first
                                                     {"role": "system", "content": system_instructions},
                                                     # Followed by the user's message
                                                     {"role": "user", "content": user_first_message},],)

# Let's Show the AI's reply
ai_character_reply = response.choices[0].message.content

print("\nReceived response!")
print(f"🤖 {chosen_character}'s Reply: \n")
print(f"{ai_character_reply}")

# Received response!
# 🤖 David Tran's Reply: 

# I'm here and ready to help! Do you need assistance with developing any scientific software or have questions related to life sciences or LLM engineering?

