# ---------------------------------- Libraries ----------------------------------
import base64
from dotenv import load_dotenv
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
# def print_markdown(text):
#     """Displays text as Markdown in Jupyter."""
#     display(Markdown(text))


def encode_image_to_base64(image_path_or_pil):
    # Check if a file path (str) or a PIL Image object is provided
    if isinstance(image_path_or_pil, str):
        # Check if the file exists
        if not os.path.exists(image_path_or_pil):
            raise FileNotFoundError(f"Image file not found at: {image_path_or_pil}")
        with open(image_path_or_pil, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
            
    elif isinstance(image_path_or_pil, Image.Image):  # If it's a PIL Image object
        buffer = io.BytesIO()
        image_format = image_path_or_pil.format or "JPEG"  # Default to JPEG if format unknown
        image_path_or_pil.save(buffer, format=image_format)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    else:
        raise ValueError("Input must be a file path (str) or a PIL Image object.")


def query_openai_vision(client, image, prompt, model = "gpt-4o", max_tokens = 100):
    # Encode the image to base64
    base64_image = encode_image_to_base64(image)
    
    try:
        # Construct the message payload
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ]

        # Make the API call
        response = client.chat.completions.create(
            model = model,
            messages = messages,
            max_tokens = max_tokens,
        )

        # Extract the response
        return response.choices[0].message.content
    
    except Exception as e:
        return f"Error calling API: {e}"
    

# ---------------------------------- Load Image ----------------------------------
image_filename = r"C:\Users\Laptop\Desktop\Coding\LLM Engineering - Ryan Ahmed\02\Images\cook-out-tray.png"

# Load the image
img = Image.open(image_filename)
print(f"Image       '{image_filename}' loaded successfully.")
print(f"Format:      {img.format}")
print(f"Size:        {img.size}")
print(f"Mode:        {img.mode}")

# Display the image
# display(img)

# Keep the loaded image object in a variable for later use
image_to_analyze = img


# ---------------------------------- LLM Food Recognition: Zero-Shot Prompting ----------------------------------
# Zero-shot prompting - Refers to prompting a language model to perform a task without providing any prior examples or demonstrations.
# Chain of Thought Prompt – Improves the reasoning of LLMs by breaking down complex problems or tasks into smaller problems or tasks.  

prompt = """
Context: I'm anallyzing a food image for a calorie estimation application.
Instruction: Please identify the food in this image. 
Input: [The image i'm showing you]
Output: Provide a list of the food items you recognize in the image and mention its typical ingredients.
"""


print("🤖 Querying OpenAI Vision...")
openai_description = query_openai_vision(
    openai_client, 
    image_to_analyze, 
    prompt
)
print(openai_description)

# 1. **Cheeseburger**
#    - **Typical Ingredients:**
#      - Bun
#      - Beef patty
#      - Cheese
#      - Pickles
#      - Ketchup and/or mustard
#      - Lettuce, tomato, and onions (optional)

# 2. **French Fries**
#    - **Typical Ingredients:**
#      - Potatoes
#      - Oil for frying (e.g., vegetable or canola oil)
#      - Salt

# 3. **Chicken Nuggets:**
#    - **Ingredients:** Chicken meat, breading (flour and breadcrumbs)
                                              

# Uncomment to see the markdown format when working within Jupyter Notebook
# print_markdown(openai_description)


# ---------------------------------- LLM Food Recognition: Prompt Engineering ----------------------------------
# Prompt engineering - The process of designing and refining prompts to elicit desired responses from language models.

structured_nutrition_prompt = """
# Nutritional Analysis Task

## Context
You are a nutrition expert analyzing food images to provide accurate nutritional information.

## Instructions
Analyze the food item in the image and provide estimated nutritional information based on your knowledge.

## Input
- An image of a food item

## Output
Provide the following estimated nutritional information for a typical serving size or per 100g:
- food_name (string)
- serving_description (string, e.g., '1 slice', '100g', '1 cup')
- calories (float)
- fat_grams (float)
- protein_grams (float)
- confidence_level (string: 'High', 'Medium', or 'Low')

**IMPORTANT:** Respond ONLY with a single JSON object containing these fields. Do not include any other text, explanations, or apologies. The JSON keys must match exactly: "food_name", "serving_description", "calories", "fat_grams", "protein_grams", "confidence_level". If you cannot estimate a value, use `null`.

Example valid JSON response:
{
  "food_name": "Banana",
  "serving_description": "1 medium banana (approx 118g)",
  "calories": 105.0,
  "fat_grams": 0.4,
  "protein_grams": 1.3,
  "confidence_level": "High"
}
"""


openai_nutrition_result = query_openai_vision(client = openai_client,
                                              image = image_to_analyze,
                                              prompt = structured_nutrition_prompt,)
print(openai_nutrition_result)

# {
#   "food_name": "Cheeseburger with fries and chicken nuggets",
#   "serving_description": "1 meal",
#   "calories": 950.0,
#   "fat_grams": 50.0,
#   "protein_grams": 30.0,
#   "confidence_level": "Medium"
# }



# ---------------------------------- Sentiment Analysis ----------------------------------
message = """Operator: Good morning, and welcome to Solid Power's Fourth Quarter 2024 Earnings Conference Call. At this time, all participants are in a listen-only mode. After management’s prepared remarks, we will open the call for questions. I would now like to turn the call over to our CEO, Mark Reynolds. Please go ahead.
CEO Mark Reynolds: Thank you, and good morning, everyone. I’m pleased to share our results for Q4 2024 and our outlook for the year ahead. Despite ongoing macroeconomic uncertainties, Solid Power posted strong revenue
growth of 8.2% year-over-year, reaching $420 million for the quarter. This marks our ninth consecutive quarter of revenue expansion, driven by continued demand for high-performance air suspension systems and strategic investments in supply chain resilience.
Key Highlights:
- Gross margin expanded to 42.1%, reflecting improved production efficiency and favourable pricing strategies.
- EBITDA came in at $78.5 million, a 6.5% increase from last year.
- Net income for the quarter was $24.8 million, or $1.35 per share, up from $1.20 per share in Q3 2024.
- Cash flow from operations totalled $50 million, reinforcing our strong liquidity
position."""

sentiment_prompt = """
Context: You are a sentiment analysis expert. Analyze the sentiment of the following text from an earnings call transcript.
Instruction: Provide the overall sentiment as 'Positive', 'Negative', or 'Neutral', along with a brief explanation for your assessment.
Output Format: Respond only with a single JSON object containing these fields: "sentiment" and "explanation".  Do not include anything else.
Example valid JSON response:
{
  "sentiment": "Positive",
  "explanation": "The CEO highlights strong revenue growth, improved gross margins, and increased net income, indicating a positive outlook for the company."
}
"""

openai_sentiment_result = openai_client.chat.completions.create(
    model="gpt-4o",   
    messages = [
        {
            "role": "user",
            "content": sentiment_prompt + "\n Text:\n" + message
        }
    ],

    max_tokens=150,
)


print(openai_sentiment_result.choices[0].message.content)
# {
#   "sentiment": "Positive",
#   "explanation": "The CEO reports strong revenue growth, improved gross margins, increased EBITDA, net income growth, and strong cash flow from operations, indicating a favorable financial performance and outlook for the company."
# }