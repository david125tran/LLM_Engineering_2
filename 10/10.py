# ---------------------------------- Libraries ----------------------------------
from dotenv import load_dotenv
import gradio as gr
from langchain_openai import OpenAIEmbeddings, OpenAI as LangChainOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQAWithSourcesChain 
from openai import OpenAI as OpenAIClient
import os


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
    mid = 29 - banner_len // 2

    print("\n\n\n")
    print("*" + "-*" * 30)
    if (banner_len % 2 != 0):
        print("*"  + " " * mid + text + " " * mid + "*")
    else:
        print("*"  + " " * mid + text + " " + " " * mid + "*")
    print("*" + "-*" * 30)



# ---------------------------------- Load Environment Variables ----------------------------------
print_banner("Load Environment Variables")
# Load environment variables and create OpenAI client
load_dotenv(dotenv_path=parent_dir + r"\.env", override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAIClient(api_key = openai_api_key)

# View the first few characters in the key
print(openai_api_key[:15])


# ---------------------------------- (1) Knowledge Base ----------------------------------
print_banner("(1) Knowledge Base")
DATA_FILE_PATH = script_dir + r"\eleven_madison_park_data.txt"
print(f"Data file path set to: {DATA_FILE_PATH}")


# ---------------------------------- (2) Load ----------------------------------
print_banner("(2) Load")
# Load the data from the text file and specify UTF-8 encoding
loader = TextLoader(DATA_FILE_PATH, encoding = "utf-8")

# Load the documents from LangChain as a Document object
raw_documents = loader.load()
print(f"Successfully loaded {len(raw_documents)} document(s).")
print(f"Document preview:\n{raw_documents[0].page_content[:50]}" + "...")


# ---------------------------------- (3) Preprocess and Chunk ----------------------------------
print_banner("(3) Preprocess and Chunk")

# Split the document into chunks
print("\nSplitting the loaded document into smaller chunks...")

# Iitialize the splitter, which tries to split the document on common separators like paragraphs (\n\n), sentences (.), and spaces (' ').
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,  # Aim for chunks of about 1000 characters
                                               chunk_overlap = 150,)  # Each chunk overlaps with the previous by 150 characters

# Split the raw documents into smaller Document objects (chunks)
documents = text_splitter.split_documents(raw_documents)

# Check if splitting produced any documents
if not documents:
    raise ValueError("Error: Splitting resulted in zero documents. Check the input file and splitter settings.")
print(f"Document split into {len(documents)} chunks.")

# Display an example chunk and its metadata
print("\n--- Preview of Chunk (Chunk 2) ---")
print(documents[2].page_content[:200] + "...")
print("\n--- Example Chunk (Chunk 2) ---")
print(documents[2].page_content)
print("\n--- Metadata for Chunk 2 ---")
print(documents[2].metadata)


# ---------------------------------- (4) Embedding Model ----------------------------------
print_banner("(4) Embedding Model")
# Initialize the OpenAI Embeddings model
print("Initializing OpenAI Embeddings model...")

# Create an instance of the OpenAI Embeddings model
embeddings = OpenAIEmbeddings(openai_api_key = openai_api_key)
print("OpenAI Embeddings model initialized.")


# ---------------------------------- (5) Create Vector Store and Embeddings ----------------------------------
print_banner("(5) Create Vector Store and Embeddings")
# Create ChromaDB Vector Store
print("\nCreating ChromaDB vector store and embedding documents...")

# Convert the chunks from the documents into vectors and store them in ChromaDB using the embeddings model.
# You can add `persist_directory="./my_chroma_db"` to save it to disk
vector_store = Chroma.from_documents(documents = documents, embedding = embeddings)  

# Verify the number of items in the store
vector_count = vector_store._collection.count()
print(f"ChromaDB vector store created with {vector_count} items.")

if vector_count == 0:
    raise ValueError("Vector store creation resulted in 0 items. Check previous steps.")

# Retrieve the first chunk of stored data from the vector store
stored_data = vector_store._collection.get(include=["embeddings", "documents"], limit = 1)  

# Display the results of the raw text chunk originally inserted at index 0
print("First chunk text:\n", stored_data['documents'][0])
print("\nEmbedding vector:\n", stored_data['embeddings'][0])
print(f"\nFull embedding has {len(stored_data['embeddings'][0])} dimensions.")

# Test the embedding model with a sample text (perform a similarity search in our vector store)
test_query = "What different menus are offered?"
print(f"Searching for documents similar to: '{test_query}'")

# Perform a similarity search. 'k=2' retrieves the top 2 most similar chunks
try:
    # Chroma computes cosine similarity between that query vector and all stored document vectors.
    # There are other methods but cosine similarity is common for text embeddings.
    similar_docs = vector_store.similarity_search(test_query, k = 2)
    print(f"\nFound {len(similar_docs)} similar documents:")

    # Display snippets of the retrieved documents and their sources
    for i, doc in enumerate(similar_docs):
        print(f"\n--- Document {i+1} ---")
        # Displaying the first 700 chars for brevity
        content_snippet = doc.page_content[:700].strip() + "..."
        source = doc.metadata.get("source", "Unknown Source")  # Get source from metadata
        print(f"Content Snippet: {content_snippet}")
        print(f"Source: {source}")

except Exception as e:
    print(f"An error occurred during similarity search: {e}")


# ---------------------------------- (6) Define the Retriever ----------------------------------
print_banner("(6) Define the Retriever")
# --- 1. Define the Retriever ---
# The retriever uses the vector store to fetch documents
# We configure it to retrieve the top 'k' documents
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
print("Retriever configured successfully from vector store.")

# --- 2. Define the Language Model (LLM) from OpenAI---
# Temperature controls the model's creativity; 'temperature=0' aims for more deterministic, less creative answers
llm = LangChainOpenAI(temperature=1.0, openai_api_key = openai_api_key)
print("OpenAI LLM successfully initialized.")

# --- 3. Create the RetrievalQAWithSourcesChain ---
# This chain type is designed specifically for Q&A with source tracking.
# chain_type="stuff": Puts all retrieved text directly into the prompt context.
#                      Suitable if the total text fits within the LLM's context limit.
#                      Other types like "map_reduce" handle larger amounts of text.
qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm = llm,
                                                       chain_type = "stuff",
                                                       retriever = retriever,
                                                       return_source_documents = True,  # Request the actual Document objects used
                                                       verbose = True)  # Set to True to see Langchain's internal steps (can be noisy)

print("RetrievalQAWithSourcesChain created")


# ---------------------------------- (7) Semantic Search ----------------------------------
print_banner("(7) Semantic Search")
print("\n--- Performing a semantic search ---")
chain_test_query = "What kind of food does Eleven Madison Park serve?"
print(f"Query: {chain_test_query}")

try:
    result = qa_chain.invoke({"question": chain_test_query})

    # Print the answer and sources from the result dictionary
    print("\n--- Answer ---")
    print(result.get("answer", "No answer generated."))

    print("\n--- Sources ---")
    print(result.get("sources", "No sources identified."))

    # Optionally print snippets from the source documents returned
    if "source_documents" in result:
        print("\n--- Source Document Snippets ---")
        for i, doc in enumerate(result["source_documents"]):
            content_snippet = doc.page_content[:250].strip()
            print(f"Doc {i+1}: {content_snippet}")

except Exception as e:
    print(f"\nAn error occurred while running the chain: {e}")


# ---------------------------------- Gradio App ----------------------------------
print_banner("Gradio App")

# This function takes the user's input, runs the chain, and formats the output
# Ensure the `qa_chain` variable is accessible in this scope.
def ask_elevenmadison_assistant(user_query):
    """
    Processes the user query using the RAG chain and returns formatted results.
    """
    print(f"\nProcessing Gradio query: '{user_query}'")
    if not user_query or user_query.strip() == "":
        print("--> Empty query received.")
        return "Please enter a question.", ""  # Handle empty input gracefully

    try:
        # Run the query through our RAG chain
        result = qa_chain.invoke({"question": user_query})

        # Extract answer and sources
        answer = result.get("answer", "Sorry, I couldn't find an answer in the provided documents.")
        sources = result.get("sources", "No specific sources identified.")

        # Basic formatting for sources (especially if it just returns the filename)
        if sources == DATA_FILE_PATH:
            sources = f"Retrieved from: {DATA_FILE_PATH}"
        elif isinstance(sources, list):  # Handle potential list of sources
            sources = ", ".join(list(set(sources)))  # Unique, comma-separated

        print(f"--> Answer generated: {answer[:100].strip()}...")
        print(f"--> Sources identified: {sources}")

        # Return the answer and sources to be displayed in Gradio output components
        return answer.strip(), sources

    except Exception as e:
        error_message = f"An error occurred: {e}"
        print(f"--> Error during chain execution: {error_message}")
        # Return error message to the user interface
        return error_message, "Error occurred"


# --- Create the Gradio Interface using Blocks API ---
print("\nSetting up Gradio interface...")

with gr.Blocks(theme=gr.themes.Soft(), title="Eleven Madison Park Q&A Assistant") as demo:
    # Title and description for the app
    gr.Markdown(
        """
        # Eleven Madison Park - AI Q&A Assistant 💬
        Ask questions about the restaurant based on its website data.
        The AI provides answers and cites the source document.
        *(Examples: What are the menu prices? Who is the chef? Is it plant-based?)*
        """
    )

    # Input component for the user's question
    question_input = gr.Textbox(
        label = "Your Question:",
        placeholder = "e.g., What are the opening hours on Saturday?",
        lines = 2,  # Allow a bit more space for longer questions
    )

    # Row layout for the output components
    with gr.Row():
        # Output component for the generated answer (read-only)
        answer_output = gr.Textbox(label="Answer:", interactive=False, lines=6)  # User cannot edit this
        # Output component for the sources (read-only)
        sources_output = gr.Textbox(label="Sources:", interactive=False, lines=2)

    # Row for buttons
    with gr.Row():
        # Button to submit the question
        submit_button = gr.Button("Ask Question", variant="primary")
        # Clear button to reset inputs and outputs
        clear_button = gr.ClearButton(components=[question_input, answer_output, sources_output], value="Clear All")

    # Add some example questions for users to try
    gr.Examples(
        examples=[
            "What are the different menu options and prices?",
            "Who is the head chef?",
            "What is Magic Farms?"],
        inputs=question_input,  # Clicking example loads it into this input
        # We could potentially add outputs=[answer_output, sources_output] and cache examples
        # but that requires running the chain for each example beforehand.
        cache_examples=False,  # Don't pre-compute results for examples for simplicity
    )

    # --- Connect the Submit Button to the Function ---
    # When submit_button is clicked, call 'ask_emp_assistant'
    # Pass the value from 'question_input' as input
    # Put the returned values into 'answer_output' and 'sources_output' respectively
    submit_button.click(fn = ask_elevenmadison_assistant, inputs = question_input, outputs = [answer_output, sources_output])

print("Gradio interface defined.")

# --- Launch the Gradio App ---
print("\nLaunching Gradio app... (Stop the kernel or press Ctrl+C in terminal to quit)")
# demo.launch() # Launch locally in the notebook or browser
demo.launch()  


# http://127.0.0.1:7860









# ------------------------------------ Output ----------------------------------


# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
# *                    (1) Knowledge Base                     *
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
# Data file path set to: c:\Users\Laptop\Desktop\Coding\LLM Engineering - Ryan Ahmed\10\eleven_madison_park_data.txt




# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
# *                         (2) Load                          *
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
# Successfully loaded 1 document(s).
# Document preview:
# Source: https://www.elevenmadisonpark.com/
# Title: ...




# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
# *                 (3) Preprocess and Chunk                  *
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

# Splitting the loaded document into smaller chunks...
# Document split into 38 chunks.

# --- Preview of Chunk (Chunk 2) ---
# Join Our Team Eleven Madison Park ▾ All Businesses Eleven Madison Park Clemente Bar Daniel Humm Hospitality Filter Categories Culinary Pastry Wine & Beverage Dining Room Office & Admin Other Job Types...

# --- Example Chunk (Chunk 2) ---
# Join Our Team Eleven Madison Park ▾ All Businesses Eleven Madison Park Clemente Bar Daniel Humm Hospitality Filter Categories Culinary Pastry Wine & Beverage Dining Room Office & Admin Other Job Types Full Time Part Time Compensation Salary Hourly Apply filters OPEN OPPORTUNITIES Staff Accountant - Part Time Eleven Madison Park Part Time • Hourly ($20 - $25) Host/Reservationist Eleven Madison Park Full Time • Hourly ($24) Sous Chef Eleven Madison Park Full Time • Salary ($72K - $75K) Pastry Cook Eleven Madison Park Full Time • Hourly ($18 - $20) Kitchen Server Eleven Madison Park Full Time • Hourly ($16) plus tips Dining Room Manager Eleven Madison Park Full Time • Salary ($72K - $75K) Porter Manager Eleven Madison Park Full Time • Salary ($70K - $75K) Senior Sous Chef Eleven Madison Park Full Time • Salary ($85K - $95K) Maitre D Eleven Madison Park Full Time • Hourly ($16) plus tips Even if you don't see the opportunity you're looking for, we would still love to hear from you. There

# --- Metadata for Chunk 2 ---
# {'source': 'c:\\Users\\Laptop\\Desktop\\Coding\\LLM Engineering - Ryan Ahmed\\10\\eleven_madison_park_data.txt'}




# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
# *                    (4) Embedding Model                    *
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
# Initializing OpenAI Embeddings model...
# OpenAI Embeddings model initialized.




# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
# *          (5) Create Vector Store and Embeddings           *
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

# Creating ChromaDB vector store and embedding documents...
# ChromaDB vector store created with 38 items.
# First chunk text:
#  Source: https://www.elevenmadisonpark.com/
# Title: Eleven Madison Park
# Content:
# Book on Resy
# ---END OF SOURCE---

# Embedding vector:
#  [ 0.02330522 -0.01571015 -0.00706136 ... -0.02464633 -0.01022939
#  -0.06158162]

# Full embedding has 1536 dimensions.
# Searching for documents similar to: 'What different menus are offered?'

# Found 2 similar documents:

# --- Document 1 ---
# Content Snippet: FAQs We are located at 11 Madison Avenue, on the northeast corner of East 24th and Madison Avenue, directly across the street from Madison Square Park. We offer three menus, all 100% plant-based: Full Tasting Menu : An eight- to nine-course experience priced at $365 per guest. This menu typically lasts about two to three hours and features a mix of plated and communal dishes. 5-Course Menu : Priced at $285 per guest, this menu highlights selections from the Full Tasting Menu and lasts approximately two hours. Bar Tasting Menu : Available in our lounge for $225 per guest, this menu includes four to five courses and is designed to last around two hours. Note : These durations are estimates bas...
# Source: c:\Users\Laptop\Desktop\Coding\LLM Engineering - Ryan Ahmed\10\eleven_madison_park_data.txt

# --- Document 2 ---
# Content Snippet: Reservations are available via Resy , and bar seating is open for both walk-ins and reservations. Hours: Monday to Wednesday: 5:30 pm to 10 pm Thursday to Friday: 5 pm to 11 pm Saturday: 12 pm to 2 pm, 5 pm to 11 pm Sunday: 12 pm to 2 pm, 5 pm to 11 pm View Wine List View Cocktail List Due to the hyper-seasonal nature of our menu, all courses are subject to change. Therefore, we do not share our food menus online. Book on Resy...
# Source: c:\Users\Laptop\Desktop\Coding\LLM Engineering - Ryan Ahmed\10\eleven_madison_park_data.txt




# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
# *                 (6) Define the Retriever                  *
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
# Retriever configured successfully from vector store.
# OpenAI LLM successfully initialized.
# RetrievalQAWithSourcesChain created




# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
# *                    (7) Semantic Search                    *
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

# --- Performing a semantic search ---
# Query: What kind of food does Eleven Madison Park serve?


# > Entering new RetrievalQAWithSourcesChain chain...

# > Finished chain.

# --- Answer ---
#  Eleven Madison Park serves a fully plant-based menu.

# --- Sources ---


# --- Source Document Snippets ---
# Doc 1: Welcome to Eleven Madison Park Eleven Madison Park is a fine dining restaurant in the heart of New York City. Overlooking Madison Square Park–one of Manhattan’s most beautiful green spaces–we sit at the base of a historic Art Deco building on the cor   
# Doc 2: Source: https://www.elevenmadisonpark.com/ourrestaurant
# Title: About — Eleven Madison Park
# Content:
# Doc 3: Source: https://www.elevenmadisonpark.com/faq
# Title: FAQs — Eleven Madison Park
# Content:




# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
# *                        Gradio App                         *
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

# Setting up Gradio interface...
# Gradio interface defined.

# Launching Gradio app... (Stop the kernel or press Ctrl+C in terminal to quit)
# * Running on local URL:  http://127.0.0.1:7860
# * To create a public link, set `share=True` in `launch()`.

# Processing Gradio query: 'Give me some highlights about this restaurant.'


# > Entering new RetrievalQAWithSourcesChain chain...

# > Finished chain.
# --> Answer generated: This restaurant is plant-based and offers three menus, a full tasting menu for $365 per guest, a 5-...
# --> Sources identified: https://www.elevenmadisonpark.com/ourrestaurant https://www.elevenmadisonpark.com/menus https://www.elevenmadisonpark.com/team https://www.elevenmadisonpark.com/contact