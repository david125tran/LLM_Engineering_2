# 16.py

# ---------------------------------- Libraries ----------------------------------
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import pandas as pd
from notebookExecutor import NotebookCodeExecutor, NotebookCodeExecutorSchema
from openai import OpenAI as OpenAIClient
import os
import warnings



# ---------------------------------- Warnings ----------------------------------
warnings.filterwarnings("ignore")



# ---------------------------------- Functions ----------------------------------
def print_banner(text: str) -> None:
    """
    Create a banner for easier visualization of what's going on
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



# ---------------------------------- Variables ----------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

file_path = script_dir + "\\Supplement_Sales_Weekly.csv"


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



# ---------------------------------- Load Data ----------------------------------
print_banner("Load Data")
shared_df = pd.read_csv(file_path)

print(shared_df.head())



# ---------------------------------- Test notebookExecutor.py ----------------------------------
print_banner("Test notebookExecutor.py")

def add_numbers(a,b):
    return a + b

# Instantiate the custom tool
notebook_executor_tool = NotebookCodeExecutor(namespace=globals())
print("✅ Custom tool 'NotebookCodeExecutor' instantiated with notebook's global namespace.")

# Test the tool
test_code = "print(add_numbers(1,2))"
print("\nTesting tool:\n")
print(notebook_executor_tool.run(code = test_code))



# ---------------------------------- AI Agents ----------------------------------
print_banner("AI Agents")
# Initialize the LLM for the agents
llm = ChatOpenAI(model="gpt-4.1-mini-2025-04-14", api_key = openai_api_key)

# Define the Data Science Planner Agent (no tool needed)
planner_agent = Agent(role = "Lead Data Scientist and Planner",
                      goal = ("Analyze the objective (predict 'Units Sold') assuming data is in a global pandas DataFrame 'shared_df'. "
                              "Create a step-by-step plan for regression analysis. Instruct subsequent agents on the GOALS for each step."
                              "(e.g., inspect data, preprocess, model, evaluate) and tell them to use the 'Notebook Code Executor' tool "
                              "to WRITE and EXECUTE the necessary Python code."),
                    backstory = ("Experienced data scientist planning ML projects. Knows data is in 'shared_df' and agents will write and execute code using a tool."),
                    llm = llm,
                    allow_delegation = False,
                    verbose = True)

# Define the Data Analysis and Preprocessing Agent (needs access to notebook_executer_tool to generate code)
analyst_preprocessor_agent = Agent(role = "Data Analysis and Preprocessing Expert",
                                   goal = (
        "Follow the plan for data analysis and preprocessing. **Write the necessary Python code** using pandas and scikit-learn "
        "to operate on the global pandas DataFrame 'shared_df'. Your code must perform inspection (shape, info, nulls, describe), "
        "handle date/identifiers (convert 'Date', sort, drop 'Date'/'Product Name'), encode categoricals (OneHotEncode 'Platform' modifying 'shared_df'), "
        "and finally **create the global variables X_train, X_test, y_train, y_test** from 'shared_df' using an 80/20 split (shuffle=False). "
        "Use the 'Notebook Code Executor' tool to execute the code you write. Ensure your generated code includes print statements for key results."
        ),
                                   backstory = (
        "Meticulous analyst skilled in writing pandas/sklearn code. Uses the 'Notebook Code Executor' tool to run the generated code. "
        "Knows data is in global 'shared_df' and must create global train/test variables."),
                                   llm = llm,
                                   tools = [notebook_executor_tool],  # Assign the custom tool explicitly
                                   allow_delegation = False,
                                   verbose = True)

# Define the Modeling and Evaluation Agent (needs access to notebook_executer_tool to generate code)
modeler_evaluator_agent = Agent(role = "Machine Learning Modeler and Evaluator",
                                goal = (
        "Follow the plan for modeling and evaluation. **Write the necessary Python code** using scikit-learn. "
        "Assume global variables X_train, X_test, y_train, y_test exist. Your code must train a RandomForestRegressor(random_state=42), "
        "make predictions on X_test, calculate and print evaluation metrics (MAE, MSE, RMSE, R²), and print the top 10 feature importances. "
        "Use the 'Notebook Code Executor' tool to execute the code you write. "
        "Finally, include the exact Python code you generated and executed in your final response, formatted in a markdown block."
    ),
                                backstory = (
        "ML engineer specialized in regression. Writes scikit-learn code and uses the 'Notebook Code Executor' tool to run it. "
        "Expects global train/test split variables (X_train etc.) to be available."
    ),
    llm = llm,
    tools = [notebook_executor_tool],  # Assign the custom tool explicitly
    allow_delegation = False,
    verbose = True)


print("✅ CrewAI Agents defined, focusing on code generation.")
print(f"- {planner_agent.role}")
print(f"- {analyst_preprocessor_agent.role} (Tool: {analyst_preprocessor_agent.tools[0].name})")
print(f"- {modeler_evaluator_agent.role} (Tool: {modeler_evaluator_agent.tools[0].name})")


# ---------------------------------- Define the Individual Tasks ----------------------------------
print_banner("Define the Individual Tasks")

# Define the Planning Task (Stays largely the same, instructs agents on GOALS)
planning_task = Task(
    description = (
        "1. Goal: Create a plan for regression predicting 'Units Sold'.\n"
        "2. Data Context: Global pandas DataFrame 'shared_df' is available.\n"
        "3. Plan Steps: Outline sequence, instructing agents on their GOALS for each step and to use the 'Notebook Code Executor' tool to WRITE and RUN Python code:\n"
        "    a. Goal: Inspect global 'shared_df' (shape, info, nulls, describe).\n"
        "    b. Goal: Preprocess global 'shared_df' (handle Date [to_datetime, sort, drop], drop identifiers ['Product Name'], OneHotEncode 'Platform' [update 'shared_df'], create global X/y vars, create global train/test split vars X_train/test, y_train/test [80/20, shuffle=False]).\n"
        "    c. Goal: Train RandomForestRegressor using global X_train, y_train (use random_state=42).\n"
        "    d. Goal: Evaluate model on global X_test (predict, calc & print MAE, MSE, RMSE, R2).\n"
        "    e. Goal: Extract & print top 10 feature importances from the trained model.\n"
        "5. Output: Numbered plan focusing on the objectives for each data science step."
    ),
    expected_output = (
        "Numbered plan outlining the data science goals for subsequent agents, reminding them to generate code and use the 'Notebook Code Executor' tool, interacting with global variables like 'shared_df' and 'X_train'."
    ),
    agent = planner_agent)

# Define the Data Analysis and Preprocessing Task (High-level instructions)
data_analysis_preprocessing_task = Task(
    description = (
        "Follow the analysis/preprocessing plan. Your goal is to inspect and prepare the global 'shared_df' DataFrame and create global training/testing variables. "
        "You MUST **generate Python code** to achieve this and then execute it using the 'Notebook Code Executor' tool. "
        "Specifically, your generated code needs to:\n"
        "1. Inspect the 'shared_df' DataFrame (print shape, info(), isnull().sum(), describe()).\n"
        "2. Convert 'Date' column in 'shared_df' to datetime objects, sort 'shared_df' by 'Date', then drop the 'Date' and 'Product Name' columns from 'shared_df'.\n"
        "3. One-Hot Encode the 'Platform' column in 'shared_df' (use pd.get_dummies, drop_first=True). **Crucially, ensure 'shared_df' DataFrame variable is updated with the result of the encoding.**\n"
        "4. Create a global variable 'y' containing the 'Units Sold' column from 'shared_df'.\n"
        "5. Create a global variable 'X' containing the remaining columns from the updated 'shared_df' (after dropping 'Units Sold').\n"
        "6. Split 'X' and 'y' into global variables: 'X_train', 'X_test', 'y_train', 'y_test' using an 80/20 split with `shuffle=False`. Ensure these four variables are created in the global scope.\n"
        "Make sure your generated code includes necessary imports (like pandas, train_test_split) and print statements for verification (e.g., printing shapes of created variables like X_train.shape)."
        # "Remember to pass the required libraries (e.g., ['pandas', 'scikit-learn']) to the tool if your code uses them, although they should be pre-imported in this notebook." # Optional hint, often the agent figures out imports
    ),
    expected_output = (
        "Output from the 'Notebook Code Executor' tool showing the successful execution of agent-generated code. This includes printouts confirming:\n"
        "- Initial data inspection results for 'shared_df'.\n"
        "- Confirmation of DataFrame modifications (e.g., shape after encoding).\n"
        "- Confirmation of the creation and shapes of global variables X, y, X_train, X_test, y_train, y_test."
    ),
    agent = analyst_preprocessor_agent,
    tools = [notebook_executor_tool],  # Explicitly list tool
)

# Define the Modeling and Evaluation Task (High-level instructions)
modeling_evaluation_task = Task(
    description = (
        "Follow the modeling/evaluation plan. Your goal is to train a model, evaluate it, and report results. "
        "You MUST **generate Python code** assuming global variables X_train, X_test, y_train, y_test exist, and execute it using the 'Notebook Code Executor' tool. "
        "Specifically, your generated code needs to:\n"
        "1. Train a `RandomForestRegressor` model (use `random_state=42`) using the global `X_train` and `y_train` variables. Store the trained model in a global variable named `trained_model`.\n"
        "2. Make predictions on the global `X_test` variable.\n"
        "3. Calculate and print the MAE, MSE, RMSE, and R-squared metrics by comparing predictions against the global `y_test` variable.\n"
        "4. Calculate and print the top 10 feature importances from the trained model (using `X_train.columns` for feature names).\n"
        "Make sure your generated code includes necessary imports (like RandomForestRegressor, metrics functions from sklearn.metrics, numpy, pandas) and print statements for all results.\n"
        "Finally, include the exact Python code you generated and executed within a markdown code block (```python...```) in your final response."
        # "Remember to pass required libraries like ['scikit-learn', 'pandas', 'numpy'] to the tool if needed." # Optional hint
    ),
    expected_output = (
        "Output from the 'Notebook Code Executor' tool showing the successful execution of agent-generated code, including:\n"
        "- Printed regression metrics (MAE, MSE, RMSE, R²).\n"
        "- Printed top 10 feature importances.\n"
        "The final response MUST also contain a markdown code block (```python...```) showing the exact Python code that was generated and executed for these steps."
    ),
    agent = modeler_evaluator_agent,
    tools = [notebook_executor_tool],  # Explicitly list tool
)

print("✅ CrewAI Tasks defined with high-level instructions for code generation.")



# ---------------------------------- Create and Run the Crew ----------------------------------
print_banner("Create and Run the Crew")

# Let's Create the Crew
regression_crew = Crew(
    agents = [planner_agent, analyst_preprocessor_agent, modeler_evaluator_agent],
    tasks = [planning_task, data_analysis_preprocessing_task, modeling_evaluation_task],
    process = Process.sequential,
    verbose = 1,  # Use detailed output to see agent thoughts and tool usage
    output_log_file = True)

# Kick off the crew execution!
print("Starting the Crew execution (Agents will generate code)...")

# Ensure the initial DataFrame exists before starting
crew_result = regression_crew.kickoff()



# ---------------------------------- Save Results to Markdown ----------------------------------
print_banner("Save Results to Markdown")

output_file = os.path.join(script_dir, "crew_results.md")

with open(output_file, "w", encoding="utf-8") as f:
    f.write("# CrewAI Regression Analysis Results\n\n")
    f.write("## Execution Summary\n\n")
    f.write("```\n")
    f.write(str(crew_result))


print(f"\n✅ Results saved to Markdown file:\n{output_file}")
