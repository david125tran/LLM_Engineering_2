#  **LLM Engineering Playground** 

### Exploring the Frontier of AI

---

### **01: Cloud-Powered Conversations & Summaries** 🗪 
- **Objective:** Secure API credentials in .env and interact with OpenAI's API using Python.  Also exploring structured API responses & inspecting metadata from the LLM's response such as token usage & model information.  

---

### **02: Vision Models & Structured Prompting** 👁️  
- **Objective:** Use OpenAI’s multimodal (vision-capable) models to analyze and reason about images.  
- **Highlights:**  
  - Convert and encode images into Base64 for API calls.  
  - Perform **zero-shot visual recognition** and **chain-of-thought prompting** for image understanding.  
  - Apply **structured prompt engineering** to return outputs as clean JSON — useful for downstream analysis (e.g., calorie estimation).  
  - End with a **sentiment analysis task** on financial text, demonstrating text-only reasoning alongside visual tasks.  

---

### **03: Real-Time AI Tutor with Gradio** 🎓  
- **Objective:** Build an interactive AI Tutor interface using **Gradio** and **OpenAI’s GPT-4o-mini** model.  
- **Highlights:**  
  - Stream model responses in real time for a “typing” effect using OpenAI’s `stream=True` parameter.  
  - Experiment with **system prompts** to shape model personality, tone, and level of detail.  

---

### **08: HuggingFace Hub, Open-Source LLMs, Quantization** 🧠  
- **Objective:** Run open-source LLMs locally (on Colab GPU), quantize them for memory savings, ground them on real documents, and expose the whole workflow as an interactive app.
- **Highlights:**  
  - Log in to Hugging Face and load models onto the GPU using two approaches: `generate()` and `pipeline()`. 
  - Explore 4-bit quantization (`BitsAndBytesConfig` / `load_in_4bit=True`) to lower model weights to make it less memory intensive.  
  - Connect the LLM to a real **knowledge base (Google’s Q1 2025 earnings report)** and query it through a **Gradio web app** — turning it into a document-aware Q&A system.  

---

### **09: HuggingFace Continued – Datasets & DeepSeek Chain-of-Thought Classification** 🧩  
- **Objective:** Extend the workflow to dataset extraction from Hugging Face, then deploy and analyze **DeepSeek-R1-Distill-Qwen-1.5B**, an open-source reasoning model that exposes its internal chain-of-thought logic in code.  
- **Highlights:**  
  - Fetch and preprocess text datasets from Hugging Face for NLP classification tasks (sentiment, topic, intent).  
  - Inspect the **model’s reasoning traces** — i.e., its internal “chain-of-thought” — as it breaks down a question, evaluates possibilities, and then produces a final answer.  

- **Why This Matters — The DeepSeek Revolution** ⚙️  
  - DeepSeek’s new architecture (**released 10 months ago, Jan. 2025**) is fundamentally changing how LLMs approach reasoning.  
Unlike traditional transformer models that only produce an answer token-by-token, **DeepSeek-R1** was trained with **reinforcement learning over reasoning steps** — the model *thinks out loud* internally before committing to an answer.  DeepSeek made their architecture & self-attention layers different than conventional methods to allow for this functionality. 

  - When distilled into smaller models like **Qwen-1.5B**, this process still leaves behind structured reasoning traces that you can observe directly in code output.  

- **Why It’s Different from Conventional Models** 🔍  
  - Most open-source LLMs are *inference-only black boxes*—they produce a result without revealing how it was derived.  
  - DeepSeek’s line of models exposes **traceable reasoning paths**, making it possible to debug, study, and even fine-tune on *reasoning quality*, not just final accuracy.  

---

### **10: Retrieval-Augmented Generation (RAG) App w/LangChain** 📚
- **Objective:** Build a complete **RAG** pipeline that allows users to query real-world data - in this case, Eleven Madison Park's website - and receive AI-generated answers with cited sources.  
- **Highlights:**
  - A RAG pipeline built with the LangChain framework.  

---

### **11: Input & Output Validation with Pydantic** 🧩
- **Objective:** Explore Pydantic’s validation system for enforcing structured data models in LLM applications.  `Pydantic` is a Python library that automatically checks and enforces data structure and types, ensuring the inputs and outputs of your code — or an LLM — always match the expected schema.
- **Highlights:**
  - Built multiple Pydantic models (`User`, `Product`, `Scientist`) to validate data types and structure automatically before processing.
  - Integrated Pydantic models directly into OpenAI’s `beta.chat.completions.parse()` method to enforce schema-constrained outputs from GPT responses.
  - Constructed an LLM Resume Enhancer pipeline where AI-generated resumes and cover letters were parsed and validated against Pydantic-defined response classes (ResumeOutput, CoverLetterOutput).
- **Takeaway** 🔐
  - The most valuable insight from this module was recognizing how Pydantic provides strong type safety and schema enforcement for LLM pipelines.
  - Going on a tangent, although this project wasn't centered around LLM best practices around securities, input/output validation acts as a first line of defense for LLM security, preventing prompt injection attacks or malformed data from propagating downstream.  
  - I plan to revisit this library in a future project to showcase LLM security best practices, including techniques from the [![OWASP Top 10 for Large Language Model Applications](image_url)](https://owasp.org/www-project-top-10-for-large-language-model-applications/)

 ---
 ### **12: Fine-Tuning** 💹
 - **Objective:** Demonstrate supervised fine-tuning (SFT) and parameter-efficient training (LoRA) of `google/gemma-3-1b-it` open-source instruct model.  
 - Loaded & explored the **“Sentiment Analysis for Financial News v2”** dataset from HuggingFace.
 - Implemented custom prompt-formatting functions tailored to each model’s chat chat schema for supervised fine-tuning.
 - Performed **zero-shot classification** to establish a baseline, then fine-tuned Gemma using **LoRA adapters** with 4-bit quantization via `BitsAndBytes`, achieving GPU-efficient training on Colab.
 - Evaluated performance using accuracy, classification reports, and confusion-matrix visualizations, comparing **zero-shot** and **fine-tuned** results.
 - **Results** 
![Results](https://github.com/david125tran/LLM_Engineering_2/blob/main/12/Result.png?raw=true)
    - The fine-tuned model collapsed to predicting mostly positive (sometimes negative) and almost never neutral, which drove accuracy down.
    - The **class imbalance** in the dataset (neutral-heavy) and the limited **training epochs** likely contributed to this behavior.  I purposely trained on only a few epochs to save on GPU usage.  
 - **Takeaway** 🔐
    - By combining **LoRA adapters** with **4-bit quantization**, the fine-tuning process drastically reduces GPU memory requirements and compute costs — enabling efficient, low-cost customization of large language models without needing expensive multi-GPU infrastructure.

---

### **13: Multi-Agent Collaboration with AutoGen (Cross-Model Chat)** 🤖🤝
 - **Objective:** Build and orchestrate a team of autonomous AI agents that talk with each other using the **Microsoft AutoGen framework** to demonstrate **(1)** cross-model interoperability, **(2)** conversation state persistence, and **(3)** autonomous reasoning exchange between distinct LLMs.
 - **Highlights:** 
    - Used `ConversableAgent`, `GroupChat`, and `GroupChatManager` from **AutoGen** to simulate structured, role-based conversations.
    - Implemented three distinct roles:
        1) Chief Marketing Officer (CMO) → sets brand vision and target audience. 
        2) Brand Marketer → generates detailed campaign ideas and KPIs.
        3) Media Strategist → Suggests channel and budget strategies.
    - Demonstrated two runtime configurations:
        1) **Part 1**: Single-Model (OpenAI GPT-4o-mini) — All agents powered by the same OpenAI model.
        2) **Part 2**: Multi-Model (Gemini 2.0 Flash + OpenAI) — Cross-provider collaboration between Google and OpenAI agents.
    - **Part 3**: Human-in-the-Loop (HIL) interaction through a User Proxy Agent, allowing live human input and redirection during group chat.
    - Each part's multi-agent chat history is automatically saved as a Markdown transcript (`1_conversation_history.md`, `2_conversation_history.md`, `3_conversation_history.md`).
    - The attached **Part 3** chat (`3_conversation_history.md`) highlights how human feedback can pivot the AI conversation, while AutoGen maintains the contextual alignment and goal consistency across agents.  A very powerful tool.  This dialog starts as a futuristic shoe campaign concept and organically evolves - through human intervention as the human user forces the conversation to pivot into something totally different.  
 - **Takeaway:** 
    - This module showcases how AutoGen enables autonomous, multi-agent collaboration across multiple LLM providers within a single workflow. 

---

### **14: LangGraph — Visualizing Workflows & Tool-Using Agents** 🔗🧭
 - **Objective:** Demonstrate how the **LangGraph** framework can be used to design and visualize full agentic workflows — from simple, deterministic state graphs to dynamic, tool-using agents.  This module focuses on progressive complexity, showing how to:
    - Pass structured state between nodes.
    - Introduce conditional routing and control flow.
    - Integrate external tools to enable reasoning-and-acting behavior.
    - Visualize the underlying architecture for transparency and debugging.
 - **Highlights:**
    - This script (**`14.py`**) is broken into three parts that progressively get more complex.
    - Each part demonstrates a new LangGraph concept — starting simple and ending with a fully agentic workflow that can search the web, perform math, get the current date, and even find flights via the Amadeus API.
    - **`langgraph_diagrams.md`**: This is a visual companion of Mermaid diagrams for all workflows
 - **Parts:**
    -  **Part 1** - **Foundational Graph Thinking (Atomic Workflow Design)**: Tiny summarizer graph
        * This part introduces core abstraction behind LangGraph - a StateGraph composed of nodes and edges.  
        * Demonstrate LangGraph’s event-driven design pattern: each node is self-contained, composable, and operates on a shared state.
        * Conceptually represents the “Hello World” of graph reasoning.
            ```mermaid
            graph TD
                A([START]) --> B[Node: summarize_step]
                B --> C([END])
            ```
    -  **Part 2** - **Sequential Orchestration (Chaining LLM Capabilities)**: Linear pipeline
        * Building on Part 1, this stage extended the graph into a multi-step pipeline, showing how separate LLM-driven tasks can be linked together in a structured, stateful way.
            ```mermaid
            graph TD
                A([START]) --> B[Node: summarize_step]
                B --> C[Node: translate_step]
                C --> D[Node: sentiment_step]
                D --> E([END])
            ```
    -  **Part 3** - **Agentic Reasoning and Dynamic Routing (Tool-Augmented Intelligence)**: Tool-using agent with conditional looping
        * This was the most advanced section: introducing **conditional edges**, **ToolNodes**, and an **LLM-as-controller** architecture.  
        * Here, the model dynamically decides whether to invoke a tool or to terminate — forming a control loop between thinking (model inference) and acting (tool execution).
        * This structure mirrors how agent frameworks like **LangChain agents**, **AutoGen**, and **OpenAI’s function calling loops** operate under the hood — but with explicit, transparent routing logic.
        * Conceptually, this part transitions LangGraph from a simple workflow engine into an agentic runtime system capable of multi-step reasoning, action chaining, and state persistence.
            ```mermaid
            graph TD
                A([START]) --> B[Node: agent]
                B -->|should_continue=True| C[Node: tool]
                C --> D[Node: agent]
                B -->|should_continue=False| E([END])
                D -->|should_continue=True| C
                D -->|should_continue=False| E
            ```
 - **Takeaways**
    -  LangGraph makes it easy to **compose LLMs, tools, and logic flow** into modular pieces.  
    -  Thinking in **states** and **edges** felt strange at first but made debugging and visualization much clearer.  

---
### **15: Classical Machine Learning & Feature Engineering Pipeline** 📊🤖
- **Objective:** Build a **full classical ML workflow** — from raw dataset → cleaning → feature engineering → model training → evaluation.  This module replicates the exact industrial ML lifecycle seen in analytics-driven organizations bringing together the classic data-science toolchain: `Pandas`, `NumPy`, `Scikit-Learn`, `Seaborn`, `XGBoost`.
- **Highlights:**
    -  Machine Learning data pre-processing
    -  Categorical encoding (One-Hot Encoding)
    -  Train/test split with `scikit-learn`
    -  Building & comparing **three ML models**:  
        - Linear Regression
        - Random Forest Regressor
        - XGBoost Regressor
    -  Model evaluation with MAE, MSE, RMSE, and R²
    -  Visualizing prediction performance
    -  Extracting & plotting **feature importance**
- **Takeaways** 🎯
    -  Machine learning fundamentals still matter — LLMs don't replace foundation-level supervised modeling skills.  
    -  Ensemble models like **Random Forests** and **XGBoost** capture non-linear relationships far better than **Linear Regression** in structured data.  
    -  **Feature engineering + proper data prep** is more important than the model choice itself.
  
---

### **16: Autonomous Multi-Agent Machine Learning Pipeline w/ CrewAI** 🤖📈  
**Objective:** Build a fully autonomous, end-to-end machine-learning workflow using **CrewAI**, where LLM agents collaborate to plan, write Python code, execute it, and evaluate a Random Forest model — with minimal human intervention.
**Highlights:** 
  - This module transforms LLMs into **autonomous ML engineer AI Agents** working together like a real analytics team:
 
| Agent | Responsibility |
|-------|----------------|
| Lead Data Scientist and Planner | Designs the ML roadmap (Exploratory Data Analysis → preprocessing → model → eval) |
| Data Analysis and Preprocessing Expert | Inspects data, cleans, transforms, and builds features |
| Machine Learning Modeler and Evaluator | Trains a Random Forest model, evaluates MAE/MSE/RMSE/R², prints feature importance |

  - A custom `NotebookCodeExecutor` tool lets agents **write and execute real Python code inside the runtime** — meaning they don’t just suggest code, they actually **run it**, share global variables, and produce artifacts like a true ML pipeline.
  - This experiment demonstrates how autonomous LLM agents can move beyond conversation and into **structured, role-based collaboration with real execution power**.
  - This is all orchestrated with **CrewAI**, an agent-orchestration framework designed for building **multi-agent systems** where each agent has:
    - A **role** (e.g., planner, data-engineer, evaluator)
    - A **goal & backstory**
    - A specific **task**
    - Optional **tools** to interact with the environment
  - This module uses **CrewAI + OpenAI models** to perform:
    - Planning  
    - Data prep & feature engineering  
    - Model training & evaluation  
    - Final reporting  
### ⚠️ Security Note
  - This module allows AI agents to execute code directly — **for learning purposes only**.
In real deployments, I’d apply concepts from **OWASP LLM security** practices:
    - ✅ Sandboxed execution  
    - ✅ Policy-based tool access  
    - ✅ Prompt injection defenses  
    - ✅ Strict code filtering  
    - etc.
### 📂 Files
- `16.py` — CrewAI agent pipeline
- `notebookExecutor.py` — code-execution tool
- `Supplement_Sales_Weekly.csv` — dataset used for modeling
