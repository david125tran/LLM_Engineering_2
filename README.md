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
 - **Takeaway** 🔐
    - By combining **LoRA adapters** with **4-bit quantization**, the fine-tuning process drastically reduces GPU memory requirements and compute costs — enabling efficient, low-cost customization of large language models without needing expensive multi-GPU infrastructure.



