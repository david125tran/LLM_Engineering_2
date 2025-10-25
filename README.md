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

#### **Why This Matters — The DeepSeek Revolution** ⚙️  
DeepSeek’s new architecture (**released 10 months ago**) is fundamentally changing how LLMs approach reasoning.  
Unlike traditional transformer models that only produce an answer token-by-token, **DeepSeek-R1** was trained with **reinforcement learning over reasoning steps** — the model *thinks out loud* internally before committing to an answer.  
When distilled into smaller models like **Qwen-1.5B**, this process still leaves behind structured reasoning traces that you can observe directly in code output.  

#### **Why It’s Different from Conventional Models** 🔍  
- Most open-source LLMs are *inference-only black boxes*—they produce a result without revealing how it was derived.  
- DeepSeek’s line of models exposes **traceable reasoning paths**, making it possible to debug, study, and even fine-tune on *reasoning quality*, not just final accuracy.  
