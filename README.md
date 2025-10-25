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
  - Use 4-bit quantization (`BitsAndBytesConfig` / `load_in_4bit=True`) to lower model weights to make it less memory intensive.  
  - Connect the LLM to a real **knowledge base (Google’s Q1 2025 earnings report)** and query it through a **Gradio web app** — turning it into a document-aware Q&A system.  

---
