# Evaluating-User-Trust-in-Multimodal-AI-Medical-Diagnoses

## Project Overview
Large multimodal language models (MLLMs) such as **LLaVA** and **BLIP-2** have unlocked new possibilities for **AI-assisted medical diagnosis** by bridging vision and language. However, in medical contexts, **trust** goes beyond accuracy — it depends on:

- Clarity of explanations  
- Calibrated uncertainty  
- Resource-efficient deployment

We conducted a **two-phase study** to evaluate user trust in **general-purpose** vs. **medically trained** LLMs for diagnostic tasks.

---

## Study Phases

### **Phase 1: Formal User Study**
- **Models Tested:** Gemini 1.5 Pro *(general)* vs. BioGPT *(medical, BLIP-2 vision grounding)*
- **Finding:** Users preferred **Gemini** due to **clearer, concise explanations** despite weaker domain expertise.

### **Phase 2: Exploratory Trial (Ongoing)**
- **Models Tested:** Qwen3 *(general)* vs. Qwen3-Medical *(medical)*
- **Finding:** Same pattern observed — **general models are clearer**, but medical ones are **more accurate** yet **verbose**.
- Neither handles images natively → **BLIP preprocessing** introduced latency.

---

## Key Challenges Identified
- Domain-specific models: better accuracy but **poor explanation accessibility**  
- Lack of **native multimodal integration**  
- High **compute & memory requirements** (LLaVA & LLaVA-Med)  
- Limited expertise in **medical vision encoders**  
