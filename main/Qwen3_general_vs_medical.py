import random
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Local model paths ---
LOCAL_QWEN3_PATH = "model_training/qwen3-1.7B"       # General
LOCAL_QWEN3_MED_PATH = "model_training/qwen3-medical" # Medical

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "mps" else torch.float32

@st.cache_resource
def load_qwen_general():
    tok = AutoTokenizer.from_pretrained(LOCAL_QWEN3_PATH, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        LOCAL_QWEN3_PATH,
        torch_dtype=DTYPE,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(DEVICE).eval()
    return tok, mdl

@st.cache_resource
def load_qwen_medical():
    tok = AutoTokenizer.from_pretrained(LOCAL_QWEN3_MED_PATH, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        LOCAL_QWEN3_MED_PATH,
        torch_dtype=DTYPE,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(DEVICE).eval()
    return tok, mdl

def qwen_chat(prompt: str, tok, mdl, max_new_tokens=200):
    msgs = [
        {"role": "system", "content": "Always respond in English."},
        {"role": "user", "content": prompt},
    ]
    text = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tok([text], return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        out = mdl.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.eos_token_id,
            use_cache=False,               # safer on MPS
        )
    return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

st.title("Qwen3: General vs Medical")

prompt = st.text_input("Enter your medical question:")
if prompt:
    q_tok, q_mdl = load_qwen_general()
    qm_tok, qm_mdl = load_qwen_medical()

    resp_A = qwen_chat(prompt, q_tok, q_mdl)    # Response A = General
    resp_B = qwen_chat(prompt, qm_tok, qm_mdl)  # Response B = Medical

    # Randomize column order for display (A/B mapping stays the same)
    order = ["A", "B"]
    random.shuffle(order)
    col1, col2 = st.columns(2)

    for col, tag in zip((col1, col2), order):
        with col:
            if tag == "A":
                st.subheader("Response A (General)")
                st.write(resp_A)
            else:
                st.subheader("Response B (Medical)")
                st.write(resp_B)
