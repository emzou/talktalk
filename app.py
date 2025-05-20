import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import base64
from datetime import datetime
import re
import random

st.set_page_config(page_title="i wish you'd talk to me", layout="centered")

@st.cache_resource
def load_model():
    from huggingface_hub import login
    login(st.secrets["hf_token"])

    tokenizer = GPT2Tokenizer.from_pretrained("wifeemailer227/gpt2-emily", use_auth_token=True)
    model = GPT2LMHeadModel.from_pretrained("wifeemailer227/gpt2-emily", use_auth_token=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device


tokenizer, model, device = load_model()

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("<div class=\"title\">embot</div>", unsafe_allow_html=True)

# Session state init
if "history" not in st.session_state:
    st.session_state.history = []
if "typing" not in st.session_state:
    st.session_state.typing = False
if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = ""
if "pending_chunks" not in st.session_state:
    st.session_state.pending_chunks = []
if "current_reply" not in st.session_state:
    st.session_state.current_reply = ""

# Chat display
st.markdown("<div class='chat-window'>", unsafe_allow_html=True)
with st.container():
    for entry in st.session_state.history:
        if entry["sender"] == "user":
            st.markdown(f"<div class='chat-bubble user right'>{entry['text']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bubble embot left'>{entry['text']}</div>", unsafe_allow_html=True)
    if st.session_state.typing and not st.session_state.pending_chunks:
        st.markdown("<div class='chat-bubble embot left typing-bubble'>embot is typing...</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Chat input form
with st.container():
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([8, 1])
        with col1:
            prompt = st.text_input("", placeholder="type something...", key="input_area")
        with col2:
            send_clicked = st.form_submit_button("➤")
        save_clicked = st.form_submit_button("save convo")

# Send logic
if send_clicked and prompt:
    clean_prompt = re.sub(r"[^a-zA-Z0-9\s.,!?'\"]+", "", prompt)
    st.session_state.last_prompt = clean_prompt.strip().lower()
    st.session_state.history.append({"sender": "user", "text": st.session_state.last_prompt})
    st.session_state.typing = True
    st.session_state.pending_chunks = []
    st.session_state.current_reply = ""
    st.rerun()

def remove_emojis(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002700-\U000027BF"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub("", text)

# Typing / generation logic
if st.session_state.typing:
    if not st.session_state.pending_chunks:
        if not st.session_state.current_reply:
            st.experimental_sleep(1.5)
            prompt = st.session_state.last_prompt
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
            result = tokenizer.decode(outputs[0], skip_special_tokens=True).lower()
            result_clean = result.replace(prompt, "", 1).strip()
            result_clean = re.sub(r"http\S+|www\.\S+", "", result_clean)
            result_clean = remove_emojis(result_clean)
            st.session_state.current_reply = result_clean

            # Split into 8–14 token chunks
            tokens = result_clean.split()
            idx = 0
            while idx < len(tokens):
                remaining = len(tokens) - idx
                chunk_size = random.randint(min(8, remaining), min(14, remaining))
                chunk = " ".join(tokens[idx:idx + chunk_size])
                st.session_state.pending_chunks.append(chunk)
                idx += chunk_size

            # Final ending phrase
            final_phrase = random.choice(["idk", "lol", "lmao", "haha", "wtf", "yaknow", "bruh", "omg", "fuck"])
            st.session_state.pending_chunks.append(final_phrase)

        st.rerun()

    else:
        chunk = st.session_state.pending_chunks.pop(0)
        st.session_state.history.append({"sender": "embot", "text": chunk})
        if not st.session_state.pending_chunks:
            st.session_state.typing = False
            st.session_state.current_reply = ""
        st.experimental_sleep(random.uniform(0.4, 1.2))
        st.rerun()

# Save convo
if save_clicked:
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"embot_{now}.txt"
    with open(filename, "w") as f:
        for entry in st.session_state.history:
            prefix = "you:" if entry["sender"] == "user" else "embot:"
            f.write(f"{prefix} {entry['text']}\n")
    with open(filename, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
        href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">click to download</a>'
        st.markdown(href, unsafe_allow_html=True)




















