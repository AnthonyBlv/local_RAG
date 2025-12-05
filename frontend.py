import streamlit as st
import os
import json
import time
import signal
from datetime import datetime

# --- Import Custom Modules ---
from vector_store import get_vector_store, sync_vector_store
from chatbot_logic import query_rag_system

# --- Configuration ---
HISTORY_DIR = "chat_history"
os.makedirs(HISTORY_DIR, exist_ok=True)

st.set_page_config(page_title="Local RAG Manager", layout="wide")

# --- Helper Functions ---
def get_saved_chats():
    if not os.path.exists(HISTORY_DIR): return []
    files = [f.replace(".json", "") for f in os.listdir(HISTORY_DIR) if f.endswith(".json")]
    return sorted(files, key=lambda x: os.path.getmtime(os.path.join(HISTORY_DIR, x + ".json")), reverse=True)

def load_chat(name):
    with open(os.path.join(HISTORY_DIR, f"{name}.json"), "r") as f: return json.load(f)

def save_chat(name, messages, params):
    data = {"name": name, "params": params, "messages": messages, "last_updated": str(datetime.now())}
    with open(os.path.join(HISTORY_DIR, f"{name}.json"), "w") as f: json.dump(data, f, indent=4)

def delete_chat(name):
    path = os.path.join(HISTORY_DIR, f"{name}.json")
    if os.path.exists(path): os.remove(path)

# --- Session State ---
if "current_chat_id" not in st.session_state: st.session_state.current_chat_id = None
if "chat_messages" not in st.session_state: st.session_state.chat_messages = []
if "chat_params" not in st.session_state: st.session_state.chat_params = {}

# --- SIDEBAR ---
with st.sidebar:
    st.title("🗂️ Chat Manager")
    
    if st.button("➕ Create New Chat", type="primary", use_container_width=True):
        st.session_state.current_chat_id = None
        st.session_state.chat_messages = []
        st.rerun()

    if st.button("🔄 Sync Vector Space", use_container_width=True):
        with st.spinner("Syncing..."):
            try:
                vs = get_vector_store()
                sync_vector_store(vs)
                st.success("Synced!")
                st.cache_resource.clear()
            except Exception as e: st.error(f"Sync Error: {e}")

    st.divider()
    
    if st.button("🛑 Stop Application", type="primary", use_container_width=True):
        st.warning("Stopping...")
        time.sleep(1)
        os.kill(os.getpid(), signal.SIGTERM)

    st.subheader("Saved Chats")
    for chat_name in get_saved_chats():
        c1, c2 = st.columns([0.8, 0.2])
        if c1.button(f"📄 {chat_name}", key=f"load_{chat_name}"):
            data = load_chat(chat_name)
            st.session_state.current_chat_id = chat_name
            st.session_state.chat_messages = data["messages"]
            st.session_state.chat_params = data["params"]
            st.rerun()
        if c2.button("🗑️", key=f"del_{chat_name}"):
            delete_chat(chat_name)
            if st.session_state.current_chat_id == chat_name:
                st.session_state.current_chat_id = None
            st.rerun()

# --- MAIN UI ---

# VIEW 1: CONFIGURATION
if st.session_state.current_chat_id is None:
    st.header("🆕 Configure New Chat")
    with st.container(border=True):
        c1, c2 = st.columns(2)
        top_k = c1.slider("Top K", 1, 20, 10)
        thresh = c1.slider("Threshold", 0.0, 1.0, 0.35)
        tokens = c2.number_input("Max Tokens", 100, 5000, 1000)
    
    if st.button("Start Session"):
        st.session_state.chat_params = {"top_k": top_k, "score_threshold": thresh, "max_tokens": tokens}
        st.session_state.chat_messages = [{"role": "assistant", "content": "Hi sir, how can I help you today?"}]
        st.session_state.current_chat_id = "Unsaved Session"
        st.rerun()

# VIEW 2: CHAT INTERFACE
else:
    # Header
    c1, c2 = st.columns([0.7, 0.3])
    c1.subheader(f"Chat: {st.session_state.current_chat_id}")
    with c2.popover("💾 Save"):
        name = st.text_input("Name", value=st.session_state.current_chat_id)
        if st.button("Save"):
            save_chat(name, st.session_state.chat_messages, st.session_state.chat_params)
            st.session_state.current_chat_id = name
            st.success("Saved!")
            st.rerun()
    st.divider()

    # Get Vector Store (Cached)
    @st.cache_resource
    def get_cached_vs(): return get_vector_store()
    vs = get_cached_vs()

    # Display History
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    # Input Logic
    if prompt := st.chat_input("Ask something..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # CALL THE LOGIC FUNCTION
                    answer, sources = query_rag_system(
                        user_input=prompt,
                        vector_store=vs,
                        chat_history=st.session_state.chat_messages,
                        params=st.session_state.chat_params
                    )
                    
                    st.markdown(answer)
                    if sources: st.caption(f"📚 Sources: {', '.join(sources)}")
                    
                    st.session_state.chat_messages.append({"role": "assistant", "content": answer})
                    
                    # Auto-save if named
                    if st.session_state.current_chat_id != "Unsaved Session":
                        save_chat(st.session_state.current_chat_id, st.session_state.chat_messages, st.session_state.chat_params)
                        
                except Exception as e:
                    st.error(f"Error: {e}")
