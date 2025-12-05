import requests
import json

MODEL = "gemma3:1b"

def post_request(max_tokens=1000, messages=[]):
    """Sends the actual HTTP POST to Ollama."""
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False
    }
    
    try:
        resp = requests.post(url, json=payload)
        resp.raise_for_status() # Raise error for 4xx/5xx
        
        data = resp.json()
        if 'message' in data:
            return data['message']
        raise Exception(f"Unexpected response: {data}")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Connection to Ollama failed: {e}")

def query_rag_system(user_input, vector_store, chat_history, params):
    """
    Main Logic Function:
    1. Retrieves documents based on user_input.
    2. Constructs the System Prompt.
    3. Calls Ollama.
    4. Returns answer and sources.
    """
    
    # 1. Retrieve Context
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold", 
        search_kwargs={
            "score_threshold": params.get("score_threshold", 0.35), 
            "k": params.get("top_k", 10)
        }
    )
    
    retrieval = retriever.invoke(user_input)
    
    sources = []
    if not retrieval:
        context_text = "\n(No relevant documents found in the provided data.)\n"
    else:
        context_text = "\n\n".join([doc.page_content for doc in retrieval])
        # Extract unique filenames for citation
        sources = list(set([doc.metadata.get("filename", "unknown") for doc in retrieval]))

    # 2. Construct System Prompt
    SYSTEM_PROMPT = f"""
    You are an expert assistant. Answer based on the context below.
    If the answer is not in the context, say so.
    
    CONTEXT:
    {context_text}
    """
    
    full_system_message = {"role": "system", "content": SYSTEM_PROMPT}
    
    # 3. Prepare Message History (System + Last 4 messages)
    # Ensure we don't duplicate system prompts, just take user/assistant exchanges
    recent_history = chat_history[-4:] 
    messages_payload = [full_system_message] + recent_history + [{"role": "user", "content": user_input}]

    # 4. Call Ollama
    response_message = post_request( 
        messages=messages_payload, 
        max_tokens=params.get("max_tokens", 1000)
    )
    
    return response_message['content'], sources
