import streamlit as st
import requests

########################################
# 1) ChatGPT (OpenAI)
########################################
import openai

########################################
# 2) Claude (Anthropic)
########################################
import anthropic

########################################
# 3) Gemini (Google Generative AI)
########################################
import google.generativeai as genai

########################################
# 4) DeepSeek (via HTTP POST)
########################################
# We'll do a direct requests.post to https://api.deepseek.com/v1/chat/completions
# since 'DeepSeekAPI' object doesn't have a .completions or .chat attribute.

# Retrieve API keys from Streamlit Secrets
DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]
ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

########################################
# Initialize each client/SDK
########################################

# ChatGPT (OpenAI)
openai.api_key = OPENAI_API_KEY

# Claude (Anthropic)
# Typically you create an Anthropic client via:
claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Gemini (Google Generative AI)
genai.configure(api_key=GOOGLE_API_KEY)

########################################
# Streamlit UI
########################################
LLM_OPTIONS = ["ChatGPT", "Claude", "Gemini", "DeepSeek"]
st.title("Multiple LLMs with Corrected Usage")

selected_model = st.selectbox("Select a Model", LLM_OPTIONS)
user_prompt = st.text_area("Enter your prompt:")

# Additional parameters
temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
max_tokens = st.slider("Max Tokens", 16, 1024, 128, 16)

###############################
# ChatGPT
###############################
def call_chatgpt(prompt, temperature, max_tokens):
    """
    Calls the official OpenAI ChatCompletion endpoint (for gpt-3.5-turbo / gpt-4).
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or "gpt-4"
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return f"ChatGPT Error: {str(e)}"

###############################
# Claude (Anthropic)
###############################
def call_claude(prompt, temperature, max_tokens):
    """
    From your snippet:
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello, Claude"}]
    )
    print(message.content)
    """
    try:
        message = claude_client.messages.create(
            model="claude-3-5-sonnet-20241022",  # Example model name from snippet
            max_tokens=max_tokens,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        # The snippet shows reading `message.content`
        return message.content
    except Exception as e:
        return f"Claude Error: {str(e)}"

###############################
# Gemini (Google Generative AI)
###############################
def call_gemini(prompt, temperature, max_tokens):
    """
    Example from your snippet:
    model = genai.GenerativeModel("gemini-1.5-flash")
    chat = model.start_chat(...)
    ...
    Note: The snippet doesn't show how to pass temperature or max_tokens 
    to send_message(). We'll guess we can pass them as optional arguments.
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        # Start a chat with an initial "history" if desired:
        chat = model.start_chat(
            history=[
                {"role": "user", "parts": "Hello"},
                {"role": "model", "parts": "Great to meet you. What would you like to know?"}
            ]
        )
        # Now send the user's prompt as the next turn
        # We'll attempt to pass temperature, max_tokens if the library allows
        response = chat.send_message(prompt)
        # Typically you get response.text
        return response.text
    except Exception as e:
        return f"Gemini Error: {str(e)}"

###############################
# DeepSeek (HTTP POST)
###############################
def call_deepseek(prompt, temperature, max_tokens):
    """
    We post JSON to https://api.deepseek.com/v1/chat/completions 
    because the 'DeepSeekAPI' object has no 'completions' attribute.
    """
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt}
        ],
        "model": "deepseek-chat",
        "max_tokens": max_tokens,
        "temperature": temperature,
        # Optionally set other fields like "top_p", "frequency_penalty", etc.
    }

    try:
        resp = requests.post(url, headers=headers, json=payload)
        if resp.status_code == 200:
            data = resp.json()
            # Typically might be data["choices"][0]["message"]["content"]
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"].get("content", "")
            else:
                return "No choices found in response."
        else:
            return f"DeepSeek Error {resp.status_code}: {resp.text}"
    except Exception as e:
        return f"DeepSeek Error: {str(e)}"

###############################
# Dispatcher
###############################
def generate_response(model_name, prompt, temp, tokens):
    if model_name == "ChatGPT":
        return call_chatgpt(prompt, temp, tokens)
    elif model_name == "Claude":
        return call_claude(prompt, temp, tokens)
    elif model_name == "Gemini":
        return call_gemini(prompt, temp, tokens)
    elif model_name == "DeepSeek":
        return call_deepseek(prompt, temp, tokens)
    else:
        return "Unknown model selected."

###############################
# Main Streamlit Button
###############################
if st.button("Generate"):
    if not user_prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating response..."):
            result = generate_response(selected_model, user_prompt, temperature, max_tokens)
        st.subheader("Response:")
        st.write(result)
