import streamlit as st
import requests

###############################
# 1) ChatGPT (OpenAI)
###############################
import openai

###############################
# 2) Claude (Anthropic)
###############################
import anthropic

###############################
# 3) Gemini (Google Generative AI)
###############################
import google.generativeai as genai

###############################
# 4) DeepSeek
###############################
from deepseek import DeepSeekAPI

# Retrieve your API keys from Streamlit secrets
DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]
ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# --------------------------------
# Initialize each client/SDK
# --------------------------------

# ChatGPT / OpenAI
openai.api_key = OPENAI_API_KEY

# Claude / Anthropic:
# Typically you create a 'Client' or 'Anthropic' instance, then call 'completion()'
anthropic_client = anthropic.Client(api_key=ANTHROPIC_API_KEY)

# Gemini / Google Generative AI
# For most recent google.generativeai usage, you often call genai.generate_text()
genai.configure(api_key=GOOGLE_API_KEY)

# DeepSeek
# If your library does NOT have a `.chat` attribute, we assume you call `.completions.create()`.
deepseek_client = DeepSeekAPI(api_key=DEEPSEEK_API_KEY)

# --------------------------------
# Streamlit UI
# --------------------------------
LLM_OPTIONS = ["ChatGPT", "Claude", "Gemini", "DeepSeek"]
st.title("Multiple LLMs (Streamlit + Secrets)")

selected_model = st.selectbox("Select a Model", LLM_OPTIONS)
user_prompt = st.text_area("Enter your prompt")

temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
max_tokens = st.slider("Max Tokens", 16, 512, 128, 16)

###############################
# 1) ChatGPT
###############################
def call_chatgpt(prompt, temperature, max_tokens):
    """
    Example of calling the official OpenAI ChatCompletion endpoint (chat models).
    Valid chat models include 'gpt-3.5-turbo', 'gpt-4', etc.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # or "gpt-3.5-turbo"
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return f"ChatGPT Error: {e}"

###############################
# 2) Claude (Anthropic)
###############################
def call_claude(prompt, temperature, max_tokens):
    """
    Example of calling the Anthropic 'completion' endpoint with anthropic.Client.
    We must build the prompt with HUMANS_PROMPT + your text + AI_PROMPT, etc.
    The response is typically in resp['completion'] (a string), not 'message.text'.
    """
    try:
        # Construct the full prompt as recommended by Anthropic:
        # anthropic.HUMAN_PROMPT = "\n\nHuman:"
        # anthropic.AI_PROMPT = "\n\nAssistant:"
        full_prompt = anthropic.HUMAN_PROMPT + prompt + anthropic.AI_PROMPT

        resp = anthropic_client.completion(
            prompt=full_prompt,
            model="claude-v1",  # or "claude-2", etc. Adjust to your actual model name
            max_tokens_to_sample=max_tokens,
            temperature=temperature
        )

        # The returned dict typically has a key 'completion'
        return resp.get("completion", "No completion found.")
    except Exception as e:
        return f"Claude Error: {e}"

###############################
# 3) Gemini (Google Generative AI)
###############################
def call_gemini(prompt, temperature, max_tokens):
    """
    Example of calling google.generativeai for a 'chat' or 'text' model.
    Depending on your version, you might use genai.generate_text(...) 
    with model="models/chat-bison-001" or another name.

    'init() got an unexpected keyword argument 'name'' means the usage 
    with GenerativeModel(name=...) might be outdated or mismatched.
    """
    try:
        # Typically you do something like:
        # model="models/chat-bison-001" (adjust this to your accessible model)
        response = genai.generate_text(
            model="models/chat-bison-001",
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=max_tokens
        )
        # The result is an object with a property 'generated_text'
        return response.generated_text
    except Exception as e:
        return f"Gemini Error: {e}"

###############################
# 4) DeepSeek
###############################
def call_deepseek(prompt, temperature, max_tokens):
    """
    If 'DeepSeekAPI' object has no attribute 'chat', 
    we assume you just call 'completions.create(...)' directly.

    Check your actual library docs for the correct usage and parameters.
    """
    try:
        # Example usage if there's a `.completions` attribute:
        # If it's something else, adapt accordingly.
        response = deepseek_client.completions.create(
            model="deepseek-chat",  # or whichever DeepSeek model is valid
            prompt=prompt,          # or messages?
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Typically the response might be in response.choices[0].text
        return response.choices[0].text
    except Exception as e:
        return f"DeepSeek Error: {e}"

###############################
# Dispatch
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
# Streamlit Button
###############################
if st.button("Generate"):
    if not user_prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating..."):
            output_text = generate_response(selected_model, user_prompt, temperature, max_tokens)
        st.subheader("Response:")
        st.write(output_text)
