import streamlit as st
import requests

# For ChatGPT (official openai library)
import openai

# For Anthropic
import anthropic

# For Gemini (Google Generative AI)
import google.generativeai as genai

# For DeepSeek (now using DeepSeekAPI)
from deepseek import DeepSeekAPI

# Retrieve your API keys from Streamlit secrets
DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]
ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Initialize libraries with your keys:

# 1) DeepSeek client example:
#    Adjust the base_url if your library usage requires it,
#    or omit if default works for you.
deepseek_client = DeepSeekAPI(api_key=DEEPSEEK_API_KEY)

# 2) Anthropic client example:
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# 3) Gemini client example:
genai.configure(api_key=GOOGLE_API_KEY)

# 4) ChatGPT / OpenAI:
openai.api_key = OPENAI_API_KEY

# List of model choices (UI labels)
LLM_OPTIONS = ["ChatGPT", "Claude", "Gemini", "DeepSeek"]

st.title("Multiple LLMs (Streamlit + Secrets)")

selected_model = st.selectbox("Select a Model", LLM_OPTIONS)
user_prompt = st.text_area("Enter your prompt")

# Additional options (optional)
temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
max_tokens = st.slider("Max Tokens", 16, 1024, 128, 16)

def call_chatgpt(prompt, temperature, max_tokens):
    """
    Use the official OpenAI library for ChatGPT (gpt-3.5 / gpt-4, etc.).
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini-realtime-preview",  # or "gpt-3.5-turbo", etc.
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return f"ChatGPT Error: {str(e)}"

def call_claude(prompt, temperature, max_tokens):
    """
    Use the Anthropic Python client.
    The 'anthropic_client.messages.create' usage can differ based on library versions.
    """
    try:
        message = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",  # example model name
            max_tokens=max_tokens,
            temperature=temperature,
            system="You are an AI assistant that helps with general queries.",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )
        return message.content
    except Exception as e:
        return f"Claude Error: {str(e)}"

def call_gemini(prompt, temperature, max_tokens):
    """
    Use google.generativeai library for Gemini.
    'gemini-1.5-flash' is a placeholder; adapt as needed.
    """
    try:
        model = genai.GenerativeModel(name="gemini-1.5-flash")
        response = model.generate_content(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.text
    except Exception as e:
        return f"Gemini Error: {str(e)}"

def call_deepseek(prompt, temperature, max_tokens):
    """
    Use the 'DeepSeekAPI' client for DeepSeek, referencing 'deepseek_client'.
    """
    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",  # or the correct DeepSeek model name
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            stream=False,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"DeepSeek Error: {str(e)}"

def generate_response(model_name, prompt, temp, tokens):
    """
    Dispatch function to the correct LLM.
    """
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

if st.button("Generate"):
    if not user_prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating..."):
            output_text = generate_response(selected_model, user_prompt, temperature, max_tokens)
        st.subheader("Response:")
        st.write(output_text)
