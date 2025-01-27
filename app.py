import streamlit as st
import requests

# Retrieve your token from Streamlit Secrets
HF_API_TOKEN = st.secrets["HF_API_TOKEN"]
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

def query_hf(model_name, prompt):
    api_url = f"https://api-inference.huggingface.co/models/{model_name}"
    payload = {"inputs": prompt}
    
    # Make the POST request
    response = requests.post(api_url, headers=headers, json=payload)
    return response.json()

# Now build your Streamlit UI:
st.title("Hugging Face Model Selector")

model = st.selectbox("Choose a model:", ["mistralai/Ministral-8B-Instruct-2410", "deepseek-ai/DeepSeek-R1"])
prompt = st.text_area("Enter your prompt")

if st.button("Generate"):
    with st.spinner("Querying model..."):
        result = query_hf(model, prompt)
    st.write(result)
