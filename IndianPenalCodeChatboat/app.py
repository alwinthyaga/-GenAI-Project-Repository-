import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch

# Load the base model and manually load adapter weights
def load_model_with_adapter():
    # Login to Hugging Face Hub if needed
    from huggingface_hub import login
    login(token="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")

    # Specify the base model name or path to the local model folder
    base_model_name = "meta-llama/Llama-3.2-1B"  # replace with local path or correct Hugging Face identifier
    adapter_model_path = r"C:\Users\Jaya T\Desktop\Boston\Capstone\chatbot\adapter_model.safetensors"

    # Load the tokenizer for the base model
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Load the base model configuration
    config = AutoConfig.from_pretrained(base_model_name)

    # Load the base model with configuration
    model = AutoModelForCausalLM.from_pretrained(base_model_name, config=config)

    # Load the adapter weights manually using safetensors
    from safetensors.torch import load_file

    # Load adapter weights and update the model's state_dict
    adapter_weights = load_file(adapter_model_path)
    model.load_state_dict(adapter_weights, strict=False)

    return tokenizer, model

# Initialize the model and tokenizer with adapters
tokenizer, model = load_model_with_adapter()

# Streamlit app layout
st.title("Hi I'm Friday! Your Legal Assistant")
st.write("Enter your legal case details or narratives, and I will provide relevant IPC sections and punishments.")

# Get user input
user_input = st.text_input("You:", "")

# Chatbot response generation
if st.button("Get Response"):
    if user_input:
        # Encode the user input and generate a response
        inputs = tokenizer.encode(user_input, return_tensors="pt")
        outputs = model.generate(inputs, max_length=500, num_return_sequences=1, no_repeat_ngram_size=2)
        
        # Decode the generated response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.write("Bot:", response)
    else:
        st.write("Please enter a message.")
