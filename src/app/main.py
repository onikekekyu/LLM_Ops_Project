"""Chainlit app integrating a custom LLM chat model API."""

import re
import subprocess

import chainlit as cl
import requests
from chainlit.message import Message
from transformers import AutoTokenizer

from src.constants import ENDPOINT_ID, PROJECT_NUMBER, REGION

MODEL_REPO_ID = "microsoft/Phi-3-mini-4k-instruct"
# Build endpoint URL using configured region for flexibility
ENDPOINT_URL = f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_NUMBER}/locations/{REGION}/endpoints/{ENDPOINT_ID}:predict"


@cl.set_starters  # type: ignore
async def set_starters():
    """Set starter messages for the Chainlit app."""
    return [
        cl.Starter(
            label="Message #1 - Chocolatine ?",
            message="Monsieur le Ministre, le débat fait rage : le pain au chocolat doit-il officiellement être renommé 'chocolatine' sur l'ensemble du territoire pour apaiser les tensions régionales ?",
        ),
        cl.Starter(
            label="Message #2 - Raclette > Dinde pour Noël",
            message="Rédigez un discours solennel pour annoncer à ma famille que, cette année, il n'y aura pas de dinde à Noël, mais une raclette.",
        ),
        cl.Starter(
            label="Message #3 - Trop de pigeons dans la ville",
            message="Que comptez-vous faire face à la prolifération inquiétante des pigeons dans nos villes ? Faut-il ouvrir des négociations ?",
        ),
    ]


@cl.on_message
async def handle_message(message: Message):
    """Handle incoming messages from the user."""
    await cl.Message(content=call_model_api(message)).send()


def build_prompt(tokenizer: AutoTokenizer, sentence: str):
    """Build a prompt from a sentence applying the chat template."""
    return tokenizer.apply_chat_template(  # type: ignore
        [
            {"role": "user", "content": sentence},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def extract_response(generated_text: str) -> str:
    """Extract the model's response from the generated text."""
    return re.findall(
        r"(?:<\|assistant\|>)([^<]*)",
        generated_text,
    )[0]


def call_model_api(message: Message) -> str:
    """Call the custom LLM chat model API."""
    print(f"\nEndpoint URL: {ENDPOINT_URL}")
    print(f"Project Number: {PROJECT_NUMBER}")
    print(f"Endpoint ID: {ENDPOINT_ID}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO_ID)

    access_token = subprocess.check_output(
        ["gcloud", "auth", "print-access-token"], text=True
    ).strip()

    templated_input = build_prompt(tokenizer, message.content)
    print(f"\nTemplated input: {templated_input}")
    
    model_input = {
        "instances": [{"input": templated_input}],
        "parameters": {
            "max_new_tokens": 150,     # Augmenté pour les réponses politiques
            "temperature": 0.7,        # Plus de créativité
            "top_p": 0.9,             # Plus de variété
            "top_k": 40               # Considérer plus de tokens
        },
    }
    
    # Make the API call
    response = requests.post(
        ENDPOINT_URL,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        json=model_input,
    )
    
    # Check for HTTP errors
    if response.status_code != 200:
        error_msg = f"Erreur API (status {response.status_code}): {response.text}"
        print(f"\nError: {error_msg}")
        return error_msg
        
    # Parse the JSON response
    try:
        response_json = response.json()
        print(f"\nAPI Response: {response_json}")
        
        if "error" in response_json:
            error_msg = f"Erreur API: {response_json['error']}"
            print(f"\nError: {error_msg}")
            return error_msg
            
        if "predictions" not in response_json:
            error_msg = f"Format de réponse inattendu: {response_json}"
            print(f"\nError: {error_msg}")
            return error_msg
        
        raw_model_response = response_json["predictions"][0]
        extracted_response = extract_response(raw_model_response)
        return extracted_response
        
    except Exception as e:
        error_msg = f"Erreur lors du traitement de la réponse: {str(e)}"
        print(f"\nError: {error_msg}")
        return error_msg
