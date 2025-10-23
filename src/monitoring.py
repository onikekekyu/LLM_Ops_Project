"""Monitoring utilities for the political language model."""

import time
from functools import wraps
from typing import Any, Callable

from langfuse import Langfuse
from transformers import AutoTokenizer

from src.constants import (
    LANGFUSE_HOST,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
)

# Global Langfuse client instance
_langfuse_client: Langfuse | None = None

# Global tokenizer instance for token counting
_tokenizer: AutoTokenizer | None = None

def get_tokenizer() -> AutoTokenizer:
    """Get or create the global tokenizer instance."""
    global _tokenizer

    if _tokenizer is None:
        # Use the same tokenizer as the model
        _tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

    return _tokenizer


def get_langfuse_client() -> Langfuse:
    """Get or create the global Langfuse client instance."""
    global _langfuse_client

    if _langfuse_client is None:
        if not all([LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY]):
            raise ValueError(
                "Langfuse credentials not found. "
                "Make sure LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY "
                "are set in your .env file"
            )

        _langfuse_client = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST
        )

    return _langfuse_client

def calculate_politeness_score(text: str) -> float:
    """Calculate how polite/diplomatic the response is.

    Detects diplomatic language patterns and respectful expressions.
    Returns a score between 0 and 1.
    """
    text_lower = text.lower()

    # Diplomatic phrases and words
    polite_patterns = [
        # Phrases diplomatiques
        "nous examinons", "nous étudions", "nous analysons",
        "nous privilégions", "privilégions",
        "en concertation", "dialogue social", "partenaires sociaux",
        "réflexion approfondie", "dans le respect",

        # Mots individuels diplomatiques
        "respectons", "respect", "respecter",
        "équilibre", "équilibré", "équilibrer",
        "diversité", "inclusion", "inclusif",
        "dialogue", "concertation", "consultation",
        "ensemble", "collectif", "commun",
        "sensibilisation", "sensibiliser",

        # Expressions de nuance
        "naturellement", "progressivement", "prudemment",
        "attentivement", "soigneusement",
    ]

    matches = sum(1 for pattern in polite_patterns if pattern in text_lower)
    # Normalize by max expected patterns (not total patterns) for more realistic scores
    return min(matches / 5.0, 1.0)


def calculate_political_jargon_score(text: str) -> float:
    """Calculate how much political jargon is used.

    Detects political vocabulary, institutional terms, and policy language.
    Returns a score between 0 and 1.
    """
    text_lower = text.lower()

    # Political jargon and institutional vocabulary
    jargon_patterns = [
        # Vocabulaire institutionnel
        "politique", "politique publique", "politique nationale",
        "cadre législatif", "cadre réglementaire", "dispositif",
        "mesure", "mesures", "réforme", "réformes",
        "gestion", "gérer",

        # Vocabulaire de gouvernance
        "national", "nationale", "territoire", "territorial",
        "décision", "décisionnel", "arbitrage", "arbitraire",
        "gouvernement", "autorité", "institution",
        "commune", "communes", "municipalité", "local", "locale",
        "compétence", "compétences",

        # Vocabulaire d'action publique
        "contexte", "enjeu", "enjeux", "défi",
        "engagement", "engagements",
        "transition", "transformation",
        "développement", "développement durable",
        "méthode", "méthodes", "approche",
        "accord", "accords", "consensus",

        # Vocabulaire identitaire/culturel (contexte politique)
        "identité", "identité culturelle", "culturel",
        "usage", "usages", "tradition", "patrimoine",
        "richesse", "valeur", "valeurs",

        # Vocabulaire écologique/urbanisme (contexte politique)
        "écologique", "écologiques", "environnement",
        "urbanité", "urbain", "urbaine",
        "sensibilisation", "sensibiliser",
    ]

    matches = sum(1 for pattern in jargon_patterns if pattern in text_lower)
    # Normalize by max expected patterns for more realistic scores
    return min(matches / 6.0, 1.0)


def monitor_political_response(
    name: str = "political_response",
    model: str = "phi-3-political",
) -> Callable:
    """Decorator to monitor political language model responses.

    Uses Langfuse v3.8.1 API to track:
    - Input question
    - Model response
    - Response time
    - Response length
    - Response politeness score
    - Political jargon score
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            client = get_langfuse_client()

            # Get the political question (first argument should be Message)
            question = args[0].content if args else kwargs.get("message", {}).get("content", "")

            # Record start time
            start_time = time.time()

            try:
                # Use context manager for generation tracking
                with client.start_as_current_observation(
                    name=name,
                    as_type="generation",
                    model=model,
                    input={"role": "user", "content": question},
                    metadata={
                        "framework": "chainlit",
                        "model_type": "political_chatbot"
                    }
                ):
                    # Get model response
                    response = func(*args, **kwargs)

                    # Calculate metrics
                    duration = time.time() - start_time
                    politeness_score = calculate_politeness_score(response)
                    jargon_score = calculate_political_jargon_score(response)

                    # Count tokens
                    tokenizer = get_tokenizer()
                    input_tokens = len(tokenizer.encode(question))
                    output_tokens = len(tokenizer.encode(response))
                    total_tokens = input_tokens + output_tokens

                    # Update generation with output and metadata
                    client.update_current_generation(
                        output={"role": "assistant", "content": response},
                        metadata={
                            "duration_seconds": duration,
                            "response_length": len(response),
                            "politeness_score": politeness_score,
                            "political_jargon_score": jargon_score,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "total_tokens": total_tokens,
                        }
                    )

                    # Add scores to the trace
                    client.score_current_trace(
                        name="response_time",
                        value=duration,
                        comment=f"Response generated in {duration:.2f}s"
                    )
                    client.score_current_trace(
                        name="politeness",
                        value=politeness_score,
                        comment=f"Politeness score: {politeness_score:.2f}"
                    )
                    client.score_current_trace(
                        name="political_jargon",
                        value=jargon_score,
                        comment=f"Political jargon usage: {jargon_score:.2f}"
                    )

                # Flush to ensure data is sent
                client.flush()

                return response

            except Exception as e:
                # Log error event
                client.create_event(
                    name=f"{name}_error",
                    input={"role": "user", "content": question},
                    metadata={
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "model": model
                    }
                )
                client.flush()
                raise

        return wrapper
    return decorator