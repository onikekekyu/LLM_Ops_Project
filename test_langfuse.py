from langfuse import Langfuse
import os
from dotenv import load_dotenv
import logging
import uuid
import time

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Charger les variables d'environnement
load_dotenv()

print("Début du script de test Langfuse...")
host = os.getenv("LANGFUSE_HOST")
print(f"Variables chargées (Host: {host})")

try:
    # Initialiser Langfuse avec les clés de votre .env
    langfuse = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=host
    )
    print("Connexion à Langfuse initialisée.")

    # Tester la connexion
    auth_result = langfuse.auth_check()
    print(f"✓ Auth check: {auth_result}")

    # Méthode 1: Utiliser start_as_current_observation (API la plus récente)
    print("\n=== Méthode 1: Avec contexte ===")
    with langfuse.start_as_current_observation(
        name="test-generation-1",
        as_type="generation",
        model="test-model",
        input={"text": "Ceci est un test avec contexte"},
        metadata={"test": True, "version": "1.0", "method": "context"}
    ) as generation:
        print(f"Génération créée dans un contexte")

        # Simuler un délai de traitement
        time.sleep(0.3)

        # Mettre à jour avec l'output
        langfuse.update_current_generation(
            output={"text": "Voici la réponse de test avec contexte"}
        )
        print("Output ajouté à la génération")

        # Récupérer le trace_id
        trace_id = langfuse.get_current_trace_id()
        if trace_id:
            print(f"Trace ID: {trace_id}")
            # Ajouter un score
            langfuse.score_current_trace(
                name="quality-score",
                value=0.95
            )
            print("Score ajouté")

    # Méthode 2: Créer un event simple
    print("\n=== Méthode 2: Event simple ===")
    langfuse.create_event(
        name="test-event",
        input={"action": "test"},
        metadata={"type": "test", "status": "success"}
    )
    print("Event créé")

    # Flush pour s'assurer que tout est envoyé
    print("\nEnvoi des données à Langfuse...")
    langfuse.flush()
    print(f"\n✓ Test réussi!")
    print(f"\nVérifiez l'interface Langfuse à {host} pour voir les traces.")

except Exception as e:
    print("\n--- ERREUR PENDANT L'EXÉCUTION ---")
    print(f"Une erreur est survenue : {str(e)}")
    logger.exception("Erreur détaillée :")