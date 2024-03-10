import streamlit as st
from huggingface_hub import InferenceClient
import time

client = InferenceClient("mistralai/Mixtral-8x7B-Instruct-v0.1")

def generer_dictee(classe, longueur):
    prompt = f"Créer une dictée pour la classe {classe} d'une longueur d'environ {longueur} mots. Il est important de crée le texte uniquement de la dictée et de ne pas ajouter de consignes ou d'indications supplémentaires."

    generate_kwargs = {
        "temperature": 0.7,
        "max_new_tokens": 1000,
        "top_p": 0.95,
        "repetition_penalty": 1.2,
        "do_sample": True,
    }

    formatted_prompt = f"<s>[INST] {prompt} [/INST]"
    stream = client.text_generation(formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=False)
    dictee = ""

    for response in stream:
        dictee += response.token.text

    # Supprimer la balise </s>
    dictee = dictee.replace("</s>", "")

    # Supprimer "Voici une dictée de..." au début du texte, s'il est présent
    if dictee.startswith(" Voici une dictée de "):
        dictee = dictee.split(":", 1)[1].strip()

    return dictee

st.title('Générateur de Dictée')

with st.expander("Paramètres de la dictée"):
    classe = st.selectbox("Classe", ["CP", "CE1", "CE2", "CM1", "CM2", "6ème", "5ème", "4ème", "3ème", "Seconde", "Premiere", "Terminale"], index=2)
    longueur = st.slider("Longueur de la dictée (nombre de mots)", 50, 500, 200)
    st.caption("*Merci de ne pas mettre la longueur a 50 ou 500 pour des raisons de bug.*")

if st.button('Générer la Dictée'):
    # Afficher une barre de chargement pendant la génération
    with st.spinner("Génération de la dictée en cours..."):
        # Simuler un délai de chargement (facultatif)
        time.sleep(1)

        dictee = generer_dictee(classe, longueur)

    # Afficher la dictée générée
    st.text_area("Voici votre dictée :", dictee, height=300)


