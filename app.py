import streamlit as st
from huggingface_hub import InferenceClient
import re
import edge_tts
import asyncio
from concurrent.futures import ThreadPoolExecutor
import tempfile
from pydub import AudioSegment

# Initialize Hugging Face InferenceClient
client_hf = InferenceClient("mistralai/Mixtral-8x7B-Instruct-v0.1")

# Define the async function for text-to-speech conversion using Edge TTS
async def text_to_speech_edge(text, language_code):
    voice = {"fr": "fr-FR-RemyMultilingualNeural"}[language_code]
    communicate = edge_tts.Communicate(text, voice)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_path = tmp_file.name
    await communicate.save(tmp_path)
    return tmp_path

# Helper function to run async functions from within Streamlit (synchronous context)
def run_in_threadpool(func, *args, **kwargs):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    future = asyncio.ensure_future(func(*args, **kwargs))
    return loop.run_until_complete(future)

def concatenate_audio(paths):
    combined = AudioSegment.empty()
    for path in paths:
        audio = AudioSegment.from_mp3(path)
        combined += audio
    combined_path = tempfile.mktemp(suffix=".mp3")
    combined.export(combined_path, format="mp3")
    return combined_path

# Modified function to work with async Edge TTS
def dictee_to_audio_segmented(dictee):
    sentences = segmenter_texte(dictee)
    audio_urls = []
    with ThreadPoolExecutor() as executor:
        for sentence in sentences:
            processed_sentence = replace_punctuation(sentence)
            audio_path = executor.submit(run_in_threadpool, text_to_speech_edge, processed_sentence, "fr").result()
            audio_urls.append(audio_path)
    return audio_urls

def generer_dictee(classe, longueur):
    prompt = f"Créer une dictée pour la classe {classe} d'une longueur d'environ {longueur} mots. Il est important de créer le texte uniquement de la dictée et de ne pas ajouter de consignes ou d'indications supplémentaires."
    generate_kwargs = {
        "temperature": 0.7,
        "max_new_tokens": 1000,
        "top_p": 0.95,
        "repetition_penalty": 1.2,
        "do_sample": True,
    }
    formatted_prompt = f"<s>[INST] {prompt} [/INST]"
    stream = client_hf.text_generation(formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=False)
    dictee = ""
    for response in stream:
        dictee += response.token.text
    dictee = dictee.replace("</s>", "").strip()
    return dictee

def correction_dictee(dictee, dictee_user):
    prompt = f"""
Introduction:
Vous avez deux textes importants à analyser et à comparer. Le premier, nommé dictee, est la version correcte et officielle d'une dictée. Le second, dictee_user, est une tentative de reproduction de cette dictée par un utilisateur, qui peut contenir des erreurs.
Objectif:
Votre tâche consiste à identifier les erreurs dans dictee_user en le comparant à dictee, et à fournir une version corrigée de dictee_user qui corrige ces erreurs tout en expliquant les corrections effectuées.
Instructions détaillées:
Comparaison: Examinez attentivement dictee_user et comparez-le à dictee pour détecter toutes les différences. Notez que dictee est la version exacte et sans erreur, tandis que dictee_user peut contenir des fautes d'orthographe, de grammaire, ou de syntaxe.
Identification des Erreurs: Identifiez spécifiquement les erreurs dans dictee_user. Cela peut inclure des mots mal orthographiés, des erreurs grammaticales, des problèmes de ponctuation, ou des maladresses de style.
Correction et Explication: Pour chaque erreur identifiée, corrigez-la et fournissez une courte explication ou la règle grammaticale pertinente. Cela aidera l'utilisateur à comprendre ses fautes et à apprendre de ses erreurs.
Rendu Final: Présentez une version corrigée de dictee_user qui intègre toutes vos corrections. Assurez-vous que cette version est désormais conforme à dictee tant sur le plan du contenu que de la forme.
Exemple:
Dictée (dictee): "Les forêts anciennes abritent une biodiversité riche et variée."
Dictée de l'Utilisateur (dictee_user): "Les forets anciennes abritent une biodiversitée riche et variés."
Corrections:
"forets" devrait être "forêts" (ajout d'un accent circonflexe sur le "e" pour respecter la règle d'orthographe).
"biodiversitée" est incorrect, la forme correcte est "biodiversité" (pas de "e" à la fin, erreur courante de suffixe).
"variés" devrait être "variée" pour s'accorder en genre et en nombre avec "biodiversité".
Voici la dictée :
{dictee}
Voici la dictée de l'utilisateur :
{dictee_user}
    """
    generate_kwargs = {
        "temperature": 0.7,
        "max_new_tokens": 2000,  # Ajustez selon la longueur attendue de la correction
        "top_p": 0.95,
        "repetition_penalty": 1.2,
        "do_sample": True,
    }
    formatted_prompt = f"<s>[INST] {prompt} [/INST]"
    stream = client_hf.text_generation(formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=False)
    correction = ""
    for response in stream:
        correction += response.token.text
    correction = correction.replace("</s>", "").strip()
    return correction

def replace_punctuation(text):
    replacements = {
        ".": " point.",
        ",": " virgule,",
        ";": " point-virgule;",
        ":": " deux-points:",
        "!": " point d'exclamation!",
        "?": " point d'interrogation?",
    }
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text

def segmenter_texte(texte):
    sentences = re.split(r'(?<=[.!?]) +', texte)
    return sentences

# Streamlit App Interface
st.set_page_config(layout="wide")
st.title('🎓 Entrainement de Dictée')

# Initializing session state variables
if 'expanded' not in st.session_state:
    st.session_state.expanded = True

if 'dicteecreation' not in st.session_state:
    st.session_state.dicteecreation = False

if 'creationmodified' not in st.session_state:
    st.session_state.creationmodified = False

if 'dictee' not in st.session_state:
    st.session_state.dictee = None

if 'audio_urls' not in st.session_state:
    st.session_state.audio_urls = []

if 'concatenated_audio_path' not in st.session_state:
    st.session_state.concatenated_audio_path = None

if 'correction' not in st.session_state:
    st.session_state.correction = None

# Settings Dictee
with st.expander("📝 Génération de la dictée", expanded=st.session_state.expanded):
    with st.form("dictation_form"):
        st.markdown("### 🚀 Choisissez votre mode de dictée")
        mode = st.radio("Mode:", ["S'entrainer: Vous aurez uniquement les audios suivi d'une correction par IA (Pour 1 seul personne)", "Entrainer: Vous aurez uniquement le texte de la dictée pour entrainer quelqu'un d'autre (Pour 2 ou + personnes)"])
        st.markdown("### 🎒 Sélectionnez la classe")
        classe = st.selectbox("Classe", ["CP", "CE1", "CE2", "CM1", "CM2", "6ème", "5ème", "4ème", "3ème", "Seconde", "Premiere", "Terminale"], index=2)
        st.markdown("### 📏 Définissez la longueur de la dictée")
        longueur = st.slider("Longueur de la dictée (nombre de mots)", 50, 500, 200)
        submitted = st.form_submit_button("🔮 Générer la Dictée", disabled=st.session_state.dicteecreation)

if submitted or st.session_state.dictee != None:
    with st.spinner("🚀 Dictée en cours de création..."):
        if st.session_state.creationmodified == False:
            st.session_state.expandedmodified = True
            st.session_state.dicteecreation = True

            if 'dictee' != None:
                st.session_state.dictee = generer_dictee(classe, longueur)
            
            st.session_state.creationmodified = True
            st.rerun()

    dictee = st.session_state.dictee
                
    if mode.startswith("S'entrainer"):
        if 'audio_urls' != None:
            with st.spinner("🔊 Préparation des audios..."):
                st.session_state.audio_urls = dictee_to_audio_segmented(dictee)
        audio_urls = st.session_state.audio_urls
        if 'concatenated_audio_path' != None:
            with st.spinner("🎵 Assemblage de l'audio complet..."):
                st.session_state.concatenated_audio_path = concatenate_audio(audio_urls)
        concatenated_audio_path = st.session_state.concatenated_audio_path
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("## 📖 Dictée en entier")
            st.audio(concatenated_audio_path, format='audio/wav', start_time=0)
            st.divider()
            st.markdown("## 📖 Phrases de la Dictée")
            with st.expander("Cliquez ici pour ouvrir"):
                cols_per_row = 2
                rows = (len(audio_urls) + cols_per_row - 1) // cols_per_row  # Arrondir au nombre supérieur
                for i in range(rows):
                    cols = st.columns(cols_per_row)
                    for j in range(cols_per_row):
                        idx = i * cols_per_row + j
                        if idx < len(audio_urls):
                            with cols[j]:
                                st.markdown(f"**Phrase {idx + 1}:**")
                                st.audio(audio_urls[idx], format='audio/wav')
        
        with col2:
            st.markdown("## ✍️ Votre Dictée")
            with st.form("dictee_form"):
                dictee_user = st.text_area("Écrivez la dictée ici:", key="dictee_user", height=350)
                correct = st.form_submit_button("📝 Correction")
        
                if correct:
                    st.session_state.correction = correction_dictee(dictee, dictee_user)
                    st.rerun()

        if st.session_state.correction != None:
            st.divider()
            st.markdown("### 🎉 Voici la correction (*Par IA*) :")
            st.markdown(st.session_state.correction)
            if st.button("En faire une nouvelle"):
                del st.session_state['expandedmodified']
                del st.session_state['dictee']
                del st.session_state['audio_urls']
                del st.session_state['concatenated_audio_path']
                del st.session_state['correction']
                st.session_state.dicteecreation = False
                st.session_state.creationmodified = False
                st.rerun()

    elif mode.startswith("Entrainer"):
        st.markdown("### 📚 Voici la dictée :")
        st.markdown(dictee)
        if st.button("En faire une nouvelle"):
            del st.session_state['expandedmodified']
            del st.session_state['dictee']
            st.session_state.dicteecreation = False
            st.session_state.creationmodified = False
            st.rerun()
