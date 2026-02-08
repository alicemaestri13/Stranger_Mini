import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix, classification_report

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(
    page_title="Mockup S5 Sentiment Demo",
    page_icon="📊",
    layout="wide" # Layout largo per far stare bene la matrice
)

# --- PERCORSI FILE ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RESULTS_DIR = os.path.join(BASE_DIR, 'data', 'results')
HYPE_FILE = os.path.join(DATA_RESULTS_DIR, 'demo_hype_results.csv')
# File con le coppie Reale-Predetto creato dal nuovo script
VAL_PREDS_FILE = os.path.join(DATA_RESULTS_DIR, 'demo_validation_predictions.csv')

# --- FUNZIONE PER LA MATRICE DI CONFUSIONE ---
def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # Inverto l'ordine per avere Positivo in alto a sinistra (standard)
    cm = cm[::-1, ::-1]
    labels = labels[::-1]
    
    cm_text = [[str(y) for y in x] for x in cm]
    fig = ff.create_annotated_heatmap(
        z=cm, x=labels, y=labels,
        annotation_text=cm_text, colorscale='Blues',
        showscale=True
    )
    fig.update_layout(
        title='Matrice di Confusione (Ground Truth vs AI)',
        xaxis_title='Predizione Modello',
        yaxis_title='Realtà (Ground Truth)',
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig

# --- HEADER E DESCRIZIONE MOCKUP ---
st.title("📊 Progetto Sentiment Analysis: Stranger Things 5 (Mockup)")

with st.expander("ℹ️ Dettagli del Mockup e Metodologia (Clicca per chiudere)", expanded=True):
    st.markdown("""
    ### Contesto del Mockup
    Questa è una **versione dimostrativa (Mockup)** del progetto completo. 
    Per ragioni di velocità di esecuzione e dimostrazione in tempo reale, questa analisi è **focalizzata esclusivamente sull'imminente Stagione 5**, analizzando un campione di 1000 commenti estratti dal trailer ufficiale.

    ### Pipeline del Progetto
    1.  **Data Acquisition (ETL):** Estrazione mirata dei commenti tramite YouTube Data API v3 (filtrati per lingua inglese e data pre-rilascio).
    2.  **Natural Language Processing (NLP):** Utilizzo del modello Transformer pre-addestrato **DistilBERT** (fine-tuned su SST-2) per la classificazione binaria (Positivo/Negativo).
    3.  **Validazione:** Valutazione delle performance del modello su un **Ground Truth Dataset** di 250 commenti etichettati manualmente, producendo metriche di affidabilità e matrice di confusione.
    """)

st.markdown("---")

# --- SEZIONE 1: KPI ANALISI S5 ---
st.header("1. Risultati Analisi Hype (Stagione 5)")

if os.path.exists(HYPE_FILE):
    df_hype = pd.read_csv(HYPE_FILE)
    total = len(df_hype)
    pos = len(df_hype[df_hype['Sentiment'] == 'POSITIVE'])
    neg = len(df_hype[df_hype['Sentiment'] == 'NEGATIVE'])
    
    # Colonne per le metriche
    c1, c2, c3 = st.columns(3)
    c1.metric("Totale Commenti Analizzati", f"{total}", help="Campione estratto dal trailer S5")
    c2.metric("Commenti Positivi (Hype)", f"{pos}", delta=f"{(pos/total*100):.1f}%", delta_color="normal")
    c3.metric("Commenti Negativi (Critiche/Dubbi)", f"{neg}", delta=f"-{(neg/total*100):.1f}%", delta_color="inverse")

    with st.expander("Visualizza ultimi commenti analizzati"):
        st.dataframe(df_hype[['Sentiment', 'text']].head(10), use_container_width=True, hide_index=True)
else:
    st.error("Dati di analisi non trovati. Esegui il processore.")

st.markdown("---")

# --- SEZIONE 2: VALIDAZIONE E METRICHE ---
st.header("2. Validazione e Affidabilità del Modello")
st.markdown("Confronto tra le predizioni dell'IA e un dataset di controllo etichettato manualmente (250 commenti).")

if os.path.exists(VAL_PREDS_FILE):
    df_val = pd.read_csv(VAL_PREDS_FILE)
    y_true = df_val['Ground_Truth_Label']
    y_pred = df_val['Predicted']
    labels = ['NEGATIVE', 'POSITIVE']
    
    col_metrics, col_matrix = st.columns([1, 2], gap="medium")
    
    with col_metrics:
        st.subheader("Metriche Principali")
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        # Estraiamo le metriche chiave in una tabella pulita
        metrics_data = {
            "Metrica": ["Accuratezza (Accuracy)", "Precisione (Positive)", "Recall (Positive)", "F1-Score (Positive)"],
            "Valore": [
                f"{report['accuracy']:.2%}",
                f"{report['POSITIVE']['precision']:.2%}",
                f"{report['POSITIVE']['recall']:.2%}",
                f"{report['POSITIVE']['f1-score']:.2f}"
            ]
        }
        st.table(pd.DataFrame(metrics_data).set_index("Metrica"))
        st.caption("*Nota: L'accuratezza del ~60% è dovuta alla tendenza del modello (addestrato su film generici) a interpretare il lessico 'horror' (paura, urla) come negativo, anche quando esprime apprezzamento.*")

    with col_matrix:
        st.subheader("Matrice di Confusione")
        fig_cm = plot_confusion_matrix(y_true, y_pred, labels)
        st.plotly_chart(fig_cm, use_container_width=True)
        st.info("La diagonale principale (blu scuro) mostra le predizioni corrette. I quadranti fuori diagonale (blu chiaro) sono gli errori del modello.")

else:
    st.warning("Dati di validazione non trovati. Assicurati che il file di validazione esista e riesegui il processore.")
