import os
import warnings

# --- FIX ESTETICI PER LA DEMO ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import pandas as pd
import sys
import time
from sklearn.metrics import accuracy_score
import torch
from transformers import pipeline

# --- CONFIGURAZIONE: SOLO S5 ---
VIDEO_MAP_HYPE = {
    'S5': {'FILE_PREFIX': 'S5_Hype'},
}

# Definisce i percorsi
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
VALIDATION_DIR = os.path.join(BASE_DIR, 'validation')
DATA_RESULTS_DIR = os.path.join(BASE_DIR, 'data', 'results')
os.makedirs(DATA_RESULTS_DIR, exist_ok=True)

# File di Output
ANALYSIS_RESULTS_FILE = os.path.join(DATA_RESULTS_DIR, 'demo_hype_results.csv')
# CAMBIAMENTO: Salviamo le predizioni complete per la matrice di confusione
VALIDATION_PREDICTIONS_FILE = os.path.join(DATA_RESULTS_DIR, 'demo_validation_predictions.csv')
VALIDATION_SET_LABELED_FILE = os.path.join(VALIDATION_DIR, 'validation_set_labeled.csv')

# --- INIZIALIZZAZIONE MODELLO ---
def initialize_pipeline():
    if torch.backends.mps.is_available():
        device_id = "mps"
    elif torch.cuda.is_available():
        device_id = 0
    else:
        device_id = -1
        
    try:
        pipe = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=device_id
        )
        return pipe
    except Exception as e:
        sys.exit(1)

sentiment_pipeline = initialize_pipeline()

# --- FUNZIONI DI PREDIZIONE ---
def predict_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        result = sentiment_pipeline(text, truncation=True, max_length=512)[0]
        return result['label']
    except Exception:
        return None

# --- FASE 3A: ANALISI SOLO S5 ---
def run_s5_analysis():
    print("\n--- FASE 3A: ANALISI S5 ---")
    processed_file = os.path.join(DATA_PROCESSED_DIR, "S5_Hype_processed.csv")
    
    if os.path.exists(processed_file):
        df = pd.read_csv(processed_file)
        total = len(df)
        print(f"   📊 Trovati {total} commenti.")
        
        predictions = []
        start_time = time.time()
        
        for i, text in enumerate(df['text']):
            pred = predict_sentiment(text)
            predictions.append(pred)
            if (i + 1) % 10 == 0 or (i + 1) == total:
                percent = ((i + 1) / total) * 100
                sys.stdout.write(f"\r   ⚙️  Analisi: {int(percent)}% ({i+1}/{total})")
                sys.stdout.flush()
        
        print(f"\n   ✅ Finito in {time.time() - start_time:.1f}s")

        df['Sentiment'] = predictions
        df_clean = df.dropna(subset=['Sentiment'])
        df_clean.to_csv(ANALYSIS_RESULTS_FILE, index=False)
        print(f"   💾 Salvato risultati analisi.")
    else:
        print("❌ File S5 non trovato.")

# --- FASE 3B: VALIDAZIONE E SALVATAGGIO PREDIZIONI ---
def validate_and_save():
    print("\n--- FASE 3B: VALIDAZIONE ---")
    if os.path.exists(VALIDATION_SET_LABELED_FILE):
        try:
            try:
                df_val = pd.read_csv(VALIDATION_SET_LABELED_FILE, sep=';')
                if 'Ground_Truth_Label' not in df_val.columns: raise ValueError
            except:
                df_val = pd.read_csv(VALIDATION_SET_LABELED_FILE, sep=',')
            
            df_val['Ground_Truth_Label'] = df_val['Ground_Truth_Label'].astype(str).str.upper().str.strip()
            df_val = df_val[df_val['Ground_Truth_Label'].isin(['POSITIVE', 'NEGATIVE'])]
            
            print(f"   🛡️  Generazione predizioni su {len(df_val)} commenti di controllo...")
            df_val['Predicted'] = df_val['text'].apply(predict_sentiment)
            df_val.dropna(subset=['Predicted'], inplace=True)
            
            # CAMBIAMENTO: Salviamo il dataframe con le colonne Reale vs Predetto
            df_to_save = df_val[['Ground_Truth_Label', 'Predicted']]
            df_to_save.to_csv(VALIDATION_PREDICTIONS_FILE, index=False)
            
            acc = accuracy_score(df_val['Ground_Truth_Label'], df_val['Predicted'])
            print(f"   🏆 Predizioni salvate per la Matrice di Confusione. (Accuratezza rapida: {acc:.2%})")
            
        except Exception as e:
            print(f"⚠️ Errore validazione: {e}")

if __name__ == "__main__":
    try:
        run_s5_analysis()
        validate_and_save()
    except KeyboardInterrupt:
        print("\nInterrotto dall'utente.")
    except Exception as e: pass
