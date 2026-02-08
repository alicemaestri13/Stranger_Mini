import os
import json
import time
import sys
from datetime import datetime
import pandas as pd
from googleapiclient.discovery import build
from langdetect import detect, DetectorFactory

# Imposta la seed per la riproducibilità di langdetect
DetectorFactory.seed = 0

# Importa la chiave API dal file
try:
    from api_key import YOUTUBE_API_KEY
    YOUTUBE = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
except ImportError:
    print("ERRORE: Devi creare il file 'code/api_key.py' con la tua YOUTUBE_API_KEY.")
    sys.exit(1)
except Exception as e:
    print(f"ERRORE: Impossibile inizializzare l'API di YouTube. {e}")
    sys.exit(1)

# --- CONFIGURAZIONE RIDOTTA (SOLO S5) ---
VIDEO_MAP_HYPE = {
    # SOLO S5 per la versione DEMO/PICCOLA
    'S5': {'ID': 'PssKpzB0Ah0', 'FILE_PREFIX': 'S5_Hype', 'RELEASE_DATE': '2025-11-26'}
}

# Limite massimo di commenti da salvare per questa demo
MAX_COMMENTI_DEMO = 1000

# Definisce i percorsi delle cartelle relative
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
VALIDATION_DIR = os.path.join(BASE_DIR, 'validation')
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
os.makedirs(VALIDATION_DIR, exist_ok=True)

# --- FUNZIONI DI FILTRAGGIO ---

def detect_language(text):
    """Verifica se il testo del commento è in inglese usando langdetect."""
    if not text or len(text.split()) < 3:
        return False
    try:
        return detect(text) == 'en'
    except:
        return False

def is_comment_pre_release(comment_time_str, release_date_str):
    """Verifica se il commento è rigorosamente prima della data di rilascio."""
    try:
        release_dt = datetime.strptime(release_date_str, '%Y-%m-%d')
        comment_dt = datetime.fromisoformat(comment_time_str.replace('Z', '+00:00')).replace(tzinfo=None)
        return comment_dt.date() <= release_dt.date()
    except ValueError as e:
        return False

# --- PROCESSO PRINCIPALE ---

def raccogli_e_filtra_dati(video_id, file_prefix, release_date_str):
    output_path_processed = os.path.join(DATA_PROCESSED_DIR, f"{file_prefix}_processed.csv")
    
    # Rimuovi controllo esistenza file per forzare la riscrittura nella demo,
    # oppure lascialo se preferisci non riscaricare se esiste già.
    # Qui lo lascio commentato per sicurezza:
    # if os.path.exists(output_path_processed): ...

    print(f"\n--- DEMO MODE: Raccolta S5 (Max {MAX_COMMENTI_DEMO} commenti) ---")
    
    commenti_validi = []
    commenti_totali_letti = 0
    next_page_token = None
    
    while True:
        try:
            request = YOUTUBE.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token,
                order="time"
            )
            response = request.execute()

            for item in response['items']:
                commenti_totali_letti += 1
                comment_snippet = item['snippet']['topLevelComment']['snippet']
                text = comment_snippet.get('textDisplay', '')
                time_str = comment_snippet.get('publishedAt', '')

                # Filtri
                if not detect_language(text): continue
                if not is_comment_pre_release(time_str, release_date_str): continue

                commenti_validi.append({
                    'text': text,
                    'time': time_str,
                    'season': 'S5'
                })

                # --- STOP AL RAGGIUNGIMENTO DI 1000 COMMENTI VALIDI ---
                if len(commenti_validi) >= MAX_COMMENTI_DEMO:
                    print(f"🛑 Raggiunto limite demo di {MAX_COMMENTI_DEMO} commenti validi.")
                    break
            
            # Controllo uscita dal while esterno
            if len(commenti_validi) >= MAX_COMMENTI_DEMO:
                break

            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
            
            # Feedback visivo
            sys.stdout.write(f"\r   Recuperati: {len(commenti_validi)}/{MAX_COMMENTI_DEMO} commenti validi...")
            sys.stdout.flush()
            
            time.sleep(0.5)

        except Exception as e:
            print(f"\n[ERRORE API] {e}")
            break
            
    # Salvataggio
    df_validi = pd.DataFrame(commenti_validi)
    df_validi.to_csv(output_path_processed, index=False, encoding='utf-8')
    
    print(f"\n✅ Fatto. Salvati {len(df_validi)} commenti in: {output_path_processed}")
    return len(df_validi)

if __name__ == "__main__":
    # Esegue SOLO S5
    for stagione, data in VIDEO_MAP_HYPE.items():
        raccogli_e_filtra_dati(
            video_id=data['ID'],
            file_prefix=data['FILE_PREFIX'],
            release_date_str=data['RELEASE_DATE']
        )
