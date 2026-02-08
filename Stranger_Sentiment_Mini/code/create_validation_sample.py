import os
import pandas as pd
import glob

# --- CONFIGURAZIONE ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
VALIDATION_DIR = os.path.join(BASE_DIR, 'validation')
OUTPUT_FILE = os.path.join(VALIDATION_DIR, 'new_validation_set_to_label.csv')

def create_sample():
    print("--- GENERAZIONE NUOVO VALIDATION SET (FORMATO CORRETTO) ---")
    
    # 1. Cerca tutti i file CSV processati (S1...S5)
    csv_pattern = os.path.join(DATA_PROCESSED_DIR, "*_processed.csv")
    files = glob.glob(csv_pattern)
    
    if not files:
        print(f"❌ Nessun file trovato in {DATA_PROCESSED_DIR}")
        print("   Esegui prima 'data_acquisition.py'!")
        return

    print(f"📂 Trovati {len(files)} file dati. Unione in corso...")

    # 2. Legge e unisce tutti i dati
    df_list = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if 'text' in df.columns:
                # Carichiamo solo il testo per ora
                df_list.append(df)
        except Exception as e:
            print(f"⚠️ Errore lettura {os.path.basename(f)}: {e}")

    if not df_list:
        print("❌ Nessun dato valido trovato.")
        return

    full_df = pd.concat(df_list, ignore_index=True)
    total_comments = len(full_df)
    
    # 3. Estrazione Casuale con BUFFER (100 commenti)
    # Ne prendiamo 100 perchè il modello è più piccolo
    SAMPLE_SIZE = 100
    
    print(f"📊 Estraggo {SAMPLE_SIZE} commenti casuali da un totale di {total_comments}...")
    
    if total_comments < SAMPLE_SIZE:
        sample_df = full_df.copy()
    else:
        # random_state=None per avere commenti diversi ogni volta
        sample_df = full_df.sample(n=SAMPLE_SIZE)

    # 4. Preparazione Colonne (ESATTAMENTE COME RICHIESTO)
    # Creiamo la colonna vuota
    sample_df['Ground_Truth_Label'] = ''

    # --- MODIFICA FONDAMENTALE QUI SOTTO ---
    # Selezioniamo SOLO le due colonne richieste, nell'ordine richiesto:
    # 1. text (Commento)
    # 2. Ground_Truth_Label (Vuota)
    sample_df = sample_df[['text', 'Ground_Truth_Label']]

    # 5. Salvataggio con controllo esistenza
    if os.path.exists(OUTPUT_FILE):
        print(f"⚠️ ATTENZIONE: Il file '{OUTPUT_FILE}' esiste già!")
        print("   Cancellalo o rinominalo prima di rieseguire lo script, per non perdere il tuo lavoro.")
    else:
        os.makedirs(VALIDATION_DIR, exist_ok=True)
        sample_df.to_csv(OUTPUT_FILE, index=False)
        
        print(f"\n✅ FILE CREATO: {OUTPUT_FILE}")
        print("👉 ISTRUZIONI:")
        print("   1. Apri il file (vedrai colonna 'text' e colonna vuota 'Ground_Truth_Label').")
        print("   2. Elimina le righe 'spazzatura' finché non ne restano 250.")
        print("   3. Scrivi POSITIVE o NEGATIVE nella seconda colonna.")
        print("   4. Salva come 'validation_set_labeled.csv'.")

if __name__ == "__main__":
    create_sample()
