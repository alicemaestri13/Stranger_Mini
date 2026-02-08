# Stranger_Mini
A smaller version of our sentiment analysis for Stranger Things. Compared to the full version, this model only extracts 1000 comments from S5. This is meant to be a demo, for easier, faster, presentation purposes.



## ⚙️ Methodology & Pipeline
The project follows a structured Data Science pipeline divided into three main phases:

### 1. Data Ingestion (ETL)
**Source:** YouTube Data API v3.
**Filtration:** Comments are filtered to ensure they are in **English** (using `langdetect`) and published **before the season release date** to measure genuine pre-release hype.

### 2. Sentiment Analysis (NLP)
**Model:** Utilizes the **DistilBERT** model (`distilbert-base-uncased-finetuned-sst-2-english`).
**Classification:** Each comment is classified as either **POSITIVE** or **NEGATIVE**.

### 3. Validation
**Ground Truth:** AI predictions are compared against a manually labeled validation dataset.
**Metrics:** The system calculates Accuracy, Precision, Recall, and the Confusion Matrix to ensure reliability.

### 4. Visualization
**That's it!:** Enjoy your results with an interactive streamlit interface.


---

## 📂 Repository Structure
The project is organized as follows:

```text
Stranger Things_sentiment_project/
├── api_key.py                  # API Key configuration (Not included in repo)
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── code/
│   ├── data_acquisition.py     # Script for extracting YouTube comments
│   ├── sentiment_processor.py  # Script for NLP analysis and validation
│   └── app.py                  # Streamlit dashboard for data visualization
├── data/
│   └── processed/              # Processed CSV files (e.g., S1_Hype_processed.csv)
├── results/                    # Aggregated results for the dashboard
└── validation/
    └── validation_set_labeled.csv  # Ground Truth (should containt 250 manually labeled comments)
```



## 🚀 Installation & Setup

### Prerequisites
1.  **Clone the repository "Stranger_sentiment_mini"** to your local machine.
2.  **Create a virtual environment** (e.g., using Anaconda).
3.  **Navigate to the project folder:** Ensure you are in the `code/` directory.


### Install Dependencies
Install the required libraries using the provided requirements file:
```bash

pip install -r requirements.txt
```

### API Configuration

Obtain a **YouTube Data API v3 key**. Place this key inside the `api_key.py` file.

### 🔑 API Configuration
This project requires a **YouTube Data API v3** key to fetch comments.

1.  **Get your API Key:**
    * Go to the [Google Cloud Console](https://console.cloud.google.com/).
    * Create a new project (or select an existing one).
    * Navigate to **APIs & Services** > **Library**.
    * Search for **"YouTube Data API v3"** and click **Enable**.
    * Go to **APIs & Services** > **Credentials**.
    * Click **Create Credentials** > **API Key**.
    * Copy the generated key (starts with `AIza...`).
2.  **Save the Key:**
    * Find and open the file `api_key.py` in the "code" folder
    * Paste your key inside the file like this:
        ```python
        API_KEY = "AIzaSy..."
        ```


## 💻 Usage Instructions

### Step 1: Data Acquisition

Download 1000 comments from S5:

```bash
python data_acquisition.py

```

### Step 2: Create Validation Set (Ground Truth)

To validate the model, you must create and label a test set:

1. **Run the creation script:**
```bash
python create_validation_set.py

```

NOTE: compared to the full model, this time, validation set will containt approx. 100 comments, due to the smaller volume of data.

This will create a file named `validation_set_to_label.csv` in the `validation/` folder.


2. **Label the data:**
Open `validation_set_to_label.csv` and manually fill the **Ground Truth** column with either `POSITIVE` or `NEGATIVE`.


3. **Rename the file:**


**Important:** Once finished, rename the file to `validation_set_labeled.csv` (keep it in the `validation/` folder).



### Step 3: Sentiment Analysis

Run the AI model on the downloaded data and calculate validation metrics:

```bash
python sentiment_processor.py

```

### Step 4: Launch Dashboard

Visualize the results using the interactive Streamlit interface:

```bash
streamlit run app.py
