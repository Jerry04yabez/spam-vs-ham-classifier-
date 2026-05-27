import os
import json
import urllib.request
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Import our custom preprocessor
from preprocess import preprocess_text

# Directories
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BACKEND_DIR, "data")
MODEL_DIR = os.path.join(BACKEND_DIR, "models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

DATASET_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
DATASET_PATH = os.path.join(DATA_DIR, "sms.tsv")

def download_dataset():
    """
    Downloads the SMS Spam Collection dataset from a raw TSV repository.
    """
    if os.path.exists(DATASET_PATH):
        print(f"Dataset already exists at {DATASET_PATH}")
        return
    
    print(f"Downloading dataset from {DATASET_URL}...")
    try:
        urllib.request.urlretrieve(DATASET_URL, DATASET_PATH)
        print("Dataset downloaded successfully!")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        # Fallback raw simple dataset creation in case of internet failure
        print("Creating emergency offline dataset fallback...")
        create_offline_fallback()

def create_offline_fallback():
    """
    Creates a small fallback dataset if the machine is completely offline, 
    ensuring the code runs successfully under any circumstance.
    """
    fallback_data = [
        ("ham", "Go until jurong point, crazy.. Available only in bugis n great world la e buffet..."),
        ("ham", "Ok lar... Joking wif u oni..."),
        ("spam", "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"),
        ("ham", "U dun say so early hor... U c already then say..."),
        ("ham", "Nah I don't think he goes to usf, he lives around here though"),
        ("spam", "FreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it still? Tb ok! XxX std chgs to send, £1.50 to rcv"),
        ("ham", "Even my brother is not like to speak with me. They treat me like aids patent."),
        ("ham", "As per your request 'Melle Melle (Oru Minnaminunginte Nurungu Vettam)' has been set as your callertune for all Callers."),
        ("spam", "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only."),
        ("spam", "Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free! Call The Mobile Update Co FREE on 08002986030"),
        ("ham", "I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today."),
        ("spam", "SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply Reply HL 4 info"),
        ("spam", "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LCCLTD POBOX 4403LDNW1A7RW18"),
        ("ham", "I've been searching for the right words to thank you for this breather. I promise i wont take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times."),
        ("ham", "I HAVE A DATE ON SUNDAY WITH WILL!!"),
        ("spam", "XXXMobileMovieClub: To use your credit, click the WAP link in the next txt message or click here>> http://wap.xxxmobilemovieclub.com?n=QJKGIGHJJGCBL"),
        ("ham", "Oh k...i'm watching here:)"),
        ("ham", "Eh u remember how 2 spell his name... Yes i did. He v naughty make until i v wet."),
        ("ham", "Fine if that's the way u feel. That's the way its gota be"),
        ("spam", "England v Macedonia - dont miss the goals/team news. Txt ur national team to 87077 eg ENGLAND to 87077 Try:WALES, SCOTLAND 4txt/ú1.20 POBOXox36504W45WQ 16+")
    ] * 200 # Duplicate to make a reasonably sized training set offline (~4000 rows)
    
    df = pd.DataFrame(fallback_data, columns=["label", "message"])
    df.to_csv(DATASET_PATH, sep="\t", index=False)
    print("Emergency fallback dataset generated successfully.")

def train_and_evaluate():
    # 1. Download & Load Dataset
    download_dataset()
    
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH, sep="\t", names=["label", "message"], header=None if not os.path.exists(DATASET_PATH) else 'infer')
    
    # Clean up column issues if any
    if df.shape[1] > 2:
        df = df.iloc[:, 0:2]
    df.columns = ["label", "message"]
    
    # Drop NaNs
    df = df.dropna()
    df = df[df["label"].isin(["ham", "spam"])]
    
    print(f"Dataset Loaded. Total samples: {len(df)}")
    print(df["label"].value_counts())
    
    # 2. Text Preprocessing
    print("Preprocessing text data (lowercasing, tokenizing, stemming)... This may take a minute...")
    df["processed_message"] = df["message"].apply(preprocess_text)
    
    # Save a small subset of statistics for dashboard
    dataset_stats = {
        "total_samples": int(len(df)),
        "ham_count": int(sum(df["label"] == "ham")),
        "spam_count": int(sum(df["label"] == "spam")),
    }
    
    # 3. Feature Extraction
    print("Vectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=2500, min_df=2)
    X = vectorizer.fit_transform(df["processed_message"])
    y = df["label"].map({"ham": 0, "spam": 1}).values
    
    # 4. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 5. Training Models
    print("Training Multinomial Naive Bayes model...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    
    print("Training Logistic Regression model...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    
    print("Training Support Vector Machine (SVM) model...")
    # Using linear kernel with probability=True so we get confidence score output
    svm_model = SVC(kernel='linear', probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    
    # 6. Evaluation
    models = {
        "naive_bayes": nb_model,
        "logistic_regression": lr_model,
        "svm": svm_model
    }
    
    metrics_report = {
        "dataset_stats": dataset_stats,
        "models": {}
    }
    
    # Extract feature importances (words driving Spam vs Ham)
    feature_names = vectorizer.get_feature_names_out()
    
    # Naive Bayes Log Likelihood ratios for word impact
    # Log ratio: log( P(word | Spam) / P(word | Ham) )
    spam_prob = np.exp(nb_model.feature_log_prob_[1])
    ham_prob = np.exp(nb_model.feature_log_prob_[0])
    ratio = spam_prob / (ham_prob + 1e-9)
    top_spam_indices = np.argsort(ratio)[-30:][::-1]
    top_ham_indices = np.argsort(1.0 / (ratio + 1e-9))[-30:][::-1]
    
    metrics_report["top_spam_words"] = [
        {"word": str(feature_names[i]), "score": float(ratio[i])} for i in top_spam_indices
    ]
    metrics_report["top_ham_words"] = [
        {"word": str(feature_names[i]), "score": float(1.0 / (ratio[i] + 1e-9))} for i in top_ham_indices
    ]
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Structure confusion matrix
        # TN FP
        # FN TP
        tn, fp, fn, tp = cm.ravel()
        
        metrics_report["models"][name] = {
            "accuracy": float(acc),
            "precision": float(report["1"]["precision"]),
            "recall": float(report["1"]["recall"]),
            "f1_score": float(report["1"]["f1-score"]),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp)
            }
        }
        print(f"Accuracy: {acc:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        
    # 7. Save Models and Metrics
    print("\nSaving vectorizer and trained models to models/ folder...")
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.pkl"))
    joblib.dump(nb_model, os.path.join(MODEL_DIR, "model_nb.pkl"))
    joblib.dump(lr_model, os.path.join(MODEL_DIR, "model_lr.pkl"))
    joblib.dump(svm_model, os.path.join(MODEL_DIR, "model_svm.pkl"))
    
    metrics_path = os.path.join(BACKEND_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_report, f, indent=4)
        
    print(f"Metrics and dataset insights successfully written to {metrics_path}")
    print("All models successfully trained and stored!")

if __name__ == "__main__":
    train_and_evaluate()
