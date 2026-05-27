import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download necessary NLTK data dynamically
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Initialize PorterStemmer and download/fetch stopwords
stemmer = PorterStemmer()
try:
    stop_words = set(stopwords.words('english'))
except Exception:
    # Fallback basic stop words if NLTK download fails or is offline
    stop_words = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
        'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
        'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
        'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
        'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
        'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
        'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
        'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
        'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should',
        "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
        'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
        'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
        'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
    }

def clean_text(text: str) -> str:
    """
    Lowercase text and remove punctuation, extra whitespaces, and digit patterns.
    """
    if not text:
        return ""
    # Lowercase
    text = text.lower()
    # Replace line breaks and tabs with space
    text = text.replace('\n', ' ').replace('\t', ' ')
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text: str) -> list:
    """
    Split clean string into word tokens.
    """
    return text.split(' ') if text else []

def remove_stopwords(tokens: list) -> list:
    """
    Filter out common English stop words.
    """
    return [token for token in tokens if token and token not in stop_words]

def stem_tokens(tokens: list) -> list:
    """
    Apply Porter Stemming algorithm to a list of tokens.
    """
    return [stemmer.stem(token) for token in tokens if token]

def preprocess_text(text: str) -> str:
    """
    Standard NLP preprocessing pipeline that transforms raw text into a cleaned, 
    stemmed, single string suitable for TF-IDF vectorization.
    """
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    filtered = remove_stopwords(tokens)
    stemmed = stem_tokens(filtered)
    return " ".join(stemmed)

def preprocess_step_by_step(text: str) -> dict:
    """
    Executes the NLP pipeline step-by-step and returns all intermediate states
    for high-fidelity frontend visualizations.
    """
    cleaned = clean_text(text)
    
    # Simple tokenizer
    tokens = tokenize(cleaned)
    # Filter empty tokens
    tokens = [t for t in tokens if t]
    
    filtered = remove_stopwords(tokens)
    stemmed = stem_tokens(filtered)
    final_str = " ".join(stemmed)
    
    return {
        "original": text,
        "cleaned": cleaned,
        "tokens": tokens,
        "no_stopwords": filtered,
        "stemmed": stemmed,
        "final": final_str
    }

if __name__ == "__main__":
    test_msg = "WINNER! As a valued network customer you have been selected to receive a £900 prize reward! Click here: http://claim.com now."
    print("Test message step-by-step:")
    steps = preprocess_step_by_step(test_msg)
    for step, val in steps.items():
        print(f"\n[{step.upper()}]:\n{val}")
