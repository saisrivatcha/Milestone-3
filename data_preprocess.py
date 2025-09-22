# Importing libraries
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Downloading data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

# Reading the dataset
df = pd.read_csv("fake_job_postings.csv")
print(df.head())

print("\nFraudulent Value Counts:\n", df['fraudulent'].value_counts())

# Balancing the dataset
df_0 = df[df['fraudulent'] == 0].sample(n=5000, random_state=42, replace=True)
df_1 = df[df['fraudulent'] == 1].sample(n=5000, random_state=42, replace=True)

df_balanced = pd.concat([df_0, df_1]).reset_index(drop=True)
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Prepare stopwords, lemmatizer, and stemmer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Clean missing and duplicate job descriptions
df_balanced.dropna(subset=['description'], inplace=True)
df_balanced.drop_duplicates(subset=['description'], inplace=True)

# Apply full preprocessing to the dataset
def clean_text(text):
    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove digits
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenization
    tokens = nltk.word_tokenize(text)

    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Optional: Stemming (commented out)
    # tokens = [stemmer.stem(word) for word in tokens]

    return ' '.join(tokens)

# Apply cleaning
df_balanced['cleaned_job_desc'] = df_balanced['description'].apply(clean_text)

# Final dataset
df = df_balanced[['cleaned_job_desc', 'fraudulent']]
print("\nSample cleaned data:\n", df.head())

# Train-test split
X = df["cleaned_job_desc"]
y = df["fraudulent"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining Size: {len(X_train)} | Testing Size: {len(X_test)}")
print("Training set:", X_train.shape, y_train.shape)
print("Testing set:", X_test.shape, y_test.shape)

# -------------------------
# TF-IDF Vectorization
# -------------------------
tfidf = TfidfVectorizer(
    max_features=5000,     # limit vocabulary size
    ngram_range=(1,2),     # unigrams + bigrams
    stop_words='english'   # remove stopwords (extra safety)
)

# Fit on training data and transform
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("\nTF-IDF Train shape:", X_train_tfidf.shape)
print("TF-IDF Test shape:", X_test_tfidf.shape)