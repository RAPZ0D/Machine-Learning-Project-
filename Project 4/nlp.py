import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Text Tokenization 
nltk.download('punkt')

# Read the TSV file
file_path = 'Restaurant_Reviews.tsv'
data = pd.read_csv(file_path, delimiter='\t', quoting=3)

# Tokenize a single review into words
first_review = data['Review'][0]
tokens_words = word_tokenize(first_review.lower())  # Tokenization and convert to lowercase
tokens_sentences = sent_tokenize(first_review)

print("Tokenized Words:")
print(tokens_words)
print("\nTokenized Sentences:")
print(tokens_sentences)

# Stopword Removal 
nltk.download('stopwords')

# Stopwords removal
stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in tokens_words if token not in stop_words]

print("\nText after Stopwords Removal:")
print(filtered_tokens)

# Stemming 
# Stemming
ps = PorterStemmer()
stemmed_words = [ps.stem(token) for token in filtered_tokens]

print("\nStemmed Words:")
print(stemmed_words)

# Lemmitization 
nltk.download('wordnet')

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(token) for token in filtered_tokens]

print("\nLemmatized Words:")
print(lemmatized_words)

# Sentiment Analysis 
nltk.download('vader_lexicon')

# Sentiment Analysis
sia = SentimentIntensityAnalyzer()
sentiment_scores = sia.polarity_scores(first_review)

print("\nSentiment Analysis:")
print(sentiment_scores)


# Combined Code 
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# Read the TSV file
file_path = 'Restaurant_Reviews.tsv'
data = pd.read_csv(file_path, delimiter='\t', quoting=3)

# Tokenization
first_review = data['Review'][0]
tokens_words = word_tokenize(first_review.lower())
tokens_sentences = sent_tokenize(first_review)

# Stopwords removal
stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in tokens_words if token not in stop_words]

# Stemming
ps = PorterStemmer()
stemmed_words = [ps.stem(token) for token in filtered_tokens]

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(token) for token in filtered_tokens]

# Sentiment Analysis
sia = SentimentIntensityAnalyzer()
sentiment_scores = sia.polarity_scores(first_review)

print("Tokenized Words:")
print(tokens_words)
print("\nTokenized Sentences:")
print(tokens_sentences)
print("\nText after Stopwords Removal:")
print(filtered_tokens)
print("\nStemmed Words:")
print(stemmed_words)
print("\nLemmatized Words:")
print(lemmatized_words)
print("\nSentiment Analysis:")
print(sentiment_scores)
