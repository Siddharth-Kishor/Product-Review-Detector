from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
#from nltk.corpus import wordnet
#from nltk.stem import WordNetLemmatizer
import string
#from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('stopwords')
nltk.download('punkt')
#nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

df = pd.read_csv(r"./Labelled Yelp Dataset.csv")

print("-"*20)

#def remove_html(text):
#    return BeautifulSoup(text, "lxml").text.032

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    cleaned_text = " ".join(filtered_words)
    return cleaned_text

def stem_text(text):
    stemmer = PorterStemmer()
    words = word_tokenize(text)
    stemmed_words = [stemmer.stem(word) for word in words]
    stemmed_text = " ".join(stemmed_words)
    return stemmed_text

#df["Review"] = df["Review"].apply(remove_html)
print("-"*20)
df["Review"] = df["Review"].apply(remove_urls)
print("-"*20)
df["Review"] = df["Review"].str.lower()
print("-"*20)
df["Review"] = df["Review"].apply(lambda text: re.sub(r"[^\w\s]", "", text))
print("-"*20)
df["Review"] = df["Review"].apply(remove_stopwords)
print("-"*20)
df["Review"] = df["Review"].apply(stem_text)
print("-"*20)
print(df)
print("-"*20)
import pickle
with open(r"./CVectorizer.pkl", "rb") as input_file:
  vectorizer = pickle.load(input_file)
with open(r"./TfidfTransformer (1).pkl", "rb") as input_file:
  transformer = pickle.load(input_file)
print("-"*20)
X_cv = vectorizer.transform(df["Review"])
X_tf = transformer.transform(X_cv)
X_tf
print("-"*20)
with open(r"./FakeReviewDetectionModel.pkl", "rb") as input_file:
  scvmodel = pickle.load(input_file)
print("-"*20)
X = scvmodel.predict(X_tf)
print("-"*20)
def label(text):
  if text == -1:
    return 0
  elif text == 1:
    return 1
  print("-"*20)
Y = df['Label'].apply(label)
print(X)
print(Y)
print("-"*20)
acc = accuracy_score(X, Y)
f1 = f1_score(X, Y)
prec = precision_score(X, Y)
rec = recall_score(X, Y)
print("-"*20)
results = pd.DataFrame([['SVC', acc, f1, prec, rec]],
                        columns = ['Model', 'Accuracy', 'F1','Precision', 'Recall'])
print(results)