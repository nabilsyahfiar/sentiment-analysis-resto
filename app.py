from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
# from sklearn.externals import joblib


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
  df = pd.read_csv("review.csv", encoding="latin-1")
  df.drop(columns = ['restaurant_name', 'name'], inplace = True)
  df.columns = ['rating', 'review']
  
  review_remove_translated = []
  reviews_dict = df.to_dict('list')

  for review in reviews_dict['review']:
    review_sep = review.split("(Translated by Google) ")
    
    # Jika terdapat (Translated by Google)
    if review_sep[0] == "":
      review_sep = ("".join(review_sep)).split("(Original)")
      review_sep = review_sep[0]
      review = "".join(review_sep)
      
    review_remove_translated.append(review)

  reviews_dict['review'] = review_remove_translated
  df = pd.DataFrame(reviews_dict)

  import string
  import re
  def clean_text(text):
      return re.sub('[^a-zA-Z]', ' ', text).lower()
  df['cleaned_text'] = df['review'].apply(lambda x: clean_text(x))
  df['label'] = df['rating'].map({5.0:0, 4.0:0, 3.0:0, 2.0:1, 1.0:1})

  def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")), 3)*100
  df['text_len'] = df['review'].apply(lambda x: len(x) - x.count(" "))
  df['punct'] = df['review'].apply(lambda x: count_punct(x))

  def tokenize_text(text):
    tokenized_text = text.split()
    return tokenized_text
  df['tokens'] = df['cleaned_text'].apply(lambda x: tokenize_text(x))

  import nltk
  from nltk.corpus import stopwords
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')

  def lemmatize_text(token_list):
    return " ".join([lemmatizer.lemmatize(token) for token in token_list if not token in set(all_stopwords)])

  lemmatizer = nltk.stem.WordNetLemmatizer()
  df['lemmatized_text'] = df['tokens'].apply(lambda x: lemmatize_text(x))

  X = df[['lemmatized_text', 'text_len', 'punct']]
  y = df['label']

  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
  

  from sklearn.feature_extraction.text import TfidfVectorizer
  tfidf = TfidfVectorizer(max_df = 0.5, min_df = 2) # ignore terms that occur in more than 50% documents and the ones that occur in less than 2
  tfidf_train = tfidf.fit_transform(X_train['lemmatized_text'])
  tfidf_test = tfidf.transform(X_test['lemmatized_text'])

  from sklearn.svm import SVC
  classifier = SVC(kernel = 'linear', random_state = 10)
  classifier.fit(tfidf_train, y_train)
  classifier.score(tfidf_test, y_test)

  if request.method == 'POST':
    message = request.form['message']
    data = [message]
    vect = tfidf.transform(data).toarray()
    my_prediction = classifier.predict(vect)
  return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=True)