# import flask libraries
from flask import Flask, render_template, url_for, request
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# load the model from disk
# model = pickle.load(open('model.pkl', 'rb'))
# Load count vector from disk
cv = pickle.load(open('transformer.pkl', 'rb'))
# Load the vocabulary
words = pickle.load(open('vocabulary.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('test.html')


@app.route('/predict', methods=['POST'])
def predict():

    if request.method == 'POST':
        sentence = request.form['message']

        sentence = [sentence]

        vec = CountVectorizer().fit(sentence)
        bag_of_words = vec.transform(sentence)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx])
                      for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        top_words = words_freq[:20]

        # Convert most freq words to dataframe for plotting bar plot
        top_df = pd.DataFrame(top_words)
        top_df.columns = ["Word", "Freq"]

        return render_template('result.html', tables=[top_df.to_html(classes='data')], titles=top_df.columns.values)


if __name__ == '__main__':
    app.run(debug=True)
