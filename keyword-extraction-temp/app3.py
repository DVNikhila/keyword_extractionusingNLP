# import libraries
import pandas as pd
import json
import plotly
import plotly.express as px
import re
import base64
import nltk
from io import BytesIO
from rake_nltk import Rake
from flask import Flask, render_template, url_for, request
# from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

app = Flask(__name__)


@app.route('/')
def home():
    
    return render_template('test.html')
   

@app.route('/predict', methods=['POST'])
def predict():

    if request.method == 'POST':
        sentence = request.form['message']
        

    # keywords extraction
    r = Rake()

    # Extraction given the list of strings where each string is a sentence.
    r.extract_keywords_from_text(sentence)

    # To get keyword phrases ranked highest to lowest with scores.
    out = r.get_ranked_phrases_with_scores()

    freq = []
    word = []
    for i in range(0, len(out)):
        freq.append(out[i][0])
        word.append(out[i][1])

    df = pd.DataFrame(list(zip(freq, word)), columns=['FREQ', 'WORD'])

    # Barplot of most freq Bi-grams
    fig = px.bar(df, x="WORD", y="FREQ")

    # Create graphJSON
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('result.html', graphJSON=graphJSON, tables=[df.to_html(classes='data')], titles=df.columns.values)


if __name__ == '__main__':
    app.run(debug=True)
