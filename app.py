import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import re
import statistics
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.corpus import wordnet
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from string import punctuation


app = Flask(__name__)
model_nb_tv = pickle.load(open('model_nb_tv.pkl', 'rb'))
model_nb_cvr = pickle.load(open('model_nb_cvr.pkl', 'rb'))
model_lr_tv = pickle.load(open('model_lr_tv.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
      
    review = [request.form.get("review")]
    
    data = {
        'review':review
        
        }
    df = pd.DataFrame(data)
    df['review']= df['review'].astype(str)
    
    def get_simple_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return None
    
    def lower_case(text):
        '''
        converting to lower case
        '''
        return text.lower()
    
    def strip_non_ascii(data_str):
        '''
        Returns the string without non ASCII characters
        '''
        stripped = (c for c in data_str if 0 < ord(c) < 127)
        return ''.join(stripped) 
    def fix_abbreviation(data_str):
        '''
        Fix Abbreviations
        '''
        data_str = data_str.lower()
        data_str = re.sub(r'\bthats\b', 'that is', data_str)
        data_str = re.sub(r'\bive\b', 'i have', data_str)
        data_str = re.sub(r'\bim\b', 'i am', data_str)
        data_str = re.sub(r'\bya\b', 'yeah', data_str)
        data_str = re.sub(r'\bcant\b', 'can not', data_str)
        data_str = re.sub(r'\bdont\b', 'do not', data_str)
        data_str = re.sub(r'\bwont\b', 'will not', data_str)
        data_str = re.sub(r'wtf', 'what the fuck', data_str)
        data_str = re.sub(r'\bwth\b', 'what the hell', data_str)
        data_str = re.sub(r'\br\b', 'are', data_str)
        data_str = re.sub(r'\bu\b', 'you', data_str)
        data_str = re.sub(r'\bk\b', 'OK', data_str)
        data_str = re.sub(r'\bsux\b', 'sucks', data_str)
        data_str = data_str.strip()
        return data_str
    def remove_features(data_str):
        '''
        Remove punctuations mentions and alphanumeric characters
        '''
    # compile regex
        url_re = re.compile('https?://(www.)?\w+\.\w+(/\w+)*/?')
        num_re = re.compile('(\\d+)')
        mention_re = re.compile('@(\w+)')
        alpha_num_re = re.compile("^[a-z0-9_.]+$")
    # convert to lowercase
        data_str = data_str.lower()
    # remove hyperlinks
        data_str = url_re.sub(' ', data_str)
    # remove @mentions
        data_str = mention_re.sub(' ', data_str)
    # remove numeric 'words'
        data_str = num_re.sub(' ', data_str)
    # remove non a-z 0-9 characters and words shorter than 1 characters
        list_pos = 0
        cleaned_str = ''
        for word in data_str.split():
            if list_pos == 0:
                if alpha_num_re.match(word):
                    cleaned_str = word
                else:
                    cleaned_str = ' '
            else:
                if alpha_num_re.match(word):
                    cleaned_str = cleaned_str + ' ' + word
                else:
                    cleaned_str += ' '
            list_pos += 1
    # remove unwanted space, *.split() will automatically split on
    # whitespace and discard duplicates, the " ".join() joins the
    # resulting list into one string.
        return " ".join(cleaned_str.split())
    def lemmatize(data_str):
        '''
        Lamatizing the text and removing stopwords and punctuation
        '''
        list_pos = 0
        cleaned_str = ''
        final_text=[]
        lmtzr = WordNetLemmatizer()
        stop = set(stopwords.words('english'))
        punc = list(punctuation)
        stop.update(punc)
        for i in data_str.split():
            if i.strip().lower() not in stop:
                pos = pos_tag([i.strip()])
                if get_simple_pos(pos[0][1]) is None:
                    pass
                else:
                    word = lmtzr.lemmatize(i.strip(),get_simple_pos(pos[0][1]))
                    final_text.append(word.lower())
        return " ".join(final_text)
    df.review = df.review.apply(lower_case)
    df.review = df.review.apply(strip_non_ascii) 
    df.review = df.review.apply(fix_abbreviation)
    df.review = df.review.apply(remove_features)
    df.review = df.review.apply(lemmatize)
    
    
    prediction_nb_tv = model_nb_tv.predict(df)
    prediction_nb_cvr = model_nb_cvr.predict(df)
    prediction_lr_tv = model_lr_tv.predict(df)
    final_pred = np.array([])
    for i in range(0,len(df)):
        final_pred = np.append(final_pred, statistics.mode([prediction_nb_tv[i], prediction_nb_cvr[i],prediction_lr_tv[i]]))
    
    output = final_pred

    return render_template('index.html', prediction_text='Sentiment of the user is: {}'.format(output))



if __name__ == "__main__":
    app.run(debug=True)