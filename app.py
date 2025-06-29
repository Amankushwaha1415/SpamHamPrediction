import streamlit as st
import pickle
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


ps=PorterStemmer()

stopword=['a',
 'about',
 'above',
 'after',
 'again',
 'against',
 'ain',
 'all',
 'am',
 'an',
 'and',
 'any',
 'are',
 'aren',
 "aren't",
 'as',
 'at',
 'be',
 'because',
 'been',
 'before',
 'being',
 'below',
 'between',
 'both',
 'but',
 'by',
 'can',
 'couldn',
 "couldn't",
 'd',
 'did',
 'didn',
 "didn't",
 'do',
 'does',
 'doesn',
 "doesn't",
 'doing',
 'don',
 "don't",
 'down',
 'during',
 'each',
 'few',
 'for',
 'from',
 'further',
 'had',
 'hadn',
 "hadn't",
 'has',
 'hasn',
 "hasn't",
 'have',
 'haven',
 "haven't",
 'having',
 'he',
 "he'd",
 "he'll",
 'her',
 'here',
 'hers',
 'herself',
 "he's",
 'him',
 'himself',
 'his',
 'how',
 'i',
 "i'd",
 'if',
 "i'll",
 "i'm",
 'in',
 'into',
 'is',
 'isn',
 "isn't",
 'it',
 "it'd",
 "it'll",
 "it's",
 'its',
 'itself',
 "i've",
 'just',
 'll',
 'm',
 'ma',
 'me',
 'mightn',
 "mightn't",
 'more',
 'most',
 'mustn',
 "mustn't",
 'my',
 'myself',
 'needn',
 "needn't",
 'no',
 'nor',
 'not',
 'now',
 'o',
 'of',
 'off',
 'on',
 'once',
 'only',
 'or',
 'other',
 'our',
 'ours',
 'ourselves',
 'out',
 'over',
 'own',
 're',
 's',
 'same',
 'shan',
 "shan't",
 'she',
 "she'd",
 "she'll",
 "she's",
 'should',
 'shouldn',
 "shouldn't",
 "should've",
 'so',
 'some',
 'such',
 't',
 'than',
 'that',
 "that'll",
 'the',
 'their',
 'theirs',
 'them',
 'themselves',
 'then',
 'there',
 'these',
 'they',
 "they'd",
 "they'll",
 "they're",
 "they've",
 'this',
 'those',
 'through',
 'to',
 'too',
 'under',
 'until',
 'up',
 've',
 'very',
 'was',
 'wasn',
 "wasn't",
 'we',
 "we'd",
 "we'll",
 "we're",
 'were',
 'weren',
 "weren't",
 "we've",
 'what',
 'when',
 'where',
 'which',
 'while',
 'who',
 'whom',
 'why',
 'will',
 'with',
 'won',
 "won't",
 'wouldn',
 "wouldn't",
 'y',
 'you',
 "you'd",
 "you'll",
 'your',
 "you're",
 'yours',
 'yourself',
 'yourselves',
 "you've"]

def text_preprocessing(text):
    text=text.lower()  #lower the text
    text=re.sub(r'\d+', '', text) # remove digits
    text=re.sub(r'[^\w\s]', '', text) # remove punctuation
    text=re.sub(r'\s+', ' ', text) # remove extra spaces
    text=text.strip() # remove leading and trailing spaces
    text=word_tokenize(text) #tokenize the word
    text=[ps.stem(word) for word in text if word not in stopword]  # remove stop words and stem the words
    # print (text) # print the preprocessed text
    text=' '.join(text)
    return text  # join the words back to a string


# Load the model
model = pickle.load(open('spamHamPrediction.pkl','rb'))

# load the vectorizer
vectorizer = pickle.load(open('tfidf_vectorizer.pkl','rb'))

st.title('Spam or Ham Email Classifier')
st.write("")
st.write("")


email_text = st.text_area('Enter the email text here',height=200)  
if st.button('Predict'):
    if email_text:
        preprocessed_text = text_preprocessing(email_text)
        vectorized_text = vectorizer.transform([preprocessed_text])
        prediction = model.predict(vectorized_text)
        
        if prediction[0] == 1:
            st.error('This email is classified as Spam.')
        
        else:
            st.success('This email is classified as Ham.')
    else:
        st.error('Please enter some text to classify.')