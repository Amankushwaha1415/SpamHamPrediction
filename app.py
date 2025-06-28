import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

stopwords.words('english')
ps=PorterStemmer()


def text_preprocessing(text):
    text=text.lower()  #lower the text
    text=re.sub(r'\d+', '', text) # remove digits
    text=re.sub(r'[^\w\s]', '', text) # remove punctuation
    text=re.sub(r'\s+', ' ', text) # remove extra spaces
    text=text.strip() # remove leading and trailing spaces
    text=word_tokenize(text) #tokenize the word
    text=[ps.stem(word) for word in text if word not in stopwords.words('english')]  # remove stop words and stem the words
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