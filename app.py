import re
import string
import pickle
import streamlit as st
import nltk
import numpy as np
import pandas
# nltk.data.path.append("C:\\Users\\arnab\\AppData\\Roaming\\nltk_data\\tokenizers\\punkt\\PY3")
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords

model = pickle.load(open('model.pkl','rb'))

# def preprocess_text(text):
#     text = text.lower()
#     text = text.translate(str.maketrans('', '', string.punctuation))

#     tokens = word_tokenize(text)

#     stop_words = set(stopwords.words('english'))
#     filtered_tokens = [word for word in tokens if word not in stop_words]

#     porter = PorterStemmer()
#     stemmed_tokens = [porter.stem(word) for word in filtered_tokens]

#     preprocessed_text = ' '.join(stemmed_tokens)

#     return preprocessed_text

def custom_progress_bar(percentage):
    green_width = 70
    red_width = 100 - 70
    
    color1 = "green"  
    color2 = "red"   

    st.markdown(
        f'<div style="width: 1000px; height: 20px; border: 1px solid #ccc; border-radius: 5px; overflow: hidden;">' +
        f'<div style="width: {green_width}%; height: 100%; float: left; background-color: {color1};"></div>' +
        f'<div style="width: {red_width}%; height: 100%; float: left; background-color: {color2};"></div>' +
        '</div>',
        unsafe_allow_html=True
    )
 
def predict_news_authenticity(news_text):
    # preprocessed_news_text = news_text
    # prediction = model.predict([preprocessed_news_text])
    prediction = model.decision_function([news_text])

    # if prediction == 0:
    #     return "Fake"
    # else:
    #     return "Real"
    return prediction
    
st.title("Fake News Detection")

input_news = st.text_area("Enter the News")

# Create two columns for buttons
col1, col2 = st.columns(2)

# Place buttons in the first column
if col1.button('Predict'):
    if not input_news:
        st.header("Please enter a News first!!!")
    else:
        res = predict_news_authenticity(input_news)
        # if result == "Fake":
        #     st.header("The news is Fake")
        # else:
        #     st.header("The news is Real")

        result = 1 / (1 + np.exp(-res))
        st.header("Real"+"-"+str(round(result[0]*100))+"%")
        st.header("Fake"+"-"+str(round(100-result[0]*100))+"%")
        custom_progress_bar(result)

# Place button in the second column
if col2.button('Clear Result'):
    st.header(" ")
    
