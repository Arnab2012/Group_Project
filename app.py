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
    green_width = percentage
    red_width = 100 - percentage

    st.markdown(f'
    <div>Hello World</div>
    <div style="width: 100%; height: 30px; border: 1px solid #ccc; border-radius: 5px; overflow: hidden;">
        <div style="width: {green_width}%; height: 100%; float: left; background-color: #00FF00;"></div>
        <div style="width: {red_width}%; height: 100%; float: left; background-color: #FF0000;"></div>
    </div>'
        unsafe_allow_html=True )
    # html_code = f"""
    # <div>Hello World</div>
    # <div style="width: 100%; height: 30px; border: 1px solid #ccc; border-radius: 5px; overflow: hidden;">
    #     <div style="width: {green_width}%; height: 100%; float: left; background-color: #00FF00;"></div>
    #     <div style="width: {red_width}%; height: 100%; float: left; background-color: #FF0000;"></div>
    # </div>
    # """
    # st.write(html_code, unsafe_allow_html=True)

color = "blue"  # choose your color

# Create a small rectangular bar with color
st.markdown(
    f'<div style="background-color: {color}; width: 100px; height: 20px;"></div>',
    unsafe_allow_html=True
)
# x = 70  # percentage for the first color
# color1 = "blue"  # color for the first percentage
# color2 = "red"   # color for the remaining percentage

# # Calculate the width of each section based on the percentage
# width1 = x
# width2 = 100 - x

# # Create the bar with two sections of different colors
# st.markdown(
#     f'<div style="background: linear-gradient(to right, {color1} {width1}%, {color2} {width1}%); width: 100px; height: 20px;"></div>',
#     unsafe_allow_html=True
# )
    
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
    
