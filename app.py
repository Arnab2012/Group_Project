import re
import string
import pickle
import streamlit as st
import nltk
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


def predict_news_authenticity(news_text):
    preprocessed_news_text = news_text
    # prediction = model.predict([preprocessed_news_text])
    prediction = model.decision_function([preprocessed_news_text])

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
        st.header(result)

# Place button in the second column
if col2.button('Clear Result'):
    st.header(" ")
