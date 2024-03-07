import streamlit as st
import pickle

# Function to load the model
def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Function to save the updated model
def save_model(updated_model, model_path):
    with open(model_path, 'wb') as f:
        pickle.dump(updated_model, f)

# Streamlit app
def main():
    st.title('Fake News Detection System')

    model_path = 'model.pkl'  # Path to the model file
    model = load_model(model_path)  # Load the initial model

    # Input field for news text
    news_text = st.text_input('Enter the news text:')

    # Button to make prediction
    if st.button('Predict'):
        prediction = model.predict([news_text])  # Wrap news_text in a list
        st.write('Prediction:', prediction)

        # Feedback section
        feedback = st.radio('Was the prediction correct?', ('Yes', 'No'))
        if feedback == 'No':
            correct_label = st.selectbox('Select the correct label:', ('Real', 'Fake'))
            # Update training data and retrain model
            # ...
            # Assume model is retrained and saved as updated_model
            save_model(updated_model, model_path)  # Save the updated model

if __name__ == '__main__':
    main()
