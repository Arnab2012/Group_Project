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

# Function to simulate retraining the model with updated data
def retrain_model(model, updated_data, correct_label):
    # Assume some process to retrain the model with updated data and correct labels
    # updated_data could be labeled examples collected from user feedback
    # correct_label indicates whether the news is 'Real' or 'Fake'
    updated_model = model.fit([updated_data], [correct_label])  # Placeholder for actual retrained model
    return updated_model

# Streamlit app
def main():
    st.title('Fake News Detection System')

    model_path = 'model.pkl'  # Path to the model file
    model = load_model(model_path)  # Load the initial model

    # Input field for news text
    news_text = st.text_area('Enter the news text:')

    # Button to make prediction
    if st.button('Predict'):
        if not news_text:
            st.warning('Please enter some news to predict.')
        else:
            prediction = model.predict([news_text])
            st.write('Prediction:', prediction)

            # Feedback section
            feedback = st.radio('Was the prediction correct?', ('Yes', 'No'))
            if feedback == 'No':
                correct_label = st.selectbox('Select the correct label:', ('Real', 'Fake'))
                # Update training data and retrain model
                # For now, we will assume model is retrained with updated data
                updated_model = retrain_model(model, updated_data=news_text, correct_label=correct_label)
                save_model(updated_model, model_path)  # Save the updated model

if __name__ == '__main__':
    main()
