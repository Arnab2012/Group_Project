import streamlit as st
import pickle


def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def save_model(updated_model, model_path):
    with open(model_path, 'wb') as f:
        pickle.dump(updated_model, f)

# Function to simulate retraining the model with updated data
def retrain_model(model, updated_data, correct_label):
    updated_model = model.fit([updated_data], [correct_label])  # Placeholder for actual retrained model
    return updated_model


updated_model_path = 'model.pkl'  # Path to the initial model
def main():
    st.title('Fake News Detection System')

    model = load_model(updated_model_path)  # Load the initial model

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
            feedback = st.radio('Was the prediction correct?', ('Yes', 'No'), index=None)
            if feedback == 'No':
                st.header("Inside if")
                correct_label = st.selectbox('Select the correct label:', ('Real', 'Fake'))
                if st.button('Submit Feedback'):
                    # Update training data and retrain model
                    updated_model = retrain_model(model, updated_data=news_text, correct_label=correct_label)
                    save_model(updated_model, updated_model_path)  # Save the updated model
                    model = load_model(updated_model_path)  # Load the updated model for subsequent predictions
            elif feedback == 'Yes':
                st.write("Feedback received: Yes")
                # Continue with prediction without retraining

if __name__ == '__main__':
    main()
