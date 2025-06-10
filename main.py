# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import re  # For regular expressions
import streamlit as st  # For creating web app interface
from sklearn.feature_extraction.text import TfidfVectorizer  # For text vectorization
from sklearn.linear_model import LogisticRegression  # For classification model
from sklearn.preprocessing import LabelEncoder  # For encoding labels
from sklearn.pipeline import make_pipeline  # For creating ML pipeline
from sklearn.feature_selection import SelectKBest, chi2  # For feature selection
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.metrics import confusion_matrix, accuracy_score  # For evaluation metrics
import nltk  # Natural Language Toolkit
from nltk.corpus import stopwords  # For stopwords removal
from nltk.stem import WordNetLemmatizer  # For word lemmatization
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For enhanced visualizations

# Download NLTK resources (run once)
nltk.download('stopwords')  # Download stopwords dataset
nltk.download('wordnet')  # Download WordNet for lemmatization

# Emotion label mapping - converts numeric predictions to human-readable emotions
emotion_mapping = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

# --- Advanced Text Preprocessing ---
lemmatizer = WordNetLemmatizer()  # Initialize lemmatizer
stop_words = set(stopwords.words('english'))  # Get English stopwords

def clean_text(text):
    """Preprocess text by:
    1. Converting to lowercase
    2. Removing punctuation
    3. Removing numbers
    4. Lemmatizing words
    5. Removing stopwords
    """
    if isinstance(text, str):
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\d+', '', text)  # Remove numbers
        # Lemmatize and remove stopwords
        text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
        return text
    return ""  # Return empty string if input is not string

# --- Load Data ---
@st.cache_data  # Streamlit decorator to cache data and avoid reloading
def load_data():
    """Load and preprocess the dataset"""
    df = pd.read_csv('emotions.csv')  # Read CSV file
    df['text'] = df['text'].apply(clean_text)  # Apply text cleaning
    return df

# --- Train Model with Feature Selection ---
@st.cache_resource  # Streamlit decorator to cache the trained model
def train_model(df):
    """Train the emotion classification model with:
    1. Label encoding
    2. Train-test split
    3. TF-IDF vectorization
    4. Feature selection
    5. Logistic regression
    """
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['label'])  # Encode emotion labels

    # Split data into train and test sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], y, test_size=0.2, random_state=42, stratify=y
    )

    # Create ML pipeline with:
    # 1. TF-IDF vectorizer (with bigrams)
    # 2. Feature selection (chi-squared test)
    # 3. Logistic regression classifier
    pipeline = make_pipeline(
        TfidfVectorizer(max_features=10000, ngram_range=(1, 2)),  # Bigrams
        SelectKBest(chi2, k=5000),  # Select top features
        LogisticRegression(max_iter=1000, class_weight='balanced', C=0.9)
    )

    pipeline.fit(X_train, y_train)  # Train the model

    # Evaluate model performance
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
    cm = confusion_matrix(y_test, y_pred)  # Generate confusion matrix

    return pipeline, label_encoder, accuracy, cm, X_test, y_test

# --- Streamlit App ---
def main():
    """Main function to run the Streamlit web application"""
    st.title("Emotion Classifier 🎭")
    st.write("Enter text to predict its emotion (sadness, joy, love, anger, fear, surprise):")

    # Load and preprocess data
    df = load_data()
    # Train model and get components
    model, label_encoder, accuracy, cm, X_test, y_test = train_model(df)

    # Display model evaluation metrics
    st.subheader("Model Evaluation")
    st.write(f"**Accuracy:** {accuracy:.2%}")  # Show accuracy percentage

    # Prediction section
    st.subheader("Try It Out")
    # Text area for user input with default example
    user_input = st.text_area("Input Text:", "I'm feeling excited about the trip!")

    if st.button("Predict Emotion"):
        cleaned_input = clean_text(user_input)  # Clean user input
        prediction = model.predict([cleaned_input])[0]  # Make prediction
        emotion = emotion_mapping[prediction]  # Get emotion label

        # Get probabilities for all classes
        probas = model.predict_proba([cleaned_input])[0]
        # Get indices of top 3 predicted emotions
        top3_indices = probas.argsort()[-3:][::-1]

        # Display prediction result
        st.success(f"**Predicted Emotion:** {emotion.upper()}")

        # Display confidence scores for top 3 emotions
        st.write("**Confidence Scores:**")
        for idx in top3_indices:
            st.write(f"- {emotion_mapping[idx]}: {probas[idx]:.1%}")

    # Display confusion matrix visualization
    st.write("**Confusion Matrix:**")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=emotion_mapping.values(),
                yticklabels=emotion_mapping.values(),
                ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)  # Show plot in Streamlit

    # Add GitHub and LinkedIn links in footer
    st.markdown("""
    ---
    🔗 [View on GitHub](https://github.com/LakshyaPRJ/Emotion_Classifier)  
    🔗 [Connect on LinkedIn](https://www.linkedin.com/in/lakshyaprajapati/)
    """)


if __name__ == "__main__":
    main()  # Run the app when script is executed