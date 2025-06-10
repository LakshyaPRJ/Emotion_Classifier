🎭 Emotion Classifier Web App:
This is a web-based emotion classification application built using Streamlit and Scikit-learn. It analyzes input text and predicts the underlying emotion among six categories: sadness, joy, love, anger, fear, and surprise.

📂 Dataset Used
  Source: [Kaggle - Emotions Dataset](https://www.kaggle.com/datasets/bhavikjikadara/emotions-dataset/data)
  Format: CSV

Columns:
  text: Raw input sentence.
  label: Corresponding emotion label.
Contains pre-labeled data suitable for supervised learning in emotion detection.

⚙️ Project Approach
1. Text Preprocessing
  Lowercasing, punctuation & number removal
  Stopword removal using NLTK
  Lemmatization with WordNetLemmatizer

2. Label Encoding
  Emotion labels encoded into integers using LabelEncoder.

3. Vectorization
  TF-IDF vectorization with unigrams and bigrams.
  Max features: 10,000.

4. Feature Selection
  SelectKBest using Chi-squared test to select top 5,000 features.

5. Model
  Logistic Regression with class balancing (class_weight='balanced') and regularization (C=0.9).
  Trained using an 80-20 train-test split.

6. Evaluation
  Accuracy score and confusion matrix visualization.

7. Web Interface
  Developed using Streamlit.
  Allows live input for emotion prediction with confidence scores and visualization of confusion matrix.

🧩 Dependencies
* Make sure the following libraries are installed:
  pip install pandas scikit-learn streamlit nltk matplotlib seaborn
* Also, download NLTK data (only once):
  import nltk
  nltk.download('stopwords')
  nltk.download('wordnet')
  
🚀 Run Locally
Clone this repository:
  git clone https://github.com/LakshyaPRJ/Emotion_Classifier.git
  cd Emotion_Classifier
  Place the emotions.csv file from the Kaggle dataset in the same directory.

Run the app:
  streamlit run app/main.py

💡 Tip: For better prediction accuracy, try inputting longer, meaningful sentences instead of single words or very short texts. This helps the model better understand the context and emotion.  
  
🔗 Links
📘 Dataset on Kaggle(https://www.kaggle.com/datasets/bhavikjikadara/emotions-dataset/data)
👤 LinkedIn - Lakshya Prajapati(www.linkedin.com/in/lakshyaprajapati)
