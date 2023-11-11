from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import cross_val_score
import pandas as pd

# Download NLTK data, including stopwords
nltk.download('stopwords')
nltk.download('punkt')

class SpamClassifier:
    def __init__(self):
        # Initialize the model pipeline with CountVectorizer and MultinomialNB
        self.model = make_pipeline(CountVectorizer(), MultinomialNB())

    def preprocess_text(self, text):
        # Check if the text is not NaN
        if pd.notna(text):
            # Tokenization and removal of stop words
            stop_words = set(stopwords.words('english'))
            words = nltk.word_tokenize(text)
            words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
            return ' '.join(words)
        else:
            # Handle missing values (you can modify this based on your needs)
            return ''

    def train(self, X_train, y_train):
        # Combine X_train and y_train into a single DataFrame
        df_train = pd.DataFrame({'Category': y_train, 'Message': X_train})

        # Drop rows with NaN values
        df_train.dropna(inplace=True)

        # Preprocess and fit the CountVectorizer on the training data
        X_train_preprocessed = [self.preprocess_text(text) for text in df_train["Message"]]
        self.model.named_steps['countvectorizer'].fit(X_train_preprocessed)

        # Train the model
        self.model.fit(X_train_preprocessed, df_train["Category"])

    def evaluate(self, X_test, y_test):
        # Preprocess the test data
        X_test_preprocessed = [self.preprocess_text(text) for text in X_test]

        # Evaluate the model using cross-validation
        accuracy = cross_val_score(self.model, X_test_preprocessed, y_test, cv=5, scoring='accuracy')
        return accuracy.mean()

    def predict(self, messages, threshold=0.3):
        # Preprocess new data
        messages_preprocessed = [self.preprocess_text(text) for text in messages]

        # Transform using the fitted CountVectorizer vocabulary
        messages_transformed = self.model.named_steps['countvectorizer'].transform(messages_preprocessed)

        # Get the predicted probabilities for each class
        probabilities = self.model.named_steps['multinomialnb'].predict_proba(messages_transformed)

        # Assuming 'spam' is the positive class
        # Extract the probability of the 'spam' class
        spam_probabilities = probabilities[:, 1]

        # Determine if the message is spam based on the threshold
        return [
            spam_probabilities > threshold,spam_probabilities
        ]
        

