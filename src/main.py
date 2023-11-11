import time
import firebase_admin
from firebase_admin import credentials
from db.firestore import DB
from model.spam_classifier import SpamClassifier
import pandas as pd


# Load your CSV dataset with a different encoding
dataset_path = "C:/Users/bluev/Documents/itbsa-project/ai-spam-detection/data/main_data.csv"
df = pd.read_csv(dataset_path, encoding='latin1')

# Initialize Firebase
cred = credentials.Certificate("C:/Users/bluev/Documents/itbsa-project/firebase.json")
firebase_admin.initialize_app(cred)

def main():
    # Initialize the database
    database = DB()
    feed_collection = database.get_collection("feed")

    # Initialize the SpamClassifier
    spam_classifier = SpamClassifier()

    # Train the model with your loaded dataset
    spam_classifier.train(df["Message"], df["Category"])

    # Listen for added documents in the "feed" collection
    feed_collection.on_snapshot(lambda col_snapshot, changes, read_time: on_snapshot_callback(spam_classifier, changes))

    retrain_interval = 3600  # 1 hour (adjust as needed)

    while True:
        # Retrain the model periodically
        if time.time() % retrain_interval == 0:
            print("Retraining the model...")
            spam_classifier.train(df["Message"], df["Category"])

        # Sleep for 60 seconds before checking again
        time.sleep(60)

def on_snapshot_callback(spam_classifier, changes):
    for change in changes:
        if change.type.name == 'ADDED':
            process_added_document(spam_classifier, change.document)

def process_added_document(spam_classifier, document):
    message = document.to_dict().get("message", "")
    print(f"Original Message: {message}")

    # Make predictions using the SpamClassifier
    predictions = spam_classifier.predict([message])

    print(predictions)

    if predictions[0]:
        print(f'Predicted as spam, marking as spam...,{document.id}')
        DB().get_document("feed", document.id).update({"spam": True})

if __name__ == "__main__":
    main()
