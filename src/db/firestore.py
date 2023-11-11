from firebase_admin import firestore

class DB:
    db = None
    def __init__(self):
        self.db = firestore.client()
    
    def get_collection(self, collection):
        return self.db.collection(collection)
    
    def get_document(self, collection, document):
        return self.db.collection(collection).document(document)
    
    def listen(self, collection, callback):
        return self.db.collection(collection).on_snapshot(callback)