from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
from bson import ObjectId
import json

import redis
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# Add JSONEncoder to handle ObjectId
class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return json.JSONEncoder.default(self, o)

app = Flask(__name__)
# Configure Flask to use the custom JSON encoder
app.json_encoder = JSONEncoder
CORS(app)

class ChatRetriever:
    def __init__(
        self, 
        redis_host: str = 'localhost', 
        redis_port: int = 6379,
        mongodb_uri: str = 'mongodb://localhost:27017/',
        embedding_model: str = 'all-MiniLM-L6-v2'
    ):
        # Redis connection
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        
        # MongoDB connection
        self.mongo_client = MongoClient(mongodb_uri)
        self.mongo_db = self.mongo_client['chat_database']
        self.chat_collection = self.mongo_db['conversations']
        
        # Embedding model
        self.model = SentenceTransformer(embedding_model)
        
        # Clear databases at startup
        self.clear_databases()
        
        # Predefine sample texts
        self.predefined_texts = [
            "Machine learning is a subset of artificial intelligence",
            "Birds are warm-blooded vertebrates",
            "Cleopatra was a pharaoh of Egypt",
            "Football was invented in England"
        ]
        
        # Store predefined embeddings in Redis
        for text in self.predefined_texts:
            self.add_to_redis(text)
    
    def clear_databases(self):
        """Clears Redis and MongoDB collections at startup."""
        self.redis_client.flushdb()
        self.chat_collection.delete_many({})
        print("Databases cleared.")

    def cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors."""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def retrieve_similar_docs(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for similar documents using cosine similarity."""
        query_embedding = self.model.encode(query).astype('float32')
        results = []
        
        for key in self.redis_client.keys('chat:*'):
            embedding_bytes = self.redis_client.hget(key, 'embedding')
            if embedding_bytes:
                stored_embedding = np.frombuffer(bytes.fromhex(embedding_bytes), dtype=np.float32)
                similarity = self.cosine_similarity(query_embedding, stored_embedding)
                
                results.append({
                    'text': self.redis_client.hget(key, 'text'),
                    'score': float(similarity)
                })
        
        return sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]

    def create_chat_log(self, input_text: str, conversation_id: Optional[str] = None) -> Dict:
        """Create a new chat log with retrieved context."""
        retrieved_docs = self.retrieve_similar_docs(input_text)
        
        chat_log = {
            "uuid": str(uuid.uuid4()),
            "conversationID": conversation_id if conversation_id else str(uuid.uuid1()),
            "question": input_text,
            "answer": retrieved_docs[0]['text'] if retrieved_docs else "",
            "rating": None,
            "context": [doc['text'] for doc in retrieved_docs],
            "similarity_score": retrieved_docs[0]['score'] if retrieved_docs else None,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save to MongoDB and get the inserted document
        result = self.chat_collection.insert_one(chat_log)
        chat_log['_id'] = str(result.inserted_id)  # Convert ObjectId to string before returning
        
        return chat_log

    def add_to_redis(self, text: str, metadata: Optional[Dict] = None):
        """Add only predefined text embeddings to Redis."""
        chat_id = f"chat:{str(uuid.uuid4())}"
        embedding = self.model.encode(text).astype('float32')
        
        doc = {
            'text': text,
            'metadata': str(metadata or {}),
            'embedding': embedding.tobytes().hex()
        }
        
        self.redis_client.hset(chat_id, mapping=doc)

    def retrieve_chat_log(self, conversation_id: str = None, uuid_str: str = None) -> Optional[Dict]:
        """Retrieve a specific chat log from MongoDB."""
        query = {}
        if conversation_id:
            query['conversationID'] = conversation_id
        if uuid_str:
            query['uuid'] = uuid_str

        chat_log = self.chat_collection.find_one(query)
        
        if chat_log:
            chat_log['_id'] = str(chat_log['_id'])

        return chat_log

    def retrieve_conversation_history(self, conversation_id: str) -> List[Dict]:
        """Retrieve all chat logs for a specific conversation."""
        chat_logs = list(self.chat_collection.find({'conversationID': conversation_id}))
        
        # Convert ObjectId to string for each document
        for log in chat_logs:
            log['_id'] = str(log['_id'])

        return chat_logs

    def close_connections(self):
        """Close database connections."""
        self.redis_client.close()
        self.mongo_client.close()

# Initialize the ChatRetriever
retriever = ChatRetriever()

@app.route('/api/chat', methods=['POST'])
def create_chat():
    data = request.json
    question = data.get('question')
    conversation_id = data.get('conversation_id')
    
    if not question:
        return jsonify({"error": "Question is required"}), 400
    
    chat_log = retriever.create_chat_log(question, conversation_id)
    return jsonify(chat_log), 201

@app.route('/api/chat/<conversation_id>', methods=['GET'])
def get_conversation(conversation_id):
    conversation_history = retriever.retrieve_conversation_history(conversation_id)
    if not conversation_history:
        return jsonify({"error": "Conversation not found"}), 404
    
    return jsonify(conversation_history), 200

@app.route('/api/chat/<conversation_id>/<uuid_str>', methods=['GET'])
def get_chat_log(conversation_id, uuid_str):
    chat_log = retriever.retrieve_chat_log(conversation_id, uuid_str)
    if not chat_log:
        return jsonify({"error": "Chat log not found"}), 404
    
    return jsonify(chat_log), 200

@app.route('/api/chat/<conversation_id>', methods=['DELETE'])
def delete_conversation(conversation_id):
    result = retriever.chat_collection.delete_many({'conversationID': conversation_id})
    if result.deleted_count == 0:
        return jsonify({"error": "Conversation not found"}), 404
    
    return jsonify({"message": "Conversation deleted"}), 200

if __name__ == "__main__":
    app.run(debug=True)