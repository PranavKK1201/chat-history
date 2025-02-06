import os
import uuid
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime

import redis
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

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

    def create_chat_log(self, input_text: str) -> Dict:
        """Create a new chat log with retrieved context."""
        retrieved_docs = self.retrieve_similar_docs(input_text)
        
        chat_log = {
            "uuid": str(uuid.uuid4()),
            "conversationID": str(uuid.uuid1()),
            "question": input_text,
            "answer": retrieved_docs[0]['text'] if retrieved_docs else "",
            "rating": None,
            "context": [doc['text'] for doc in retrieved_docs],
            "similarity_score": retrieved_docs[0]['score'] if retrieved_docs else None,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save to MongoDB
        self.chat_collection.insert_one(chat_log)
        
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
        
        return self.chat_collection.find_one(query)

    def close_connections(self):
        """Close database connections."""
        self.redis_client.close()
        self.mongo_client.close()


def main():
    retriever = ChatRetriever()
    chat_log = retriever.create_chat_log("Give me a bird fact")
    retrieved_log = retriever.retrieve_chat_log(conversation_id=chat_log['conversationID'])
    print(retrieved_log)
    retriever.close_connections()

if __name__ == "__main__":
    main()
