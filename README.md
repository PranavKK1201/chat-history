# Chat Retriever API

Simple API for retrieving and managing chat conversations with semantic search capabilities.

## Prerequisites

- Python 3.8+
- Redis server
- MongoDB server
- Required Python packages (install via `pip install -r requirements.txt`):
  ```
  flask
  flask-cors
  redis
  pymongo
  sentence-transformers
  numpy
  ```

## Setup & Run

1. Start Redis and MongoDB servers
2. Run the API:
   ```bash
   python app.py
   ```

## API Endpoints

### Create Chat
```bash
curl -X POST http://127.0.0.1:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Tell me about birds"}'
```

### Get Conversation History
```bash
curl http://127.0.0.1:5000/api/chat/<conversation_id>
```

### Delete Conversation
```bash
curl -X DELETE http://127.0.0.1:5000/api/chat/<conversation_id>
```

Replace `<conversation_id>` with the actual conversation ID returned from the create chat endpoint.
