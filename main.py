import asyncio
import os
import time
import uuid
from typing import Union
from pymongo import MongoClient
from datetime import datetime

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Redis
# from langchain_huggingface import HuggingFaceEndpointEmbeddings  # If using TEI
from comps.retriever.redis_config import EMBED_MODEL, INDEX_NAME, INDEX_SCHEMA, REDIS_URL


from comps import (  # Assuming these are defined in your comps.py
    CustomLogger,
    EmbedDoc,
    EmbedMultimodalDoc,
    SearchedDoc,
    SearchedMultimodalDoc,
    ServiceType,
    TextDoc,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)
from comps.proto.api_protocol import (  # Assuming this is where your proto definitions are
    ChatCompletionRequest,
    EmbeddingResponse,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalResponseData,
)

# ... (rest of your code as before) ...


async def test_retriever():
    retriever = ChatRetriever()

    # 1. Test create_new_chat_log
    question = "What is the capital of France?"
    chat_log = await retriever.create_new_chat_log(question)
    print("create_new_chat_log result:", chat_log)
    assert chat_log["question"] == question  # Basic assertion

    # 2. Test retrieve_chat_log (by conversationID)
    retrieved_by_convoid = retriever.retrieve_chat_log(convoid=chat_log["conversationID"])
    print("retrieve_chat_log by convoid:", retrieved_by_convoid)
    assert retrieved_by_convoid["question"] == question

    # 3. Test retrieve_chat_log (by uuid)
    retrieved_by_uuid = retriever.retrieve_chat_log(uuid=chat_log["uuid"])
    print("retrieve_chat_log by uuid:", retrieved_by_uuid)
    assert retrieved_by_uuid["question"] == question

    # 4. Test retrieve_closest_sentence
    sentence = "Largest city in France" # Slightly different wording to test retrieval
    closest_sentence = await retriever.retrieve_closest_sentence(sentence)
    print("retrieve_closest_sentence:", closest_sentence)
    assert closest_sentence is not None # You might want a more specific assert here once you know what to expect

    # 5. Test find_similar_chats
    similar_chats = retriever.find_similar_chats("capital") # Search for "capital"
    print("find_similar_chats:", similar_chats)
    assert len(similar_chats) > 0 # At least one similar chat should exist

    # 6. Test the retrieve function (using RetrievalRequest)
    retrieval_request = RetrievalRequest(text="What is the population of Paris?")
    retrieval_response = await retrieve(retrieval_request)
    print("retrieve (RetrievalRequest):", retrieval_response)

    # 7. Test the retrieve function (using EmbedDoc)
    embed_doc = EmbedDoc(text="Famous French landmarks")
    embed_response = await retrieve(embed_doc)
    print("retrieve (EmbedDoc):", embed_response)

    print("All tests passed!")


async def main():
    await test_retriever()
    opea_microservices["opea_service@retriever_redis"].start() # Keep this if you're running the microservice

if __name__ == "__main__":
    asyncio.run(main())