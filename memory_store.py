import uuid

def store_in_qdrant(client, vector, payload):
    client.upsert(
        collection_name="loomis_memory",
        points=[
            {
                "id": uuid.uuid4().hex,
                "vector": vector,
                "payload": payload
            }
        ]
    )
