def retrieve_similar(client, query_vector, limit=5):
    return client.search(
        collection_name="loomis_memory",
        query_vector=query_vector,
        limit=limit,
        with_payload=True
    )
