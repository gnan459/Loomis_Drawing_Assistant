import os
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

logger = logging.getLogger(__name__)

def init_qdrant(url: str = None, timeout: int = 5):
    """
    Initialize Qdrant client with error handling.
    
    Args:
        url: Qdrant server URL (default: env var QDRANT_URL or http://localhost:6333)
        timeout: Connection timeout in seconds
    
    Returns:
        QdrantClient instance
    
    Raises:
        ConnectionError: If unable to connect to Qdrant server
    """
    
    # Get URL from parameter, environment variable, or default
    qdrant_url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
    
    try:
        # Create client with timeout
        client = QdrantClient(url="http://localhost:6333", timeout=timeout)
        
        # Verify connection by listing collections
        client.get_collections()
        logger.info(f"✓ Connected to Qdrant at {qdrant_url}")
        
    except Exception as e:
        logger.error(f"✗ Failed to connect to Qdrant at {qdrant_url}: {str(e)}")
        raise ConnectionError(f"Cannot connect to Qdrant server at {qdrant_url}") from e
    
    # Create collection if it doesn't exist
    try:
        collections = [c.name for c in client.get_collections().collections]
        
        if "loomis_memory" not in collections:
            logger.info("Creating 'loomis_memory' collection...")
            client.create_collection(
                collection_name="loomis_memory",
                vectors_config=VectorParams(
                    size=4,
                    distance=Distance.COSINE
                )
            )
            logger.info("✓ Collection 'loomis_memory' created successfully")
        else:
            logger.info("✓ Collection 'loomis_memory' already exists")
            
    except Exception as e:
        logger.error(f"✗ Failed to create/verify collection: {str(e)}")
        raise
    
    return client