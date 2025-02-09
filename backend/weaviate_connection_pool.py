import os
from typing import List
import weaviate
from weaviate import WeaviateClient
from weaviate.auth import AuthApiKey

class WeaviateConnectionPool:
    def __init__(self, max_connections: int = 5):
        self.max_connections = max_connections
        self.connections: List[WeaviateClient] = []

    def get_connection(self) -> WeaviateClient:
        if len(self.connections) < self.max_connections:
            return self.create_new_connection()
        return self.connections.pop(0)

    def release_connection(self, conn: WeaviateClient):
        if len(self.connections) < self.max_connections:
            self.connections.append(conn)

    def create_new_connection(self) -> WeaviateClient:
        weaviate_url = os.getenv("WEAVIATE_URL")
        weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
        
        if os.getenv('USE_CLOUD'):
            return weaviate.connect_to_weaviate_cloud(
                cluster_url=weaviate_url,
                auth_credentials=AuthApiKey(api_key=weaviate_api_key),
                headers={
                    "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
                }
            )
        else:
            return weaviate.connect_to_embedded()

# DataManager クラス内での使用例
class DataManager:
    def __init__(self):
        self.connection_pool = WeaviateConnectionPool()

    def perform_weaviate_operation(self):
        conn = self.connection_pool.get_connection()
        try:
            # Weaviate操作を実行
            result = conn.some_operation()
            return result
        finally:
            self.connection_pool.release_connection(conn)
