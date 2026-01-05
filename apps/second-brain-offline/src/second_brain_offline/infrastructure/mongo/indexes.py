from langchain_mongodb.index import create_fulltext_search_index

from second_brain_offline.infrastructure.mongo import MongoDBService

class MongoDBIndex():
    
    def __init__(
        self,
        retriever,
        mongodb_client: MongoDBService
        ) -> None:
            self.retriever = retriever
            self.mongodb_client = mongodb_client

    def create(
        self, 
        embedding_dim: int,
        is_hybrid: bool
        ) -> None:
        vector_store = self.retriever.vectorstore
        
        vector_store.create_vector_search_index(
            dimensions=embedding_dim,
        )
        
        if is_hybrid:
            create_fulltext_search_index(
                collection=self.mongodb_client.collection,
                field=vector_store._text_key,
                index_name=self.retriever.search_index_name,
            )
            
        
        
        