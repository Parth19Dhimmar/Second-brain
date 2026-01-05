# mongo_uri, 
# database_name
# collection_name
from typing import Generic, Type, TypeVar
from loguru import logger
from bson import ObjectId

from pydantic import BaseModel
from pymongo import MongoClient, errors

from second_brain_offline import settings

T = TypeVar("T", bound=BaseModel)

class MongoDBService(Generic[T]):   
    """Service class for MongoDB operations, supporting ingestion, querying, and validation.

    This class provides methods to interact with MongoDB collections, including document
    ingestion, querying, and validation operations.

    Args:
        model: The Pydantic model class to use for document serialization.
        collection_name: Name of the MongoDB collection to use.
        database_name: Name of the MongoDB database to use.
        mongodb_uri: URI for connecting to MongoDB instance.

    Attributes:
        model: The Pydantic model class used for document serialization.
        collection_name: Name of the MongoDB collection.
        database_name: Name of the MongoDB database.
        mongodb_uri: MongoDB connection URI.
        client: MongoDB client instance for database connections.
        database: Reference to the target MongoDB database.
        collection: Reference to the target MongoDB collection.
    """
    
    def __init__(
        self,
        model: Type[T],
        collection_name: str,
        database_name: str = settings.MONGODB_DATABASE_NAME,
        mongo_uri: str = settings.MONGODB_URI,
    ) -> None:
        
        self.model = model
        self.mongo_uri = mongo_uri
        self.database_name = database_name
        self.collection_name = collection_name
        
        try:
            self.client = MongoClient(
                self.mongo_uri,
                appname="second_brain_ai_assistant"  
            )
            self.client.admin.command('ping')
        except Exception as e:
            logger.error(f"failed to initialize MongoDBService.")
            
        self.database = self.client[database_name]
        self.collection = self.database[collection_name]
        
    def __enter__(self) -> "MongoDBService": # whenever the mongoservice initialized this runs
        """Enable context manager support.

        Returns:
            MongoDBService: The current instance.
        """
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close MongoDB connection when exiting context.

        Args:
            exc_type: Type of exception that occurred, if any.
            exc_val: Exception instance that occurred, if any.
            exc_tb: Traceback of exception that occurred, if any.
        """
        self.close()
    
    def clear_collection(self):
        """Remove all documents from the collection.

        This method deletes all documents in the collection to avoid duplicates
        during reingestion.

        Raises:
            errors.PyMongoError: If the deletion operation fails.
        """
        
        try:
            result = self.collection.delete_many({})
            logger.debug(f"Cleared collection. Deleted {result.deleted_count} documents")
        except errors.PyMongoError as e:
            logger.error(f"Error Clearing the collection.")
    
    def ingest_documents(self, documents: list[T]) -> None:
        """Insert multiple documents into the MongoDB collection.

        Args:
            documents: List of Pydantic model instances to insert.

        Raises:
            ValueError: If documents is empty or contains non-Pydantic model items.
            errors.PyMongoError: If the insertion operation fails.
        """
        
        try:
            if not documents or not all(
                isinstance(doc, BaseModel) for doc in documents
            ):
                raise ValueError(f"Documents must be list of pydantic models")
            
            dict_documents = [doc.model_dump() for doc in documents]
            
            # Remove '_id' fields to avoid duplicate key errors
            for doc in dict_documents:
                doc.pop("_id", None)
            
            result = self.collection.insert_many(dict_documents)
            logger.debug(f"Inserted {len(result.inserted_ids)} documents into MongoDB.")
        except errors.PyMongoError as e:
            logger.error(f"Error Inserting documents to the collection.")
            raise
        
            
    def fetch_documents(self, limit: int, query: dict) -> list[T]:
        """Retrieve documents from the MongoDB collection based on a query.

        Args:
            limit: Maximum number of documents to retrieve.
            query: MongoDB query filter to apply.

        Returns:
            List of Pydantic model instances matching the query criteria.

        Raises:
            Exception: If the query operation fails.
        """
        try:
            documents = list(self.collection.find(query).limit(limit))
            logger.debug(f"Fetched {len(documents)} documents from MongoDB.")
            logger.info(f"fetched_documents : {len(documents)}")
            parsed_documents = self.__parse_documents(documents=documents)
            logger.info(f"parsd_doc:  {len(parsed_documents)}")
            
            return parsed_documents
            
        except Exception as e: 
            logger.error(f"Error fetching documents from MongoDB.", str(e))
            raise
        
    def __parse_documents(self, documents: list[dict]) -> list[T]:
        """Convert MongoDB documents to Pydantic model instances.

        Converts MongoDB ObjectId fields to strings and transforms the document structure
        to match the Pydantic model schema.

        Args:
            documents: List of MongoDB documents to parse.

        Returns:
            List of validated Pydantic model instances.
        """
        
        parsed_documents = []
        for document in documents:
            for key, value in document.items():
                if isinstance(value, ObjectId):
                    document[key] = str(value)
            
            _id = document.pop("_id", None)
            document["id"] = _id
            
            parsed_doc = self.model.model_validate(document)
            parsed_documents.append(parsed_doc)
            
        return parsed_documents
                
            
    def get_collection_count(self) -> int:
        """Count the total number of documents in the collection.

        Returns:
            Total number of documents in the collection.

        Raises:
            errors.PyMongoError: If the count operation fails.
        """
        try:
            return self.collection.count_documents({})
        except errors.PyMongoError as e:
            logger.error(f"Error getting documents count for the collection: {self.collection_name}.")
            raise
    
    def close(self) -> None:
        """Close the MongoDB connection.

        This method should be called when the service is no longer needed
        to properly release resources, unless using the context manager.
        """

        self.client.close()
        logger.debug("Closed MongoDB connection.")