import json
from typing import Any

import requests
from loguru import logger

from second_brain_offline.config import settings
from second_brain_offline.domain import DocumentMetadata


class NotionDatabaseClient():
    """Client to interact with Notion Databases.
    
    This class provides methods to query notion databases andprocess returned data
    
    Attributes : 
        api_key : Notion API secret key for user authentication.
    """
    def __init__(self, api_key : str = settings.NOTION_SECRET_KEY):
        """"Initialize the NotionDatabaseClient
        
        Arguments:
            api_key: Optional Notion API key, if not provided will take from settings.Notion_SECRET_KEY.
        """
        
        assert api_key is not None, (
            "Notion_SECRET_KEY environment variable is required, set it in .env file."
        )
        
        self.api_key = api_key
        
    def query_notion_database(
        self, database_id : str, query_object : str | None = None
        ) ->list[DocumentMetadata]:
        """Query notion database and returns results
        
        Args:
            database_id : ID of notion databas eto query
            query_object : Option JSON string as a query params
            
        Returns:
            List of dictionaries containing query results.
        """
        url = f"https://api.notion.com/v1/databases/{database_id}/query"
        
        headers = {
            "Authorization" : f"Bearer {self.api_key}",
            "Content-Type" : "application/json",
            "Notion-Version" : "2022-06-28"
        }
        
        if query_object and query_object.strip():
            try:
                query_payload = json.loads(query_object)
            except json.JSONDecodeError:
                logger.opt(exception=True).debug("Invalid JSON format for query.")
                return []
        
        try: 
            response = requests.post(
                url = url,
                headers=headers,
                json = query_payload,
                timeout = 10)
            response.raise_for_status # raise exception if any execption occurs.
            results = response.json()
            results = results["results"]
        except requests.exceptions.RequestException:
            logger.opt(exception=True).debug("Error quering Notion Database")
            return []
        except KeyError:
            logger.opt(exception=True).debug("Invalid Format from Notion Database")
            return []
        except Exception:
            logger.opt(exception=True).debug("Error quering Notion Database")
            return []
        
        return [self.__build_page_metadata(page) for page in results]
        
    def __build_page_metadata(self, page : dict[str, Any]) ->DocumentMetadata:
        """Build page metadata from notion page dictionary.

        Args:
            page (dict[str, Any]): page from notion database after quering.

        Returns:
            DocumentMetadata: Page metadata with processed data.
        """
        properties = self.__flatten_properties(page.get("properties", {}))
        title = properties.pop("Name") # as we already have flattened dict and got values 
        
        if page["parent"]:
            properties["parent"] = {
                "id" : properties["parent"]["id"],
                "url" : "",
                "title" : "",
                "properties" : {}
            }
        
        return DocumentMetadata(
            id=page["id"], url=page["url"], title=title, properties=properties
        )
        
        
    def __flatten_properties(self, properties : dict) -> dict:
        """Flatten the properties dictionary from notion to simpler key-value format.

        Args:
            properties (dict): notion properties dictionary to flatten

        Returns:
            dict: Flattened dictionary with key value pair.
        """
        flattened = {}
        
        # notion query results have properties and for types like text - "title", "description" has broken strings as each string can have different color, font and all.
        
        for key, value in properties.items():
            prop_type = value.get("type")   

            if prop_type == "select":
                select_value = value.get("select", {}) or {}
                flattened[key] = select_value.get("name")
            elif prop_type == "multi_select":
                flattened[key] = [
                    item.get("name") for item in value.get("multi_select", [])
                ]
            elif prop_type == "title":
                flattened[key] = "\n".join(
                    item.get("plain_text", "") for item in value.get("title", [])
                )
            elif prop_type == "rich_text":
                flattened[key] = " ".join(
                    item.get("plain_text", "") for item in value.get("rich_text", [])
                )
            elif prop_type == "number":
                flattened[key] = value.get("number")
            elif prop_type == "checkbox":
                flattened[key] = value.get("checkbox")
            elif prop_type == "date":
                date_value = value.get("date", {})
                if date_value:
                    flattened[key] = {
                        "start": date_value.get("start"),
                        "end": date_value.get("end"),
                    }
            elif prop_type == "database_id":
                flattened[key] = value.get("database_id")
            else:
                flattened[key] = value

        return flattened   
        

        
            
                