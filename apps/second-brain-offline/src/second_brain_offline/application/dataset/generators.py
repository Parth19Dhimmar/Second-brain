import copy
from loguru import logger
from typing_extensions import Callable

from second_brain_offline.application.agents import SummarizationAgent
from second_brain_offline.domain import Document, InstructDatasetSample, InstructDataset

class SummarizationDatasetGenerator:
    def __init__(
        self,
        summarization_model: str,
        summarization_max_characters: int,
        val_split_ratio: float = 0.1,
        test_split_ratio: float = 0.1,
        max_workers: int = 10,
        mock: bool = False,
        min_document_length: int = 50,
        min_quality_score: float = 0.2,
        max_summary_length_factor: float = 2,
        augmentation_loops: int = 4,
    ):
    
        self.summarization_model = summarization_model
        self.summarization_max_characters = summarization_max_characters
        self.val_split_ratio = val_split_ratio
        self.test_split_ratio = test_split_ratio
        self.max_workers = max_workers
        self.mock = mock
        self.min_document_length = min_document_length
        self.min_quality_score = min_quality_score
        self.max_summary_length_factor = max_summary_length_factor
        self.augmentation_loops = augmentation_loops
        
        self.pregeneration_filters: list[Callable[[Document], bool]] = [
            lambda document: len(document.content) > self.min_document_length,
            lambda document: document.content_quality_score is not None and 
                document.content_quality_score > self.min_quality_score
        ]
        
        self.postgeneration_filters: list[Callable[[Document], bool]] = [
            lambda document: document.summary is not None
            and len(document.summary) < self.summarization_max_characters * self.max_summary_length_factor 
        ]
        
    
    def generate(
        self, documents: list[Document]
    ) -> InstructDataset: 
        """Generates an instruction dataset from the documents.

        Filters, summarizes documents and converts them into instruction-answer pairs.
        Warns if input document count is less than recommended minimum of 10.

        Args:
            documents: List of Document objects to be processed into the dataset.

        Returns:
            InstructDataset containing instruction-answer pairs where instructions are
            document contents and answers are generated summaries.

        Warns:
            If less than 10 documents are provided for processing.
        """
        
        if len(documents) < 10:
            logger.warning(
                "Less than 10 documents to summarize. For accurate behavior we recommend having at least 10 documents."
            )
        
        filtered_summarized_documents = self.__summarize_documents(
            documents=documents
        )
        
        huggingface_dataset_sample = [  
            self.__to_huggigface_dataset_sample(summarized_document)
            for summarized_document in filtered_summarized_documents
        ]
        
        return InstructDataset.from_sample(
            samples=huggingface_dataset_sample,
            val_split_ratio=self.val_split_ratio,
            test_split_ratio=self.test_split_ratio,
            seed=42
        )
    
    def __summarize_documents(
        self,
        documents: list[Document],
    ) -> list[Document]: 
        
        total_docs = len(documents)

        logger.info(f"Num documents before pregeneration filtering: {len(documents)}")
        
        prefiltered_documents = self.__filter_documents(
            self.pregeneration_filters, 
            documents,
        )
        
        logger.info(
            f"Num documents after pregeneration filtering: {len(prefiltered_documents)}"
        )        
        
        augmented_documents = self.__augmentation_summarization_loops(
            documents=prefiltered_documents,
            loops=self.augmentation_loops,  
        )
        
        logger.info(f"Document summarization with augmentation loops completed, total augmented documents : {len(augmented_documents)}")
        
        postfiltered_documents = self.__filter_documents(
            filters=self.postgeneration_filters, 
            documents=augmented_documents,
        )
        
        logger.info(f"Filtered {total_docs - postfiltered_documents} documents, remaining: {len(postfiltered_documents)}")

        return postfiltered_documents
    
    def __augmentation_summarization_loops(
        self,
        documents: list[Document],
        loops: int = 3,
    ) -> list[Document]:
        
        summarizer_agent = SummarizationAgent(
            max_characters=self.summarization_max_characters,
            model_id=self.summarization_model,
            mock=self.mock,
            concurrent_requests=self.max_workers
        )
        
        augmented_documents = []
        
        for i in range(loops):
            temperature = i * (0.5/loops) # 0.5
            
            logger.info(
                f"Loop {i + 1} of {loops} - Summarizing documents with temperature {temperature}"
            )
            
            copied_documents = copy.deepcopy(documents)
            
            summarized_documents = summarizer_agent(
                documents=copied_documents,
                temperature=temperature
            )
            logger.info(f"summarized_documents : {summarized_documents}")
            valid_documents = [doc for doc in summarized_documents if doc.summary is not None]
            augmented_documents.extend(valid_documents)
            
        return augmented_documents
    
    def __filter_documents(
        self,
        filters : list[Callable[[Document], bool]],
        documents: list[Document]
    ) -> list[Document]:
        """Filters documents using provided filter functions.

        Args:
            filters: List of filter functions that take a Document and return bool.
            documents: List of documents to filter.

        Returns:
            List of documents that pass all filter functions.
        """
        
        before = len(documents)
        for filter in filters:
            documents = [doc for doc in documents if filter(doc)]
        after = len(documents)
        
        logger.info(f"Documents Before applying filter : {before}, Documents after applying filter: {after}")
        return documents
    
    def __to_huggigface_dataset_sample(
        self,
        document: Document
    ) -> InstructDataset:
        """Converts a summarized document to an instruction dataset sample.

        Args:
            document: A Document object containing both content and summary.

        Returns:
            InstructDatasetSample with document content as instruction and
            summary as answer.

        Raises:
            AssertionError: If the document's summary is None.
        """
        
        assert document.summary is not None, "Document summary is None."
        
        return InstructDatasetSample(
            instruction=document.content,
            answer=document.summary,
        ) 

        
        