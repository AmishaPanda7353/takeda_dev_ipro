import logging
import sys

from keybert import KeyBERT

from core.entity_extraction.entity_extraction_client import EntityExtractionClient


class KeyBert(EntityExtractionClient):
    """
    A class for interacting with KeyBERT EntityExtractionClient.

    This class inherits from the `EntityExtractionClient` base class and provides methods for connecting to KeyBERT EntityExtraction and
    finding entities for text.

    Parameters
    ----------
    config (dict): The configuration object containing entityextractio information.
    data_frame (pd.Dataframe): Dataframe having unique index for unique colum name and other information related to that column,
                                e.g. column description, table name etc.

    """

    def __init__(self, config, data_frame) -> None:
        super().__init__(config, data_frame)
        self.model = KeyBERT('all-MiniLM-L6-v2')
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            stream=sys.stdout,
        )

        self.logger = logging.getLogger("keybert_entity_extraction_logger")


    def get_entities(self, text):
        """
        Function to find entities.
        Parameters
        ----------
        text : str

        Returns
        -------
        List
            List of entities
        """

        try:
            entities = []
            for i in range(1, 3):
                text_entities = self.model.extract_keywords(
                    text, keyphrase_ngram_range=(1, i), top_n = 5,stop_words='english',
                )
                entities.extend(text_entities)
            self.logger.info("Entities created for question using KeyBert model")
            return entities
        except Exception as e:
            self.logger.error(
                f"Error while creating entities for question using KeyBert model. Error : {e}"
            )
