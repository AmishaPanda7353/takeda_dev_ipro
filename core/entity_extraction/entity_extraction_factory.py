from core.entity_extraction.entity_extraction_client import EntityExtractionClient


class EntityExtractionFactory:

    """
    A factory class for EntityExtractionClient.

    Args:
        entity_extraction_type (EntityExtractionClient): The type of entity_extraction to create.

    Attributes:
        entity_extraction_type (EntityExtractionClient): The type of entity_extraction.

    Methods:
        get_entities: Function to find entities.
    """

    def __init__(self, entity_extraction_type: EntityExtractionClient) -> None:
        self.entity_extraction_type = entity_extraction_type

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

        return self.entity_extraction_type.get_entities_for_question(text)