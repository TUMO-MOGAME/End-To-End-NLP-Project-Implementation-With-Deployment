import os
from textSummarizer.logging import logger
from textSummarizer.entity import DataValidationConfig

class data_validation:
    def __init__(self,config:DataValidationConfig):
        self.config = config

    def validate_all_files_exist(self)-> bool:
        try:
            validation_status = None
            all_files = os.listdir(os.path.join("artifacts", "data_ingestion", "samsum_dataset"))

            # Check if all required files exist
            validation_status = True
            for required_file in self.config.ALL_REQUIRED_FILES:
                if required_file not in all_files:
                    validation_status = False
                    break

            # Write the final status
            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation status: {validation_status}")

            return validation_status
        except Exception as e:
            raise e