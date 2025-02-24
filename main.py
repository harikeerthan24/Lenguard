import pandas as pd
from openai import OpenAI
import os
import dotenv
import json
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_openai_client():
    """Initialize OpenAI client with proper error handling and configuration."""
    try:
        # Load environment variables
        dotenv.load_dotenv()
        
        # Get API key with validation
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        
        # Initialize client with configuration
        client = OpenAI(
            api_key=api_key,
            timeout=60.0,  # Increase timeout for longer operations
            max_retries=3  # Automatic retries for reliability
        )
        
        logger.info("OpenAI client initialized successfully")
        return client
    
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {str(e)}")
        raise

def extract_metadata_with_gpt(file_path: str) -> dict:
    """
    Extracts metadata from an Excel rent roll file that might be located anywhere in the file,
    then refines it using OpenAI GPT to handle complex edge cases.
    
    Args:
        file_path (str): Path to the Excel file.
    
    Returns:
        dict: Extracted metadata as structured fields.
    """
    try:
        # Validate file path
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Initialize OpenAI client
        client = initialize_openai_client()
        
        # Load the Excel file
        xls = pd.ExcelFile(file_path)
        sheet_name = xls.sheet_names[0]  # Assume the first sheet is relevant
        df = pd.read_excel(xls, sheet_name=sheet_name, dtype=str)
        
        # Aggregate all non-empty rows as potential metadata
        all_rows = []
        for i, row in df.iterrows():
            non_empty_values = row.dropna().to_list()
            if non_empty_values:
                all_rows.append(" | ".join(non_empty_values))
        all_text = "\n".join(all_rows)
        
        # Prepare GPT prompt with detailed instructions
        system_prompt = {
            "role": "system",
            "content": (
                "You are an AI that extracts structured metadata from rent roll documents. "
                "The provided text contains both metadata and tabular rent roll data. "
                "Ignore the tabular data and extract only metadata such as the property name, report date, "
                "rent roll month/year, and any additional parameters. Format the output as valid JSON using the structure:\n\n"
                '{\n'
                '  "property_name": <string>,\n'
                '  "report_date": <string>,\n'
                '  "report_timestamp": <string>,\n'
                '  "additional_info": { ... }\n'
                '}'
                
                "Remember: DO NOT include any totals, financial summaries, or detailed data. Only extract metadata."
            )
        }
        
        user_prompt = {
            "role": "user",
            "content": f"Extract metadata from the following rent roll content:\n\n{all_text}"
        }
        
        # Call OpenAI API with error handling
        try:
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[system_prompt, user_prompt],
                temperature=0.3,  # Lower temperature for consistent output
                response_format={"type": "json_object"}  # Request JSON response
            )
            
            extracted_metadata = response.choices[0].message.content
            return json.loads(extracted_metadata)
            
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise
            
    except Exception as e:
        logger.error(f"Error in metadata extraction: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        file_path = input("Please enter the path to your rent roll Excel file: ")
        metadata_output = extract_metadata_with_gpt(file_path)
        
        print("\nExtracted Metadata:")
        print(json.dumps(metadata_output, indent=2))
        
    except Exception as e:
        logger.error(f"Program execution failed: {str(e)}")
