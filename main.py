import pandas as pd
from openai import OpenAI
import os
import dotenv
import json
from pathlib import Path
import logging

# # # Setup logging
# # # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# # logger = logging.getLogger(__name__)

# # def initialize_openai_client():
# #     """Initialize OpenAI client with proper error handling and configuration."""
# #     try:
# #         # Load environment variables
# #         dotenv.load_dotenv()
        
# #         # Get API key with validation
# #         api_key = os.getenv("OPENAI_API_KEY")
# #         if not api_key:
# #             raise ValueError("OpenAI API key not found in environment variables")
        
# #         # Initialize client with configuration
# #         client = OpenAI(
# #             api_key=api_key,
# #             timeout=60.0,  # Increase timeout for longer operations
# #             max_retries=3  # Automatic retries for reliability
# #         )
        
# #         logger.info("OpenAI client initialized successfully")
# #         return client
    
# #     except Exception as e:
# #         logger.error(f"Failed to initialize OpenAI client: {str(e)}")
# #         raise

# # def extract_metadata_with_gpt(file_path: str) -> dict:
# #     """
# #     Extracts metadata from an Excel rent roll file that might be located anywhere in the file,
# #     then refines it using OpenAI GPT to handle complex edge cases.
    
# #     Args:
# #         file_path (str): Path to the Excel file.
    
# #     Returns:
# #         dict: Extracted metadata as structured fields.
# #     """
# #     try:
# #         # Validate file path
# #         file_path = Path(file_path)
# #         if not file_path.exists():
# #             raise FileNotFoundError(f"File not found: {file_path}")
        
# #         # Initialize OpenAI client
# #         client = initialize_openai_client()
        
# #         # Load the Excel file
# #         xls = pd.ExcelFile(file_path)
# #         sheet_name = xls.sheet_names[0]  # Assume the first sheet is relevant
# #         df = pd.read_excel(xls, sheet_name=sheet_name, dtype=str)
        
# #         # Aggregate all non-empty rows as potential metadata
# #         all_rows = []
# #         for _ , row in df.iterrows():
# #             non_empty_values = row.dropna().to_list()
# #             if non_empty_values:
# #                 all_rows.append(" | ".join(non_empty_values))
# #         all_text = "\n".join(all_rows)
        
# #         # Prepare GPT prompt with detailed instructions
# #         system_prompt = {
# #             "role": "system",
# #             "content": (
# #                 "You are an AI that extracts structured metadata from rent roll documents. "
# #                 "The provided text contains both metadata and tabular rent roll data. "
# #                 "Ignore the tabular data and extract only metadata such as the property name, report date, "
# #                 "rent roll month/year, and any additional parameters. Format the output as valid JSON using the structure:\n\n"
# #                 '{\n'
# #                 '  "property_name": <string>,\n'
# #                 '  "report_date": <string>,\n'
# #                 '  "report_timestamp": <string>,\n'
# #                 '  "additional_info": { ... }\n'
# #                 '}'
                
# #                 "Remember: DO NOT include any totals, financial summaries, or detailed data. Only extract metadata."
# #             )
# #         }
        
# #         user_prompt = {
# #             "role": "user",
# #             "content": f"Extract metadata from the following rent roll content:\n\n{all_text}"
# #         }
        
# #         # Call OpenAI API with error handling
# #         try:
# #             response = client.chat.completions.create(
# #                 model="gpt-4-turbo-preview",
# #                 messages=[system_prompt, user_prompt],
# #                 temperature=0.3,  # Lower temperature for consistent output
# #                 response_format={"type": "json_object"}  # Request JSON response
# #             )
            
# #             extracted_metadata = response.choices[0].message.content
# #             return json.loads(extracted_metadata)
            
# #         except Exception as e:
# #             logger.error(f"OpenAI API error: {str(e)}")
# #             raise
            
# #     except Exception as e:
# #         logger.error(f"Error in metadata extraction: {str(e)}")
# #         raise

# # if __name__ == "__main__":
# #     try:
# #         file_path = input("Please enter the path to your rent roll Excel file: ")
# #         metadata_output = extract_metadata_with_gpt(file_path)
        
# #         print("\nExtracted Metadata:")
# #         print(json.dumps(metadata_output, indent=2))
        
# #     except Exception as e:
# #         logger.error(f"Program execution failed: {str(e)}")


# import pandas as pd
# import json
# import os
# import dotenv
# from openai import OpenAI
# import logging

# logger = logging.getLogger(__name__)


# def initialize_openai_client():
#     """Initialize OpenAI client with proper error handling and configuration."""
#     try:
#         # Load environment variables
#         dotenv.load_dotenv()
        
#         # Get API key with validation
#         api_key = os.getenv("OPENAI_API_KEY")
#         if not api_key:
#             raise ValueError("OpenAI API key not found in environment variables")
        
#         # Initialize client with configuration
#         client = OpenAI(
#             api_key=api_key,
#             timeout=60.0,  # Increase timeout for longer operations
#             max_retries=3  # Automatic retries for reliability
#         )
        
#         logger.info("OpenAI client initialized successfully")
#         return client
    
#     except Exception as e:
#         logger.error(f"Failed to initialize OpenAI client: {str(e)}")
#         raise
    
    
# def load_excel_data(file_path: str) -> pd.ExcelFile:
#     """Load Excel file from given path"""
#     return pd.ExcelFile(file_path)

# def parse_excel_data(excel_file: pd.ExcelFile) -> pd.DataFrame:
#     """Parse Excel file into DataFrame"""
#     return excel_file.parse()

# def find_header_row(data: pd.DataFrame, headers: list) -> int:
#     """Find the row containing any of the expected headers"""
#     for idx, row in data.iterrows():
#         row_str = ' '.join(str(val) for val in row.values)
#         # Check if ANY header is present instead of ALL
#         if any(header in row_str for header in headers):
#             # print(f"Headers found at row index: {idx}")
#             return idx
#     return None

# def extract_metadata(data: pd.DataFrame, header_index: int) -> dict:
#     """Extract and process metadata before header row"""
#     metadata = data.iloc[:header_index].copy()
#     metadata = metadata.dropna(how='all').reset_index(drop=True)
    
#     metadata_dict = {}
#     for idx, row in metadata.iterrows():
#         row_clean = row.dropna()
#         if not row_clean.empty:
#             metadata_dict[f"row_{idx}"] = row_clean.to_dict()
#     return metadata_dict

# def process_rent_roll_data(data: pd.DataFrame, header_index: int) -> pd.DataFrame:
#     """Process rent roll data after header row"""
#     data = data.iloc[header_index:].reset_index(drop=True)
#     data.columns = data.iloc[0]
#     rent_roll_data = data.iloc[1:].reset_index(drop=True)
#     return rent_roll_data.dropna(how='all').reset_index(drop=True)

# def main():
#     # Define expected headers - any one of these should be present
#     headers = [
#         'Resh ID', 'Lease ID', 'Unit', 'Floor Plan', 'Unit Designation', 'SQFT',
#         'Unit/Lease Status', 'Name', 'Phone Number', 'Email', 'Move-In', 
#         'Notice For Date', 'Move-Out', 'Lease Start', 'Lease End', 'Market + Addl.',
#         'Dep On Hand', 'Balance', 'Total Charges', 'RENT', 'PEST', 'VALET TRASH',
#         'TRASH', 'PETRENT', 'LOP15', 'EMPLCRED', 'OHIOCHOICE', 'WAIVED ADMIN FEE',
#         'WAIVED APP FEE', 'SETUPFEE'
#     ]
    
#     # Pipeline execution
#     file_path = input("Please enter the path to your rent roll Excel file: ")
#     excel_file = load_excel_data(file_path)
#     raw_data = parse_excel_data(excel_file)
#     header_index = find_header_row(raw_data, headers)
    
#     if header_index is not None:
#         # Extract raw metadata
#         metadata = extract_metadata(raw_data, header_index)
        
#         # Use GPT to analyze and structure the metadata
#         client = initialize_openai_client()
        
#         # Convert metadata to string format for GPT analysis
#         metadata_text = "\n".join([
#             " | ".join(str(v) for v in row.values()) 
#             for row in metadata.values()
#         ])
        
#         # Prepare prompts for GPT analysis
#         system_prompt = {
#             "role": "system", 
#             "content": """Analyze this rent roll metadata and extract key information into a structured format.
#             Focus on:
#             - Property name and details
#             - Report date/period
#             - Occupancy metrics
#             - Financial summaries
#             - Any other relevant property metrics
#             Format as detailed JSON with clear labels and categorized information."""
#         }
        
#         user_prompt = {
#             "role": "user",
#             "content": f"Analyze this rent roll metadata:\n\n{metadata_text}"
#         }
        
#         # Get GPT analysis
#         response = client.chat.completions.create(
#             model="gpt-4-turbo-preview",
#             messages=[system_prompt, user_prompt],
#             temperature=0.3,
#             response_format={"type": "json_object"}
#         )
        
#         # Parse GPT response and update metadata
#         analyzed_metadata = json.loads(response.choices[0].message.content)
#         metadata = {"raw_metadata": metadata, "analyzed_metadata": analyzed_metadata}
        
#         # Process rent roll data
#         rent_roll_data = process_rent_roll_data(raw_data, header_index)
        
#         print("Metadata:")
#         print(metadata.values())
        
#         return rent_roll_data
#     else:
#         raise ValueError("Could not find any of the expected headers in data")

# # Execute pipeline
# rent_roll_data = main()
# rent_roll_data


def load_excel_data(file_path: str) -> pd.ExcelFile:
    """Load Excel file from given path"""
    return pd.ExcelFile(file_path)

def parse_excel_data(excel_file: pd.ExcelFile) -> pd.DataFrame:
    """Parse Excel file into DataFrame"""
    return excel_file.parse()

def find_header_row(data: pd.DataFrame, headers: list) -> int:
    """Find the row containing any of the expected headers"""
    for idx, row in data.iterrows():
        row_str = ' '.join(str(val) for val in row.values)
        # Check if ANY header is present instead of ALL
        if any(header in row_str for header in headers):
            # print(f"Headers found at row index: {idx}")
            return idx
    return None

def extract_metadata(data: pd.DataFrame, header_index: int) -> dict:
    """Extract and process metadata before header row"""
    metadata = data.iloc[:header_index].copy()
    metadata = metadata.dropna(how='all').reset_index(drop=True)
    
    metadata_dict = {}
    for idx, row in metadata.iterrows():
        row_clean = row.dropna()
        if not row_clean.empty:
            metadata_dict[f"row_{idx}"] = row_clean.to_dict()
    return metadata_dict

def process_rent_roll_data(data: pd.DataFrame, header_index: int) -> pd.DataFrame:
    """Process rent roll data after header row"""
    data = data.iloc[header_index:].reset_index(drop=True)
    data.columns = data.iloc[0]
    rent_roll_data = data.iloc[1:].reset_index(drop=True)
    return rent_roll_data.dropna(how='all').reset_index(drop=True)

def main():
    # Define expected headers - any one of these should be present
    headers = [
        'Resh ID', 'Lease ID', 'Unit', 'Floor Plan', 'Unit Designation', 'SQFT',
        'Unit/Lease Status', 'Name', 'Phone Number', 'Email', 'Move-In', 
        'Notice For Date', 'Move-Out', 'Lease Start', 'Lease End', 'Market + Addl.',
        'Dep On Hand', 'Balance', 'Total Charges', 'RENT', 'PEST', 'VALET TRASH',
        'TRASH', 'PETRENT', 'LOP15', 'EMPLCRED', 'OHIOCHOICE', 'WAIVED ADMIN FEE',
        'WAIVED APP FEE', 'SETUPFEE'
    ]
    
    # Pipeline execution
    file_path = input("Please enter the path to your rent roll Excel file: ")
    excel_file = load_excel_data(file_path)
    raw_data = parse_excel_data(excel_file)
    header_index = find_header_row(raw_data, headers)
    
    if header_index is not None:
        metadata = extract_metadata(raw_data, header_index)
        rent_roll_data = process_rent_roll_data(raw_data, header_index)
        
        print("Metadata:")
        print(metadata.values())
        
        return rent_roll_data
    else:
        raise ValueError("Could not find any of the expected headers in data")

# Execute pipeline
rent_roll_data = main()
rent_roll_data
