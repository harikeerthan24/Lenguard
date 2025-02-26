import os
import pandas as pd
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from jinja2 import Template

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class RentRollPipeline:
    def __init__(self):
        self.html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                table { border-collapse: collapse; width: 100%; margin-top: 20px; }
                th, td { border: 1px solid black; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .metadata { margin-bottom: 20px; }
            </style>
        </head>
        <body>
            <div class="metadata">{{ metadata_html }}</div>
            {{ table_html }}
        </body>
        </html>
        """

    def read_file(self, file_path):
        """Read CSV or XLSX file into a pandas DataFrame with precise parsing"""
        file_ext = Path(file_path).suffix.lower()
        try:
            if file_ext == '.csv':
                df = pd.read_csv(file_path)
            elif file_ext in ['.xlsx', '.xls']:
                # First, read the entire file to capture metadata
                metadata_df = pd.read_excel(file_path, header=None)
                metadata = metadata_df.iloc[:4, 0].tolist()  # Capture first 4 rows of first column
                
                # Find the actual header row
                header_row = metadata_df[metadata_df.iloc[:, 0].str.contains('Resh ID|Unit', na=False)].index[0]
                
                # Read the data with proper headers
                df = pd.read_excel(file_path, header=header_row)
                df.metadata = metadata  # Store metadata for later use
            
            # Clean and standardize column names
            df.columns = [str(col).strip() for col in df.columns]
            
            # Remove rows that are completely empty or contain only NaN
            df = df.dropna(how='all')
            
            # Identify and convert monetary columns (any column that contains mostly numeric values with $ or ,)
            for col in df.columns:
                if df[col].dtype == 'object':  # Only process string columns
                    # Check if column contains monetary values
                    sample = df[col].dropna().astype(str).str.replace('$', '').str.replace(',', '')
                    if sample.str.match(r'^-?\d*\.?\d*$').mean() > 0.5:  # If more than 50% are numeric
                        df[col] = pd.to_numeric(sample, errors='coerce')
            
            # Convert date columns (columns with date-like values)
            for col in df.columns:
                if df[col].dtype == 'object':  # Only process string columns
                    try:
                        # Try to convert to datetime, if majority succeeds, keep the conversion
                        temp = pd.to_datetime(df[col], errors='coerce')
                        if temp.notna().mean() > 0.5:  # If more than 50% converted successfully
                            df[col] = temp
                    except:
                        continue
            
            return df
        except Exception as e:
            raise Exception(f"Error reading file: {str(e)}")

    def convert_to_html(self, df):
        """Convert DataFrame to HTML with metadata and proper formatting"""
        # Format the metadata if it exists
        metadata_html = ""
        if hasattr(df, 'metadata'):
            metadata_html = "<br>".join(df.metadata)
        
        # Set display options for better visualization
        pd.set_option('display.max_columns', None)  # Show all columns
        pd.set_option('display.max_rows', None)     # Show all rows
        pd.set_option('display.width', None)        # Auto-detect display width
        pd.set_option('display.max_colwidth', None) # Show full content of each cell
        pd.set_option('display.float_format', lambda x: '${:,.2f}'.format(x) if isinstance(x, (float, int)) else str(x))
        
        # Convert DataFrame to HTML with proper formatting
        table_html = df.to_html(classes='data-table', index=False, na_rep='')
        
        template = Template(self.html_template)
        return template.render(metadata_html=metadata_html, table_html=table_html)

    def analyze_with_ai(self, html_content):
        """Parse the table data with expert rent roll knowledge"""
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": """You are an expert rent roll analyst with decades of experience in commercial and residential property management. Your expertise includes handling complex rent rolls from various property management systems (Yardi, MRI, RealPage, AppFolio, etc.). Your task is to parse and structure EVERY SINGLE DATA POINT from rent roll tables with extreme precision.

Key Expertise Areas:
1. Rent Roll Structure Understanding:
   - Property identification and metadata 
   - Unit-level financial data
   - Tenant information
   - Lease terms and conditions
   - Additional charges and credits
   - Deposit tracking
   - Vacancy status
   - Ability to adapt to new column types and formats

2. Column Categories & Special Handling:
   a) Identification Columns:
      - Resh ID/Lease ID (unique identifiers)
      - Unit numbers (may include building/floor indicators) 
      - Floor plan codes and unit types
      - Custom property-specific identifiers
   
   b) Financial Columns:
      - Market rent vs actual rent
      - Security deposits and other deposits
      - Additional charges (RUBS, pet rent, parking)
      - Credits and concessions
      - Payment status and balances
      - Utility reimbursements
      - Property-specific financial metrics
   
   c) Tenant Information:
      - Primary and co-tenant names
      - Contact details (phone, email)
      - Lease guarantor information
      - Employment verification status
      - Custom tenant attributes
   
   d) Lease Terms:
      - Move-in/move-out dates
      - Lease start/end dates
      - Notice dates
      - Renewal status
      - Lease term length
      - Property-specific lease conditions
   
   e) Unit Status Indicators:
      - Vacant/Occupied status
      - Notice status (NTV/NTM)
      - Ready/Not Ready
      - Down/Offline units
      - Model/Office use
      - Custom status codes

3. Edge Cases to Handle:
   - Split payment arrangements
   - Multiple tenants per unit
   - Partial month charges
   - Prorated rents
   - Pending move-ins
   - Transfer units
   - Sublease arrangements
   - Corporate housing
   - Employee units
   - Affordable housing restrictions
   - Rent controlled units
   - Short-term rentals
   - Month-to-month leases
   - Concessions and specialized pricing
   - Non-standard lease terms
   - Property-specific arrangements

4. Data Validation Rules:
   - Verify unit numbers follow property's numbering system
   - Check for valid date ranges
   - Validate rent amounts against market rates
   - Cross-reference deposit amounts with lease terms
   - Verify occupancy status matches lease dates
   - Check for missing required fields
   - Validate contact information formats
   - Verify charge calculations
   - Learn and adapt to new validation rules

5. Dynamic Column Learning:
   - Identify unknown column types
   - Analyze column patterns and relationships
   - Infer column meanings from context
   - Map columns to known categories
   - Handle property-specific nomenclature
   - Adapt analysis to new data structures

Output Format Requirements:

### Data Quality Notes
[List any:
- Unusual patterns
- Inconsistent formats
- Missing data patterns
- Non-standard values
- Format variations]
```

CRITICAL REQUIREMENTS:
- Parse EVERY SINGLE CELL, no exceptions
- Maintain EXACT original values
- Preserve ALL special characters
- Keep ALL formatting intact
- Flag ANY unusual patterns
- Document ALL edge cases
- Note ANY data inconsistencies
- Maintain PRECISE monetary values
- Keep EXACT date formats
- Preserve UNIT-SPECIFIC details

NO summarization, NO aggregation, NO data transformation - just pure, precise parsing with expert rent roll context."""},
                    {"role": "user", "content": f"As a rent roll expert, parse this table with attention to every detail and edge case: {html_content}"}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error in parsing: {str(e)}")

    def process_file(self, input_file):
        """Process the file with focus on complete data parsing and display"""
        print(f"Processing file: {input_file}")
        
        # Read and parse the file
        df = self.read_file(input_file)
        
        # Convert to HTML with all data preserved
        html_content = self.convert_to_html(df)
        
        # Create output directory
        os.makedirs('html_outputs', exist_ok=True)
        
        # Save outputs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_output = f"html_outputs/processed_rent_roll_{timestamp}.html"
        excel_output = f"html_outputs/processed_rent_roll_{timestamp}.xlsx"
        
        # Save HTML with complete data
        with open(html_output, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Save Excel with all data preserved
        df.to_excel(excel_output, index=False)
        
        # Parse with AI
        parsed_data = self.analyze_with_ai(html_content)
        
        # Display the complete DataFrame
        print("\nComplete Rent Roll Data:")
        print("=" * 100)
        if hasattr(df, 'metadata'):
            print("\nMetadata:")
            for meta in df.metadata:
                print(meta)
            print("\n" + "=" * 100)
        
        # Display DataFrame in chunks for better readability
        chunk_size = 5  # Number of columns to display at once
        for i in range(0, len(df.columns), chunk_size):
            chunk_cols = df.columns[i:i + chunk_size]
            print(f"\nColumns {i+1} to {min(i+chunk_size, len(df.columns))}:")
            print("-" * 100)
            print(df[chunk_cols].to_string(index=False))
            print("\n")
        
        return {
            'parsed_data': df,
            'html_output': html_output,
            'excel_output': excel_output,
            'complete_parse': parsed_data
        }

def main():
    # Example usage
    pipeline = RentRollPipeline()
    
    # Get input file from user
    input_file = input("Enter the path to your rent roll file (CSV or XLSX): ")
    
    try:
        # Set pandas display options for better output
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.precision', 2)
        
        results = pipeline.process_file(input_file)
        
        print("\nProcessing complete!")
        print(f"HTML output saved to: {results['html_output']}")
        print(f"Excel output saved to: {results['excel_output']}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 
    
    
# parser 

# preprocessing 



