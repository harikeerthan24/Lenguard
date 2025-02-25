import pandas as pd

def load_excel_data(file_path: str) -> pd.ExcelFile:
    """Load Excel file from given path"""
    return pd.ExcelFile(file_path)

def parse_excel_data(excel_file: pd.ExcelFile) -> pd.DataFrame:
    """Parse Excel file into DataFrame"""
    return excel_file.parse()

def find_header_row(data: pd.DataFrame, headers: list) -> int:
    """Find the row containing any of the expected headers"""
    header_row_idx = None
    for i, row in data.iterrows():
        if any(col in headers for col in row.astype(str)):  # Check if any column matches expected headers
            header_row_idx = i
            break  # Stop after finding the first match

    if header_row_idx is None:
        raise ValueError("No matching header row found in the Excel file.")
    
    print(f"Header row found at index: {header_row_idx}")
    return header_row_idx
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
