import pandas as pd
import numpy as np
from datetime import datetime
from rapidfuzz import fuzz, process
from typing import Dict, List, Optional, Tuple, Any
import re
import logging
from pathlib import Path
import warnings
from dataclasses import dataclass
from collections import defaultdict
import os

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ColumnConfig:
    """Configuration for column processing."""
    standard_name: str
    variants: List[str]
    data_type: str  # 'numeric', 'date', 'boolean', 'text'
    required: bool
    validation_rules: List[str]
    default_value: Any = None

class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    def __init__(self, message: str, details: Dict[str, Any]):
        self.message = message
        self.details = details
        super().__init__(self.message)

class RentRollProcessor:
    """Advanced processor for rent roll data with enhanced error handling and validation."""
    
    def __init__(self, template_path: Optional[str] = None):
        """
        Initialize the RentRollProcessor.
        
        Args:
            template_path: Optional path to a template configuration file
        """
        self.df: Optional[pd.DataFrame] = None
        self.validation_errors: Dict[str, List[str]] = defaultdict(list)
        self.data_quality_metrics: Dict[str, Any] = {}
        self._initialize_column_configs()
        
        # Load custom template if provided
        if template_path:
            self._load_template_config(template_path)

    def _initialize_column_configs(self):
        """Initialize standard column configurations with validation rules."""
        self.column_configs = {
            'unit': ColumnConfig(
                standard_name='unit',
                variants=['unit #', 'unit', 'unit number', 'apt', 'apartment', 'apt #', 'apt number'],
                data_type='text',
                required=True,
                validation_rules=['not_empty', 'unique']
            ),
            'unit_type': ColumnConfig(
                standard_name='unit_type',
                variants=['unit type', 'type', 'style', 'layout', 'floor plan', 'model'],
                data_type='text',
                required=True,
                validation_rules=['not_empty']
            ),
            'sq_ft': ColumnConfig(
                standard_name='sq_ft',
                variants=['sq ft', 'sqft', 'square feet', 'size', 'area', 'square footage'],
                data_type='numeric',
                required=True,
                validation_rules=['positive', 'range:100-10000']
            ),
            'occupancy_status': ColumnConfig(
                standard_name='occupancy_status',
                variants=['occup stat', 'status', 'occupancy', 'occupied', 'vacancy status'],
                data_type='text',
                required=True,
                validation_rules=['not_empty']
            ),
            'tenant_name': ColumnConfig(
                standard_name='tenant_name',
                variants=['tenant name', 'resident', 'occupant', 'lessee', 'resident name'],
                data_type='text',
                required=True,
                validation_rules=['not_empty']
            ),
            'move_in_date': ColumnConfig(
                standard_name='move_in_date',
                variants=['move-in date', 'move in', 'movein', 'start date', 'moved in'],
                data_type='date',
                required=True,
                validation_rules=['not_empty']
            ),
            'lease_start': ColumnConfig(
                standard_name='lease_start',
                variants=['lease start', 'start', 'commence', 'beginning', 'lease begin'],
                data_type='date',
                required=True,
                validation_rules=['not_empty']
            ),
            'lease_end': ColumnConfig(
                standard_name='lease_end',
                variants=['lease end', 'end', 'expiration', 'termination', 'lease expiration'],
                data_type='date',
                required=True,
                validation_rules=['not_empty']
            ),
            'move_out_date': ColumnConfig(
                standard_name='move_out_date',
                variants=['move out date', 'move out', 'moveout', 'vacancy date', 'moved out'],
                data_type='date',
                required=False,
                validation_rules=[]
            ),
            'lease_term': ColumnConfig(
                standard_name='lease_term',
                variants=['lease term', 'term', 'duration', 'period', 'months'],
                data_type='numeric',
                required=True,
                validation_rules=['positive']
            ),
            'contract_rent': ColumnConfig(
                standard_name='contract_rent',
                variants=['contract rent', 'rent', 'monthly rent', 'base rent', 'current rent'],
                data_type='numeric',
                required=True,
                validation_rules=['positive']
            ),
            'premium_included': ColumnConfig(
                standard_name='premium_included',
                variants=['premium included in rent', 'premium', 'additional charges', 'premium amount'],
                data_type='numeric',
                required=False,
                validation_rules=['positive']
            ),
            'utilities_included': ColumnConfig(
                standard_name='utilities_included',
                variants=['are utilities included', 'utilities', 'included utilities', 'utility inclusion'],
                data_type='boolean',
                required=False,
                validation_rules=[]
            ),
            'non_revenue_type': ColumnConfig(
                standard_name='non_revenue_type',
                variants=['non revenue type', 'non-revenue', 'revenue type', 'non rev type'],
                data_type='text',
                required=False,
                validation_rules=[]
            ),
            'non_revenue_amount': ColumnConfig(
                standard_name='non_revenue_amount',
                variants=['non-revenue amount', 'revenue loss', 'loss amount', 'non rev amt'],
                data_type='numeric',
                required=False,
                validation_rules=['positive']
            ),
            'concessions_amount': ColumnConfig(
                standard_name='concessions_amount',
                variants=['concessions amt', 'concession', 'discount', 'concession amount'],
                data_type='numeric',
                required=False,
                validation_rules=['positive']
            ),
            'concessions_term': ColumnConfig(
                standard_name='concessions_term',
                variants=['concessions term', 'discount period', 'concession duration'],
                data_type='text',
                required=False,
                validation_rules=[]
            ),
            'inspected': ColumnConfig(
                standard_name='inspected',
                variants=['inspected?', 'inspection', 'checked', 'unit inspected'],
                data_type='boolean',
                required=False,
                validation_rules=[]
            ),
            'subsidy_rent_type': ColumnConfig(
                standard_name='subsidy_rent_type',
                variants=['subsidy rent type', 'subsidy type', 'assistance type', 'subsidy program'],
                data_type='text',
                required=False,
                validation_rules=[]
            ),
            'subsidy_rent_amount': ColumnConfig(
                standard_name='subsidy_rent_amount',
                variants=['subsidy rent amount', 'subsidy amount', 'assistance amount', 'subsidy'],
                data_type='numeric',
                required=False,
                validation_rules=['positive']
            ),
            'comments': ColumnConfig(
                standard_name='comments',
                variants=['comments', 'notes', 'remarks', 'description', 'additional info'],
                data_type='text',
                required=False,
                validation_rules=[]
            ),
            'asking_rent': ColumnConfig(
                standard_name='asking_rent',
                variants=['asking rent', 'list price', 'advertised rent', 'market rent'],
                data_type='numeric',
                required=False,
                validation_rules=['positive']
            ),
            'appraiser_market_rent': ColumnConfig(
                standard_name='appraiser_market_rent',
                variants=['appraiser market rent', 'market value', 'appraised rent'],
                data_type='numeric',
                required=False,
                validation_rules=['positive']
            ),
            'underwriting_rent': ColumnConfig(
                standard_name='underwriting_rent',
                variants=['underwriting rent', 'projected rent', 'model rent'],
                data_type='numeric',
                required=False,
                validation_rules=['positive']
            ),
            'ami_restrictions': ColumnConfig(
                standard_name='ami_restrictions',
                variants=['ami restrictions', 'ami limits', 'income restrictions'],
                data_type='text',
                required=False,
                validation_rules=[]
            ),
            'max_restricted_rent': ColumnConfig(
                standard_name='max_restricted_rent',
                variants=['max restricted rent', 'max allowed rent', 'rent ceiling'],
                data_type='numeric',
                required=False,
                validation_rules=['positive']
            ),
            'max_foreclose_rent': ColumnConfig(
                standard_name='max_foreclose_rent',
                variants=['max foreclose rent', 'foreclosure rent', 'distressed rent'],
                data_type='numeric',
                required=False,
                validation_rules=['positive']
            ),
            'hap_overhang': ColumnConfig(
                standard_name='hap_overhang',
                variants=['hap overhang', 'hap obligation', 'housing assistance'],
                data_type='numeric',
                required=False,
                validation_rules=['positive']
            )
        }

    def _load_template_config(self, template_path: str):
        """Load custom template configuration."""
        try:
            with open(template_path, 'r') as f:
                template_config = pd.read_json(f)
                # Merge with existing configs
                for col, config in template_config.items():
                    if col in self.column_configs:
                        self.column_configs[col].variants.extend(config.get('variants', []))
        except Exception as e:
            logger.warning(f"Failed to load template config: {str(e)}")

    def _preprocess_excel(self, file_path: str, sheet_name: Optional[str] = None) -> Tuple[pd.DataFrame, int]:
        """
        Preprocess Excel file to handle various formats and find the actual data table.
        
        Returns:
            Tuple of (DataFrame, header_row_index)
        """
        excel_file = pd.ExcelFile(file_path)
        available_sheets = excel_file.sheet_names
        logger.info(f"Available sheets: {available_sheets}")

        if sheet_name is None:
            # Try to identify the most likely sheet containing rent roll data
            sheet_scores = {}
            for sheet in available_sheets:
                score = sum(1 for keyword in ['rent', 'roll', 'unit', 'tenant']
                          if keyword.lower() in sheet.lower())
                sheet_scores[sheet] = score
            
            sheet_name = max(sheet_scores.items(), key=lambda x: x[1])[0]
            logger.info(f"Selected sheet: {sheet_name} (score: {sheet_scores[sheet_name]})")

        # Read the entire sheet
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        
        # Find the most likely header row
        header_candidates = {}
        for idx in range(min(15, len(df))):
            row = df.iloc[idx]
            score = self._evaluate_header_row(row)
            header_candidates[idx] = score
            
        header_row = max(header_candidates.items(), key=lambda x: x[1])[0]
        
        # Read the file again with the correct header
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row)
        
        return df, header_row

    def _evaluate_header_row(self, row: pd.Series) -> float:
        """
        Evaluate how likely a row is to be the header row.
        Returns a score between 0 and 1.
        """
        score = 0
        non_empty_cells = sum(1 for cell in row if pd.notna(cell))
        
        if non_empty_cells == 0:
            return 0
        
        # Calculate match score against known column variants
        all_variants = [variant.lower() for config in self.column_configs.values()
                       for variant in config.variants]
        
        matches = 0
        for cell in row:
            if pd.isna(cell):
                continue
            cell_str = str(cell).lower().strip()
            
            # Check for exact matches
            if cell_str in all_variants:
                matches += 1
                continue
            
            # Check for fuzzy matches using rapidfuzz
            best_match = process.extractOne(
                cell_str,
                all_variants,
                scorer=fuzz.ratio,
                score_cutoff=80
            )
            if best_match:
                matches += 0.8
        
        return matches / len(self.column_configs)

    def _clean_and_standardize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize the data."""
        # Remove completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Handle merged cells and duplicated headers
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Clean whitespace and special characters
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace(r'^\s*$', np.nan, regex=True)
        
        return df

    def _validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate data against defined rules.
        Returns True if validation passes, False otherwise.
        """
        self.validation_errors.clear()
        
        for col_name, config in self.column_configs.items():
            if col_name not in df.columns:
                if config.required:
                    self.validation_errors['missing_columns'].append(col_name)
                continue
            
            col_data = df[col_name]
            
            # Apply validation rules
            for rule in config.validation_rules:
                if rule == 'not_empty':
                    empty_rows = col_data.isna()
                    if empty_rows.any():
                        self.validation_errors[col_name].append(
                            f"Empty values in rows: {empty_rows[empty_rows].index.tolist()}")
                
                elif rule == 'unique':
                    duplicates = col_data.duplicated()
                    if duplicates.any():
                        self.validation_errors[col_name].append(
                            f"Duplicate values in rows: {duplicates[duplicates].index.tolist()}")
                
                elif rule.startswith('range:'):
                    min_val, max_val = map(float, rule.split(':')[1].split('-'))
                    invalid_range = (col_data < min_val) | (col_data > max_val)
                    if invalid_range.any():
                        self.validation_errors[col_name].append(
                            f"Values out of range ({min_val}-{max_val}) in rows: {invalid_range[invalid_range].index.tolist()}")
        
        return len(self.validation_errors) == 0

    def _calculate_data_quality_metrics(self):
        """Calculate data quality metrics."""
        if self.df is None:
            return
        
        metrics = {
            'total_rows': len(self.df),
            'missing_values_by_column': self.df.isna().sum().to_dict(),
            'completeness_ratio': 1 - (self.df.isna().sum().sum() / (self.df.shape[0] * self.df.shape[1])),
            'duplicate_rows': self.df.duplicated().sum(),
            'columns_found': list(self.df.columns),
            'columns_mapped': list(set(self.df.columns) & set(self.column_configs.keys())),
        }
        
        self.data_quality_metrics = metrics
        logger.info("Data quality metrics calculated")
        return metrics

    def print_validation_errors(self):
        """Print validation errors in a formatted way."""
        if not self.validation_errors:
            logger.info("No validation errors found.")
            return

        logger.warning("\n=== Validation Errors ===")
        for col, errors in self.validation_errors.items():
            logger.warning(f"\nColumn: {col}")
            for error in errors:
                logger.warning(f"  - {error}")

    def print_data_quality_metrics(self):
        """Print data quality metrics in a formatted way."""
        if not self.data_quality_metrics:
            logger.info("No data quality metrics available.")
            return

        logger.info("\n=== Data Quality Metrics ===")
        
        # Print basic metrics
        logger.info(f"\nBasic Metrics:")
        logger.info(f"  - Total rows processed: {self.data_quality_metrics['total_rows']}")
        logger.info(f"  - Duplicate rows: {self.data_quality_metrics['duplicate_rows']}")
        logger.info(f"  - Data completeness: {self.data_quality_metrics['completeness_ratio']:.2%}")
        
        # Print column statistics
        logger.info(f"\nColumn Statistics:")
        logger.info(f"  - Total columns found: {len(self.data_quality_metrics['columns_found'])}")
        logger.info(f"  - Successfully mapped columns: {len(self.data_quality_metrics['columns_mapped'])}")
        
        # Print missing values by column
        logger.info(f"\nMissing Values by Column:")
        missing_values = self.data_quality_metrics['missing_values_by_column']
        for col, count in missing_values.items():
            if count > 0:
                percentage = (count / self.data_quality_metrics['total_rows']) * 100
                logger.info(f"  - {col}: {count} missing values ({percentage:.1f}%)")

        # Print unmapped columns
        unmapped_columns = set(self.data_quality_metrics['columns_found']) - set(self.data_quality_metrics['columns_mapped'])
        if unmapped_columns:
            logger.warning(f"\nUnmapped Columns:")
            for col in unmapped_columns:
                logger.warning(f"  - {col}")

    def _extract_metadata(self, df: pd.DataFrame, max_rows: int = 5) -> Dict[str, str]:
        """Extract metadata from the top rows of the rent roll."""
        metadata = {}
        
        try:
            # Look for metadata in the first few rows
            for idx in range(min(max_rows, len(df))):
                row = df.iloc[idx]
                row_str = ' '.join(str(x) for x in row if pd.notna(x)).lower()
                
                # Extract property name
                if 'apartments' in row_str or 'manor' in row_str:
                    metadata['property_name'] = ' '.join(str(x) for x in row if pd.notna(x))
                
                # Extract date information
                if 'as of' in row_str:
                    date_parts = row_str.split('as of')
                    if len(date_parts) > 1:
                        metadata['as_of_date'] = date_parts[1].strip()
                
                # Extract month/year
                if 'month' in row_str and 'year' in row_str:
                    try:
                        metadata['report_period'] = row_str.split('=')[-1].strip()
                    except:
                        # If split fails, use the whole string
                        metadata['report_period'] = row_str.strip()
        except Exception as e:
            logger.warning(f"Error extracting metadata: {str(e)}")
            
        return metadata

    def _handle_multi_row_headers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Handle multi-row headers in rent roll.
        Returns cleaned DataFrame and header mapping.
        """
        # Find the main header row (usually contains Unit, Type, etc.)
        header_row = None
        header_mapping = {}
        
        # Skip metadata rows (usually first 3-4 rows)
        start_idx = 3
        
        for idx in range(start_idx, min(20, len(df))):
            row = df.iloc[idx]
            row_str = ' '.join(str(x) for x in row if pd.notna(x)).lower()
            
            # Look for key header indicators with more specific patterns
            if ('unit' in row_str and any(x in row_str for x in ['type', 'resident', 'rent'])):
                header_candidates = idx
                
                # Check next row for sub-headers
                if idx + 1 < len(df):
                    next_row = df.iloc[idx + 1]
                    
                    # Combine headers if needed
                    headers = []
                    for col1, col2 in zip(row, next_row):
                        col1_str = str(col1) if pd.notna(col1) else ''
                        col2_str = str(col2) if pd.notna(col2) else ''
                        
                        if col1_str and col2_str:
                            # Both rows have values, combine them
                            headers.append(f"{col1_str} {col2_str}".strip())
                        elif col1_str:
                            headers.append(col1_str.strip())
                        elif col2_str:
                            headers.append(col2_str.strip())
                        else:
                            headers.append('')
                    
                    # Create mapping, filtering out empty headers
                    header_mapping = {i: h for i, h in enumerate(headers) if h and not h.startswith('Unnamed')}
                    header_row = idx
                    break
        
        if header_row is not None:
            # Skip the header rows and rename columns
            df = df.iloc[header_row + 2:].reset_index(drop=True)
            df.columns = range(len(df.columns))
            df = df.rename(columns=header_mapping)
            
            # Remove any remaining unnamed columns
            unnamed_cols = [col for col in df.columns if isinstance(col, int) or str(col).startswith('Unnamed')]
            df = df.drop(columns=unnamed_cols)
            
            # Clean up the data - remove any rows that are actually headers or metadata
            df = df[~df.iloc[:, 0].astype(str).str.contains('current|notice|vacant|total', case=False)]
        else:
            logger.warning("Could not find proper header row, using first row as header")
            df = df.iloc[0:].reset_index(drop=True)
        
        return df, header_mapping

    def _handle_sections(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle different sections in the rent roll (Current/Notice/Vacant)."""
        # Identify section rows
        section_rows = []
        current_section = None
        
        for idx, row in df.iterrows():
            row_str = ' '.join(str(x) for x in row if pd.notna(x)).lower()
            
            # Look for section headers
            if any(section in row_str for section in ['current', 'notice', 'vacant']):
                section_rows.append(idx)
                if 'current' in row_str:
                    current_section = 'Current'
                elif 'notice' in row_str:
                    current_section = 'Notice'
                elif 'vacant' in row_str:
                    current_section = 'Vacant'
            
            # Add section information if we're tracking sections
            if current_section and not row_str.startswith(('current', 'notice', 'vacant')):
                df.loc[idx, 'Section'] = current_section
        
        # Remove section header rows
        df = df.drop(section_rows)
        
        return df

    def process_file(self, file_path: str, sheet_name: Optional[str] = None,
                    validate: bool = True) -> pd.DataFrame:
        """
        Process a rent roll file with advanced error handling and validation.
        """
        logger.info(f"Processing file: {file_path}")
        
        try:
            # First, get the Excel file object to check sheets
            excel_file = pd.ExcelFile(file_path)
            available_sheets = excel_file.sheet_names
            logger.info(f"Available sheets: {available_sheets}")

            # If no sheet specified, try to find the most relevant one
            if sheet_name is None:
                sheet_scores = {}
                for sheet in available_sheets:
                    score = sum(1 for keyword in ['rent', 'roll', 'unit', 'tenant']
                              if keyword.lower() in sheet.lower())
                    sheet_scores[sheet] = score
                
                sheet_name = available_sheets[0]  # Default to first sheet
                if sheet_scores:
                    sheet_name = max(sheet_scores.items(), key=lambda x: x[1])[0]
                logger.info(f"Selected sheet: {sheet_name}")

            # Read the selected sheet
            raw_df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
            
            # Extract metadata
            metadata = self._extract_metadata(raw_df)
            logger.info("\nExtracted Metadata:")
            for key, value in metadata.items():
                logger.info(f"  - {key}: {value}")
            
            # Handle multi-row headers
            self.df, header_mapping = self._handle_multi_row_headers(raw_df)
            
            # Handle sections
            self.df = self._handle_sections(self.df)
            
            # Clean and standardize data
            self.df = self._clean_and_standardize_data(self.df)
            
            # Map columns
            mapped_cols = self._fuzzy_match_columns(self.df.columns)
            logger.info("\nColumn Mapping Results:")
            for orig, mapped in mapped_cols.items():
                logger.info(f"  - {orig} -> {mapped}")
            
            self.df = self.df.rename(columns=mapped_cols)
            
            # Add metadata as additional columns
            if metadata:
                for key, value in metadata.items():
                    self.df[f'metadata_{key}'] = value
            
            # Validate data if requested
            if validate:
                is_valid = self._validate_data(self.df)
                self.print_validation_errors()
            
            # Calculate and print quality metrics
            self._calculate_data_quality_metrics()
            self.print_data_quality_metrics()
            
            return self.df
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            raise

    def export_to_excel(self, output_path: str) -> None:
        """
        Export processed data to Excel with enhanced formatting.
        """
        if self.df is None:
            raise ValueError("No data to export. Please process a file first.")
        
        # Ensure the output directory exists
        output_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(output_dir, exist_ok=True)
        
        # Try to find an available filename if the target file is locked
        base_name, ext = os.path.splitext(output_path)
        counter = 1
        while True:
            try:
                writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
                break
            except PermissionError:
                output_path = f"{base_name}_{counter}{ext}"
                counter += 1
                if counter > 100:  # Prevent infinite loop
                    raise ValueError("Unable to find available filename for export")
        
        try:
            # Write main data
            self.df.to_excel(writer, sheet_name='Processed Rent Roll', index=False)
            self._format_excel_output(writer, 'Processed Rent Roll')
            writer.close()
            logger.info(f"\nData exported to {output_path}")
        except Exception as e:
            logger.error(f"Error exporting to Excel: {str(e)}")
            try:
                # Fallback: Try to save without formatting
                self.df.to_excel(output_path, index=False)
                logger.info(f"\nData exported to {output_path} (without formatting)")
            except Exception as e2:
                logger.error(f"Fallback export also failed: {str(e2)}")
                raise

    def _format_excel_output(self, writer: pd.ExcelWriter, sheet_name: str):
        """Apply advanced Excel formatting."""
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]
        
        # Define formats
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D3D3D3',
            'border': 1
        })
        
        date_format = workbook.add_format({
            'num_format': 'mm/dd/yyyy',
            'border': 1
        })
        
        currency_format = workbook.add_format({
            'num_format': '$#,##0.00',
            'border': 1
        })
        
        number_format = workbook.add_format({
            'num_format': '#,##0',
            'border': 1
        })
        
        # Apply formats
        for col_num, (col_name, col_data) in enumerate(self.df.items()):
            worksheet.write(0, col_num, col_name, header_format)
            
            # Set column width
            max_length = max(
                col_data.astype(str).str.len().max(),
                len(str(col_name))
            )
            worksheet.set_column(col_num, col_num, min(max_length + 2, 30))
            
            # Apply data type specific formatting
            if col_name in self.column_configs:
                config = self.column_configs[col_name]
                if config.data_type == 'date':
                    worksheet.set_column(col_num, col_num, 12, date_format)
                elif config.data_type == 'numeric':
                    if 'rent' in col_name.lower() or 'amount' in col_name.lower():
                        worksheet.set_column(col_num, col_num, 12, currency_format)
                    else:
                        worksheet.set_column(col_num, col_num, 12, number_format)

    def _fuzzy_match_columns(self, columns: List[str]) -> Dict[str, str]:
        """
        Match input columns to standard columns using fuzzy string matching.
        
        Args:
            columns: List of column names to match
            
        Returns:
            Dictionary mapping original column names to standardized names
        """
        mapped_cols = {}
        unmapped_cols = []
        
        # First pass: Try exact matches
        for col in columns:
            col_lower = str(col).lower().strip()
            matched = False
            
            # Check for exact matches first
            for config_name, config in self.column_configs.items():
                if col_lower == config.standard_name or col_lower in config.variants:
                    mapped_cols[col] = config.standard_name
                    matched = True
                    break
            
            if not matched:
                unmapped_cols.append(col)
        
        # Second pass: Try fuzzy matching for remaining columns
        for col in unmapped_cols:
            col_lower = str(col).lower().strip()
            
            # Create a list of all possible matches (standard names and variants)
            all_possible_matches = []
            for config_name, config in self.column_configs.items():
                if config.standard_name not in mapped_cols.values():  # Skip if already mapped
                    all_possible_matches.extend([(variant, config.standard_name) 
                                              for variant in [config.standard_name] + config.variants])
            
            # Use rapidfuzz to find the best match
            if all_possible_matches:
                choices, standard_names = zip(*all_possible_matches)
                best_match = process.extractOne(
                    col_lower,
                    choices,
                    scorer=fuzz.ratio,
                    score_cutoff=80
                )
                
                if best_match:
                    match_idx = choices.index(best_match[0])
                    standard_name = standard_names[match_idx]
                    mapped_cols[col] = standard_name
                    logger.info(f"Fuzzy matched '{col}' to '{standard_name}' with score {best_match[1]}")
                else:
                    logger.warning(f"Could not match column '{col}' to any standard column")
        
        return mapped_cols

    def _clean_numeric(self, value: str) -> float:
        """Clean numeric values by removing currency symbols and commas."""
        if pd.isna(value):
            return np.nan
        if isinstance(value, (int, float)):
            return float(value)
        try:
            # Remove currency symbols, commas, and spaces
            cleaned = re.sub(r'[$,\s]', '', str(value))
            return float(cleaned)
        except (ValueError, TypeError):
            return np.nan

    def _clean_date(self, value) -> Optional[datetime]:
        """Clean and standardize date values."""
        if pd.isna(value):
            return None
        if isinstance(value, datetime):
            return value
        try:
            return pd.to_datetime(value)
        except (ValueError, TypeError):
            return None

# Example usage
if __name__ == "__main__":
    try:
        # Initialize processor
        processor = RentRollProcessor()
        
        # Process the rent roll file
        file_path = r"E:\Lenguard\Rent Roll_Marlon Manor_3.7.2023.xlsx"
        logger.info(f"Processing file: {file_path}")
        
        df = processor.process_file(file_path, validate=True)
        
        # Export processed data with timestamp to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"processed_rent_roll_{timestamp}.xlsx"
        processor.export_to_excel(output_path)
        
    except Exception as e:
        logger.error(f"Error processing rent roll: {str(e)}", exc_info=True) 