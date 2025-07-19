import cv2
import pytesseract
import pandas as pd
import re
import numpy as np
from typing import Dict, List
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpecificFormParser:
    """
    Specialized parser for your specific form documents
    """
    
    def __init__(self, config_path='fields_config.json'):
        # Load column names from config file
        with open(config_path, 'r') as f:
            self.columns = json.load(f)
        
        # Regex patterns for specific data types
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'date': r'\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b',
            'phone': r'\b\d{10,15}\b',
            'imei': r'\b\d{15}\b',  # IMEI is typically 15 digits
            'postcode': r'\b[A-Z0-9\s]{4,10}\b',
            'uk_postcode': r'\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b'
        }
    
    def remove_watermark_effects(self, image):
        """
        Advanced watermark removal and image enhancement
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply multiple denoising techniques
        # 1. Non-local means denoising
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # 2. Morphological opening to remove small artifacts
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)
        
        # 3. Apply adaptive threshold instead of global threshold
        adaptive_thresh = cv2.adaptiveThreshold(
            opened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # 4. Final cleanup with closing operation
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        final = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel2)
        
        return final
    
    def extract_raw_text(self, image_path: str) -> str:
        """
        Extract raw text from image with optimized OCR settings
        """
        # Read and preprocess image
        img = cv2.imread(image_path)
        processed = self.remove_watermark_effects(img)
        
        # OCR configuration optimized for forms
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz@.-_/\\(\\)\\s'
        
        # Extract text
        raw_text = pytesseract.image_to_string(processed, config=custom_config)
        
        return raw_text
    
    def parse_document_structure(self, raw_text: str) -> Dict[str, str]:
        """
        Parse the document based on your specific format
        Looking at your sample, it appears to be structured with multiple entries per line
        """
        lines = raw_text.split('\n')
        parsed_data = {col: '' for col in self.columns}
        
        # Clean lines
        clean_lines = [line.strip() for line in lines if line.strip()]
        
        # Join all text for pattern matching
        full_text = ' '.join(clean_lines)
        
        # Extract email
        email_matches = re.findall(self.patterns['email'], full_text)
        if email_matches:
            parsed_data['Email'] = email_matches[0]
        
        # Extract dates
        date_matches = re.findall(self.patterns['date'], full_text)
        if len(date_matches) >= 1:
            # Try to assign dates based on context
            for i, date in enumerate(date_matches[:3]):  # Limit to first 3 dates
                if '1976' in date or '1983' in date or '1958' in date or '1952' in date:
                    parsed_data['DOB'] = date
                elif '2005' in date or '2003' in date or '2004' in date or '2002' in date:
                    if not parsed_data['Date of issue']:
                        parsed_data['Date of issue'] = date
                    else:
                        parsed_data['Date of renewal'] = date
        
        # Extract phone numbers / IMEI
        phone_matches = re.findall(self.patterns['phone'], full_text)
        imei_matches = re.findall(self.patterns['imei'], full_text)
        
        if imei_matches:
            if len(imei_matches) >= 1:
                parsed_data['IMMEI-1'] = imei_matches[0]
            if len(imei_matches) >= 2:
                parsed_data['IMMEI-2'] = imei_matches[1]
        
        # Extract UK postcodes
        uk_postcode_matches = re.findall(self.patterns['uk_postcode'], full_text, re.IGNORECASE)
        if uk_postcode_matches:
            parsed_data['Mailing postal code'] = uk_postcode_matches[0]
        
        # Parse names and titles from the beginning of lines
        for line in clean_lines[:10]:  # Focus on first few lines where names typically appear
            words = line.split()
            if len(words) >= 2:
                if words[0] in ['Ms.', 'Mr.', 'Mrs.', 'Dr.']:
                    parsed_data['Title'] = words[0].rstrip('.')
                    if len(words) > 1:
                        # Look for name patterns
                        name_part = ' '.join(words[1:4])  # Take next few words as potential name
                        name_words = name_part.split()
                        if len(name_words) >= 1:
                            parsed_data['First Name'] = name_words[0]
                        if len(name_words) >= 2:
                            parsed_data['Last Name'] = name_words[1]
        
        # Extract gender
        if 'Male' in full_text:
            parsed_data['Gender'] = 'Male'
        elif 'Female' in full_text:
            parsed_data['Gender'] = 'Female'
        
        # Extract cities (look for common UK cities)
        uk_cities = ['London', 'Manchester', 'Birmingham', 'Leeds', 'Glasgow', 'Sheffield', 'Liverpool', 'Newcastle', 'Chester', 'Gateshead']
        for city in uk_cities:
            if city in full_text:
                parsed_data['Mailing city'] = city
                break
        
        # Extract service providers
        providers = ['Hutchison', 'Vodafone', 'O2', 'Three', 'EE']
        for provider in providers:
            if provider in full_text:
                parsed_data['Service provider'] = provider
                break
        
        return parsed_data
    
    def extract_multiple_entries(self, raw_text: str) -> list:
        """
        Extract multiple sets of form data from OCR output.
        Returns a list of dicts, each mapping field names to values.
        """
        # Heuristic: split entries by lines starting with a title or email
        entry_starts = []
        lines = raw_text.split('\n')
        for idx, line in enumerate(lines):
            if re.match(r'^(Ms\.|Mr\.|MR|Mrs\.|Dr\.|[A-Za-z]+\s+[A-Za-z]+\s+@)', line.strip()):
                entry_starts.append(idx)
            elif re.match(self.patterns['email'], line.strip()):
                entry_starts.append(idx)
        # Ensure unique and sorted
        entry_starts = sorted(set(entry_starts))
        if not entry_starts:
            # fallback: treat whole text as one entry
            entry_starts = [0]
        entry_starts.append(len(lines))  # sentinel for last entry

        entries = []
        for i in range(len(entry_starts) - 1):
            entry_lines = lines[entry_starts[i]:entry_starts[i+1]]
            entry_text = ' '.join([l.strip() for l in entry_lines if l.strip()])
            # Simple split: try to split by whitespace for each field
            values = re.split(r'\s{2,}|\t|\s\|\s', entry_text)
            # Fallback: split by single space if not enough fields
            if len(values) < len(self.columns):
                values = entry_text.split()
            # Pad or trim to match columns
            values = (values + [''] * len(self.columns))[:len(self.columns)]
            entry = {col: val for col, val in zip(self.columns, values)}
            entries.append(entry)
        return entries

    def process_image_to_csv(self, image_path: str, output_csv: str):
        """
        Process a single image with potentially multiple entries and write to CSV.
        """
        logger.info(f"Processing image: {image_path}")
        raw_text = self.extract_raw_text(image_path)
        entries = self.extract_multiple_entries(raw_text)
        df = pd.DataFrame(entries)
        df.to_csv(output_csv, index=False)
        logger.info(f"Extracted {len(entries)} entries and saved to {output_csv}")
    
    def process_single_document(self, image_path: str) -> Dict[str, str]:
        """
        Process a single document
        """
        logger.info(f"Processing document: {image_path}")
        
        try:
            # Extract text
            raw_text = self.extract_raw_text(image_path)
            logger.info("Raw text extracted successfully")
            
            # Parse structured data
            parsed_data = self.parse_document_structure(raw_text)
            
            # Add source file information
            parsed_data['Source_File'] = image_path
            
            return parsed_data
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            return {col: '' for col in self.columns + ['Source_File']}
    
    def batch_process(self, image_paths: List[str], output_csv: str) -> pd.DataFrame:
        """
        Process multiple documents and save to CSV
        """
        all_data = []
        
        for image_path in image_paths:
            data = self.process_single_document(image_path)
            all_data.append(data)
        
        # Create DataFrame
        df = pd.DataFrame(all_data)
        
        # Ensure all columns are present
        for col in self.columns:
            if col not in df.columns:
                df[col] = ''
        
        # Reorder columns
        column_order = self.columns + ['Source_File']
        df = df[column_order]
        
        # Save to CSV
        df.to_csv(output_csv, index=False)
        logger.info(f"Data saved to {output_csv}")
        
        return df
    
    def preview_extraction(self, image_path: str):
        """
        Preview extraction results for debugging
        """
        raw_text = self.extract_raw_text(image_path)
        parsed_data = self.parse_document_structure(raw_text)
        
        print("=" * 50)
        print("RAW OCR OUTPUT:")
        print("=" * 50)
        print(raw_text)
        print("\n" + "=" * 50)
        print("PARSED DATA:")
        print("=" * 50)
        for key, value in parsed_data.items():
            if value:  # Only show non-empty fields
                print(f"{key}: {value}")