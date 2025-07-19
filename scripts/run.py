import sys
import os
from extract_info import SpecificFormParser

def print_usage():
    print("Usage: python run.py <image_path> <output_csv> [fields_config.json]")
    print("  <image_path>: Path to the image to process")
    print("  <output_csv>: Path to save the extracted CSV")
    print("  [fields_config.json]: (Optional) Path to field names config file (default: fields_config.json)")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print_usage()
        sys.exit(1)
    image_path = sys.argv[1]
    output_csv = sys.argv[2]
    config_path = sys.argv[3] if len(sys.argv) > 3 else "../fields_config.json"

    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        sys.exit(1)
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    parser = SpecificFormParser(config_path=config_path)
    parser.process_image_to_csv(image_path, output_csv)
    print(f"Done. Extracted data saved to {output_csv}") 