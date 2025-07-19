import argparse
from extract_info import SpecificFormParser
import os


def main():
    parser = argparse.ArgumentParser(description="Extract info from form images using SpecificFormParser.")
    parser.add_argument('--image', type=str, default="../data/demo.jpeg", help="Path to the image file to process (default: ../data/demo.jpeg)")
    parser.add_argument('--batch', nargs='+', help="List of image files to process in batch mode.")
    parser.add_argument('--output', type=str, default="extracted_form_data.csv", help="Output CSV file for batch mode.")
    parser.add_argument('--preview', action='store_true', help="Preview extraction for a single image (default mode).")
    args = parser.parse_args()

    form_parser = SpecificFormParser()

    if args.batch:
        print(f"Batch processing {len(args.batch)} images...")
        df = form_parser.batch_process(args.batch, args.output)
        print(f"Processed {len(df)} documents. Output saved to {args.output}")
    else:
        image_path = args.image
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            return
        print(f"Previewing extraction for: {image_path}")
        form_parser.preview_extraction(image_path)
        print("\nAvailable columns:")
        for i, col in enumerate(form_parser.columns, 1):
            print(f"{i:2d}. {col}")

if __name__ == "__main__":
    main() 