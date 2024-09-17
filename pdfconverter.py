import pdfplumber
import pytesseract
from PIL import Image
import argparse
import os

def pdf_to_text(pdf_path, output_txt):
    text = ''
    with pdfplumber.open(pdf_path) as pdf:
        num_pages = len(pdf.pages)
        print(f"Number of pages: {num_pages}")
        
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                print(f"Text for page {i + 1}:\n{page_text}\n")
                text += page_text
            else:
                # If no text is extracted, use OCR
                print(f"No text found on page {i + 1}, using OCR...")
                for img in page.images:
                    x0, y0, x1, y1 = img["x0"], img["top"], img["x1"], img["bottom"]
                    cropped_image = page.within_bbox((x0, y0, x1, y1)).to_image()
                    ocr_text = pytesseract.image_to_string(cropped_image.original)
                    print(f"OCR Text for page {i + 1}:\n{ocr_text}\n")
                    text += ocr_text
    
    with open(output_txt, 'w', encoding='utf-8') as txt_file:
        txt_file.write(text)

    # Print the size of the output text file
    file_size = os.path.getsize(output_txt)
    print(f"Size of the output text file: {file_size} bytes")

def main():
    parser = argparse.ArgumentParser(description="Convert PDF to text")
    parser.add_argument('pdf_path', type=str, help='Path to the PDF file')
    parser.add_argument('output_txt', type=str, help='Output text file name')
    args = parser.parse_args()

    pdf_to_text(args.pdf_path, args.output_txt)
    print("PDF converted to text successfully!")

if __name__ == "__main__":
    main()