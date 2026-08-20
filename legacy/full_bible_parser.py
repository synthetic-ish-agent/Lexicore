import json
import re
import uuid
import os

def parse_full_bible_to_json(
    input_file: str = "raw_full_bible.txt",
    output_file: str = "lexicore_full_bible.json",
    source_name: str = "Bible-KJV",
    segment_type: str = "Scripture",
    language: str = "English"
):
    """
    Reads a full Bible text file (assuming [Book C:V Text] format) and 
    formats it into the LexiCore JSON structure.
    
    CRITICAL: This regex is designed to be flexible. It looks for a sequence of 
    words/numbers followed by a Chapter:Verse pattern, and then captures the rest as text.
    """
    print(f"--- Starting parsing from {input_file} to {output_file} ---")
    
    if not os.path.exists(input_file):
        print(f"FATAL: Input file not found: {input_file}. Please create it first.")
        print("Hint: Find a full-text Bible dataset online and save it as raw_full_bible.txt.")
        return

    # Regex: (Book Name + Chapter) (e.g., Genesis 1) : (Verse) (e.g., :1) (Rest of the line)
    # This pattern is robust for books like "1 Corinthians 1:1"
    # Group 1: Book Name (e.g., 1 Corinthians)
    # Group 2: Chapter:Verse (e.g., 1:1)
    # Group 3: Verse Text
    citation_pattern = re.compile(r'^(.*?)\s(\d+:\d+)\s*(.*)$', re.DOTALL)
    
    formatted_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            match = citation_pattern.match(raw_line)
            
            if match:
                # Group 1: The citation string up to the Chapter:Verse part (e.g., "Genesis 1")
                book_chapter_part = match.group(1).strip()
                citation_ref = match.group(2).strip() # e.g., "1:1"
                text_segment = match.group(3).strip()
                
                # Extract the Book Name from the book_chapter_part
                # We assume the last number in book_chapter_part is the Chapter number.
                book_name = re.sub(r'\s\d+$', '', book_chapter_part).strip()
                
                # Fallback check
                if not book_name:
                     book_name = "Unknown Book"

                # Use the extracted Book Name as the scripture source
                source = book_name
                
                # Generate a unique ID (Book_Chapter_Verse_UID)
                unique_id = f"{source.upper().replace(' ', '_').replace('.', '')}_{citation_ref.replace(':', '_')}"
                
                new_entry = {
                    "id": unique_id,
                    "scripture_source": source,
                    "original_language": language,
                    "text_segment": text_segment,
                    "citation_ref": citation_ref,
                    "segment_type": segment_type,
                    "concept_tags": []
                }
                formatted_data.append(new_entry)
            # else: print(f"Format Error: {raw_line[:50]}...")

    # 3. Outputting the JSON
    if formatted_data:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, indent=4)
            
        print(f"\nSuccessfully parsed and formatted {len(formatted_data)} segments.")
        print(f"Data saved to {output_file}")
    else:
        print("No valid segments were parsed. Check the format of your input file.")


if __name__ == "__main__":
    parse_full_bible_to_json()