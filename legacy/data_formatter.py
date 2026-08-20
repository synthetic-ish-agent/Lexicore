import json
import uuid
import re

def format_text_to_json(
    input_file: str,
    output_file: str,
    source_name: str,
    segment_type: str,
    language: str = "English"
):
    """
    Reads a raw text file, segments it by paragraph, and formats it 
    into the LexiCore JSON structure.
    """
    print(f"--- Starting formatting for {source_name} ---")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    # 1. Segmentation: Split the text into segments (paragraphs)
    # This uses two or more newline characters to define a new paragraph.
    segments = re.split(r'\n\s*\n', raw_text.strip())
    
    formatted_data = []
    
    # 2. Iteration and Formatting
    for i, segment in enumerate(segments):
        # Clean up the segment (remove extra spaces/newlines within the paragraph)
        clean_segment = ' '.join(segment.split()).strip()

        if not clean_segment:
            continue

        # 3. Generating Metadata
        
        # Create a basic, sequential citation reference (e.g., "Section 1", "Section 2")
        # You will need to manually refine these later if the source has chapters/pages.
        citation_ref = f"Section {i + 1}" 
        
        # Generate a unique ID
        unique_id = f"{source_name.upper().replace(' ', '_')}_{str(uuid.uuid4())[:8]}"

        # 4. Building the JSON Object
        new_entry = {
            "id": unique_id,
            "scripture_source": source_name,
            "original_language": language,
            "text_segment": clean_segment,
            "citation_ref": citation_ref,
            "segment_type": segment_type,
            "concept_tags": []  # Leave empty for manual tagging later
        }
        formatted_data.append(new_entry)
        
    # 5. Outputting the JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, indent=4)
        
    print(f"\nSuccessfully formatted {len(formatted_data)} segments.")
    print(f"Data saved to {output_file}")


if __name__ == "__main__":
    # --- CONFIGURATION ---
    RAW_INPUT_FILE = "calvin_institutes_raw.txt"  # <--- Change this name
    JSON_OUTPUT_FILE = "lexicore_calvin_institutes.json" # <--- Change this name
    SOURCE_TITLE = "Calvin's Institutes"  # <--- Title for the 'scripture_source' field
    SEGMENT_CATEGORY = "Commentary"      # <--- Type for the 'segment_type' field
    SOURCE_LANGUAGE = "English"          # <--- Language for the 'original_language' field
    # --- END CONFIGURATION ---

    format_text_to_json(
        input_file=RAW_INPUT_FILE,
        output_file=JSON_OUTPUT_FILE,
        source_name=SOURCE_TITLE,
        segment_type=SEGMENT_CATEGORY,
        language=SOURCE_LANGUAGE
    )