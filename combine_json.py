import os
import json

SCANNED_FILE_TRACKER = 'scanned_files.json'
COMBINED_OUTPUT = 'combined_output.json'

def load_json(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def save_json(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def get_all_json_files(base_path):
    json_files = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith('.json') and file not in [SCANNED_FILE_TRACKER, COMBINED_OUTPUT]:
                json_files.append(os.path.join(root, file))
    return json_files

def main():
    # Load previously scanned files
    scanned_files = set()
    if os.path.exists(SCANNED_FILE_TRACKER):
        scanned_files = set(load_json(SCANNED_FILE_TRACKER) or [])

    # Find all JSON files in the directory
    all_json_files = set(get_all_json_files('.'))

    # Identify new files
    new_files = all_json_files - scanned_files
    if not new_files:
        print("No new JSON files found. Combined file is up-to-date.")
        return

    # Load existing combined data
    combined_data = []
    if os.path.exists(COMBINED_OUTPUT):
        combined_data = load_json(COMBINED_OUTPUT) or []

    # Combine new JSON files
    for filepath in new_files:
        data = load_json(filepath)
        if data is not None:
            if isinstance(data, list):
                combined_data.extend(data)
            else:
                combined_data.append(data)

    # Save updated combined JSON
    save_json(COMBINED_OUTPUT, combined_data)

    # Update scanned files
    save_json(SCANNED_FILE_TRACKER, list(all_json_files))

    print(f"Combined {len(new_files)} new JSON files into {COMBINED_OUTPUT}.")

if __name__ == "__main__":
    main()
