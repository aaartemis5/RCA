import os
import json

# You can tweak these if you like:
BASE_DIR = 'data_parsing'
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
    for root, dirs, files in os.walk(base_path):
        # 👉 Don’t even descend into any .venv folder
        dirs[:] = [d for d in dirs if d != '.venv']
        # skip if current folder is a .venv
        if '.venv' in root.split(os.sep):
            continue
        for file in files:
            if not file.endswith('.json'):
                continue
            # don’t re-process our own trackers if they somehow live under data_parsing
            if file in (os.path.basename(SCANNED_FILE_TRACKER), os.path.basename(COMBINED_OUTPUT)):
                continue
            json_files.append(os.path.join(root, file))
    return json_files

def main():
    # Load list of already-scanned files (if any)
    scanned_files = set()
    if os.path.exists(SCANNED_FILE_TRACKER):
        prev = load_json(SCANNED_FILE_TRACKER)
        scanned_files = set(prev) if isinstance(prev, list) else set()

    # Gather all JSONs under data_parsing/
    all_json_files = set(get_all_json_files(BASE_DIR))

    # Figure out which ones are new
    new_files = all_json_files - scanned_files
    if not new_files:
        print("No new JSON files found under data_parsing/. Combined output is up-to-date.")
        return

    # Start with whatever we’ve already combined
    combined_data = []
    if os.path.exists(COMBINED_OUTPUT):
        prev = load_json(COMBINED_OUTPUT)
        combined_data = prev if isinstance(prev, list) else []

    # Load & merge each new file
    for filepath in sorted(new_files):
        data = load_json(filepath)
        if data is None:
            continue
        if isinstance(data, list):
            combined_data.extend(data)
        else:
            combined_data.append(data)

    # Write out the updated combined list
    save_json(COMBINED_OUTPUT, combined_data)

    # Update our tracker
    save_json(SCANNED_FILE_TRACKER, sorted(all_json_files))

    print(f"✅ Combined {len(new_files)} new file(s) into {COMBINED_OUTPUT}.")

if __name__ == '__main__':
    main()
