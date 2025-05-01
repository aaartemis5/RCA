import json
import copy
import re

# Load your JSON file
input_file = 'RCA_ABSgroup_grouped_text_chunks.json'
output_file = 'RCA_by_ABSgroup.json'

# Read the input data
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Initialize output list
reformed_data = []

# Markers to detect cuts
markers = [
    "Primary DirFicutty SOURCE",
    "Primary Dirricutty SOURCE",
]

# Regex to catch garbage lines like "ener e erence eee eee eee"
garbage_pattern = re.compile(r'(ener\s+e+\s+erence\s+e+|e+\s+e+\s+e+)', re.IGNORECASE)

# Patterns to check in second pass
final_cut_patterns = [
    "Root Cause ANALysis HANDBOOK",
    "APPENDIX A"
]

typical_issues_keyword = "Typical Issues"

def clean_text(text):
    # Remove markers
    for marker in markers:
        text = text.replace(marker, '')
    # Remove garbage patterns
    text = garbage_pattern.sub('', text)
    return text.strip()

def find_first_marker(text):
    lowest_index = len(text) + 1
    found_marker = None

    for marker in markers:
        idx = text.find(marker)
        if idx != -1 and idx < lowest_index:
            lowest_index = idx
            found_marker = marker

    match = garbage_pattern.search(text)
    if match:
        idx = match.start()
        if idx < lowest_index:
            lowest_index = idx
            found_marker = match.group(0)

    return lowest_index if found_marker else -1, found_marker

def count_markers(text):
    count = 0
    for marker in markers:
        count += text.count(marker)
    count += len(garbage_pattern.findall(text))
    return count

# First pass
def process_object(obj):
    text = obj['text_chunk']

    # Step 1: Try splitting at 'Typical Issues' first
    typical_index = text.find(typical_issues_keyword)
    if typical_index != -1:
        before_typical = text[:typical_index].strip()
        after_typical = text[typical_index + len(typical_issues_keyword):].strip()

        before_typical = clean_text(before_typical)
        after_typical = clean_text(after_typical)

        if before_typical:
            obj['text_chunk'] = before_typical
            reformed_data.append(obj)

        if after_typical:
            new_obj = copy.deepcopy(obj)
            new_obj['start_page'] = obj['start_page'] + 1
            new_obj['text_chunk'] = after_typical
            process_object(new_obj)  # Recursively process the new object

    else:
        # Step 2: No 'Typical Issues', handle markers normally
        marker_count = count_markers(text)

        if marker_count >= 1:
            cut_index, marker_found = find_first_marker(text)
            if cut_index != -1:
                new_text = text[:cut_index].strip()
                new_text = clean_text(new_text)
                if new_text:
                    obj['text_chunk'] = new_text
                    reformed_data.append(obj)
        else:
            # No marker, clean and keep
            text = clean_text(text)
            if text:
                obj['text_chunk'] = text
                reformed_data.append(obj)

# Second pass
def final_pass_cleanup(data):
    for obj in data:
        text = obj['text_chunk']
        cut_positions = []

        for pattern in final_cut_patterns:
            idx = text.find(pattern)
            if idx != -1:
                cut_positions.append(idx)

        if cut_positions:
            first_cut = min(cut_positions)
            text = text[:first_cut].strip()

        obj['text_chunk'] = text

# Third pass
# Third pass: Full last fullstop + unicode garbage clean
# Third pass: Last fullstop + remove unicode escape + remove non-ASCII symbols
def last_fullstop_and_unicode_cleanup(data):
    unicode_escape_pattern = re.compile(r'\\u[0-9a-fA-F]{4}')
    
    for obj in data:
        text = obj['text_chunk']
        
        # Step 1: Keep text only up to last fullstop
        last_dot_index = text.rfind('.')
        if last_dot_index != -1:
            text = text[:last_dot_index + 1].strip()  # Keep the dot

        # Step 2: Remove literal unicode escape sequences like \u00a2
        text = unicode_escape_pattern.sub('', text)

        # Step 3: Remove non-ASCII characters (anything above 127)
        text = ''.join(c for c in text if ord(c) < 128)

        obj['text_chunk'] = text

# Stage 1: Process initial logic
for obj in data:
    process_object(copy.deepcopy(obj))

# Stage 2: Clean Root Cause/Appendix patterns
final_pass_cleanup(reformed_data)

# Stage 3: Last fullstop cut and \u garbage clean
last_fullstop_and_unicode_cleanup(reformed_data)

# Save the reformed data
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(reformed_data, f, indent=2)

print(f"✅ Reformed data written to {output_file} after full three-pass cleaning (including \\u word removal)!")
