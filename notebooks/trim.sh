#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 <input_directory> <output_directory>"
    echo "Example: $0 /path/to/input /path/to/output"
    exit 1
}

# Check if correct number of arguments provided
if [ $# -ne 2 ]; then
    echo "Error: Incorrect number of arguments."
    usage
fi

# Get arguments
input_dir="$1"
output_dir="$2"

# Validate input directory exists
if [ ! -d "$input_dir" ]; then
    echo "Error: Input directory '$input_dir' does not exist."
    exit 1
fi

# Check if output directory exists, create if not
if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
    echo "Created output directory: $output_dir"
fi

# Check if there are any JSON files in input directory
json_count=$(find "$input_dir" -maxdepth 1 -name "*.json" | wc -l)
if [ $json_count -eq 0 ]; then
    echo "Warning: No JSON files found in '$input_dir'"
    exit 0
fi

echo "Processing $json_count JSON files from '$input_dir' to '$output_dir'"

# Loop through all JSON files in the input directory
for file in "$input_dir"/*.json; do
    # Skip if no files match (in case of empty directory)
    [ -f "$file" ] || continue
    
    # Extract filename without extension
    filename=$(basename "$file" .json)
    
    # Build output filename with path
    output_file="$output_dir/$filename.json"
    
    # Run the command with proper arguments
    python3 00-trim_annotations_to_border_single_frame.py \
           -i "$file" \
           -o "$output_file"
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "✓ Processed: $filename.json"
    else
        echo "✗ Failed to process: $filename.json"
    fi
done

echo "All JSON files processed!"