#!/bin/bash

# List of dataset slugs (you can replace these with the actual ones you want)
datasets=(
  "digital-production/bildachersplitted:1"
  "digital-production/lightlysplitted:1"
  "digital-production/haldennord10splitted:1"
)

# Loop through and start each download in the background
for dataset in "${datasets[@]}"
do
  echo "Starting download for $dataset..."
  darwin dataset pull "$dataset" &
done

# Wait for all background jobs to complete
wait

echo "All downloads finished."
