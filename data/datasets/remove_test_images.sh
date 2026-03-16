#!/bin/bash
#
# Remove COCO 2014 test images from the COCO 2017 training set.
# COCO 2017 reuses images from COCO 2014, so test images must be removed
# to prevent data leakage. Uses the Fischer et al. labels (labels 1 and 2)
# from 02_coco_val_labels.csv to identify the 1,030 test images.
# Matching images are moved to a backup folder, not deleted.
#
# Usage: cd data/datasets && bash remove_test_images.sh

# Configuration
CSV_FILE="ds_coco/02_coco_val_labels.csv"
TRAIN_FOLDER="train_coco_2017"
BACKUP_FOLDER="train_coco_2017_removed_test_images"

# Dry run option - set to "true" to see what would happen without actually moving files
DRY_RUN="false"  # Change to "true" for dry run

# Check if CSV file exists
if [ ! -f "$CSV_FILE" ]; then
    echo "Error: CSV file '$CSV_FILE' not found!"
    exit 1
fi

# Check if train folder exists
if [ ! -d "$TRAIN_FOLDER" ]; then
    echo "Error: Training folder '$TRAIN_FOLDER' not found!"
    exit 1
fi

# Create backup folder (only if not dry run)
if [ "$DRY_RUN" = "false" ]; then
    mkdir -p "$BACKUP_FOLDER"
fi

# Counter for statistics
moved_count=0
not_found_count=0
skipped_count=0

if [ "$DRY_RUN" = "true" ]; then
    echo "=== DRY RUN MODE - No files will be moved ==="
else
    echo "=== LIVE MODE - Files will be moved ==="
fi

echo "Processing test images from $CSV_FILE (labels 1 and 2 only)"
echo "Source folder: $TRAIN_FOLDER"
echo "Destination folder: $BACKUP_FOLDER"
echo ""

# Read CSV file line by line, skipping header
{
    read # skip header
    while IFS=',' read -r image_name label || [ -n "$image_name" ]; do
        # Remove any potential whitespace/quotes from label
        label=$(echo "$label" | tr -d ' "')
        image_name=$(echo "$image_name" | tr -d ' "')
        
        # Only process images with labels 1 or 2
        if [ "$label" = "1" ] || [ "$label" = "2" ]; then
            # Extract image ID from COCO 2014 format
            # Convert "COCO_val2014_000000000042.jpg" to "000000000042.jpg"
            if [[ "$image_name" =~ COCO_val2014_([0-9]+\.jpg)$ ]]; then
                image_id="${BASH_REMATCH[1]}"
            else
                echo "Warning: Unexpected filename format: $image_name"
                continue
            fi
            
            source_path="$TRAIN_FOLDER/$image_id"
            dest_path="$BACKUP_FOLDER/$image_id"
            
            if [ -f "$source_path" ]; then
                if [ "$DRY_RUN" = "true" ]; then
                    echo "[DRY RUN] Would move: $image_id (label: $label)"
                else
                    mv "$source_path" "$dest_path"
                    echo "Moved: $image_id (label: $label)"
                fi
                ((moved_count++))
            else
                echo "Not found: $image_id (label: $label)"
                ((not_found_count++))
            fi
        else
            echo "Skipping: $image_name (label: $label)"
            ((skipped_count++))
        fi
    done
} < "$CSV_FILE"

echo ""
echo "=== SUMMARY ==="
if [ "$DRY_RUN" = "true" ]; then
    echo "DRY RUN - No files were actually moved"
fi
echo "Images that would be/were moved: $moved_count"
echo "Images not found in training folder: $not_found_count"
echo "Images skipped (label != 1,2): $skipped_count"

# Show some statistics about what labels we found
echo ""
echo "=== LABEL STATISTICS ==="
echo "Processing CSV to show label distribution..."
{
    read # skip header
    label_1_count=0
    label_2_count=0
    label_other_count=0
    
    while IFS=',' read -r image_name label || [ -n "$image_name" ]; do
        label=$(echo "$label" | tr -d ' "')
        case "$label" in
            "1") ((label_1_count++)) ;;
            "2") ((label_2_count++)) ;;
            *) ((label_other_count++)) ;;
        esac
    done
    
    echo "Label 1: $label_1_count images"
    echo "Label 2: $label_2_count images"  
    echo "Other labels: $label_other_count images"
    echo "Total in CSV: $((label_1_count + label_2_count + label_other_count)) images"
} < "$CSV_FILE"