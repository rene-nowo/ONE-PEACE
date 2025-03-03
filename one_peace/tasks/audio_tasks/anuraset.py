import pandas as pd
import os
import random

# Load metadata
metadata = pd.read_csv('anuraset/metadata.csv')

# Add uniq_id if not present
if 'uniq_id' not in metadata.columns:
    metadata['uniq_id'] = [f'anura_{i}' for i in range(len(metadata))]

# Prepare required columns
# Adjust this based on your actual metadata.csv format
required_columns = ['uniq_id', 'audio', 'text', 'duration']
for col in required_columns:
    if col not in metadata.columns:
        if col == 'audio':
            # Assuming audio filenames are in a column named 'filename'
            metadata['audio'] = metadata['filename'].apply(lambda x: os.path.join('anuraset/audio', x))
        elif col == 'text':
            # Create text labels from your label columns, joining multiple labels with '|'
            # This is an example - adjust based on your actual column names
            metadata['text'] = metadata[['label1', 'label2', 'label3']].apply(
                lambda x: '|'.join([l for l in x if pd.notna(l)]), axis=1)
        elif col == 'duration':
            # If duration is missing, you need to calculate it or provide a placeholder
            metadata['duration'] = 10.0  # Replace with actual durations

# Split data into train, valid, test (e.g., 70/15/15 split)
indices = list(range(len(metadata)))
random.shuffle(indices)
train_size = int(0.7 * len(metadata))
valid_size = int(0.15 * len(metadata))
train_indices = indices[:train_size]
valid_indices = indices[train_size:train_size+valid_size]
test_indices = indices[train_size+valid_size:]

train_df = metadata.iloc[train_indices]
valid_df = metadata.iloc[valid_indices]
test_df = metadata.iloc[test_indices]

# Save as TSV files
train_df[required_columns].to_csv('anuraset/anuraset_train.tsv', sep='\t', index=False)
valid_df[required_columns].to_csv('anuraset/anuraset_valid.tsv', sep='\t', index=False)
test_df[required_columns].to_csv('anuraset/anuraset_test.tsv', sep='\t', index=False)