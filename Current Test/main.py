# --- 1. INSTALLS AND IMPORTS ---
!pip install transformers datasets soundfile librosa pandas -q

import os
import torch
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from datasets import Dataset, Audio
from transformers import (
    VitsForConditionalGeneration,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)


# --- 2. DATA LOADING ---
# Define paths to your data
base_path = "/content/drive/MyDrive/dataSets/LJSpeech-1.1"
metadata_path = os.path.join(base_path, "metadata.csv")
wavs_path = os.path.join(base_path, "wavs")

try:
    # Read metadata.csv using pandas
    # The file is pipe-separated, has no header, and we'll ignore quoting issues
    df = pd.read_csv(metadata_path, sep='|', header=None, quoting=3)
    df.columns = ['id', 'transcription', 'normalized_transcription']

    # Create the full path to each audio file
    df['audio'] = df['id'].apply(lambda x: os.path.join(wavs_path, f"{x}.wav"))

    # We only need the audio path and the normalized text
    df = df[['audio', 'normalized_transcription']]
    df = df.rename(columns={'normalized_transcription': 'text'})

    # Create a Hugging Face Dataset from the pandas DataFrame
    dataset = Dataset.from_pandas(df)
    print("✅ Dataset loaded successfully!")

except Exception as e:
    print(f"❌ Error loading or parsing metadata from '{metadata_path}'.")
    print("Please make sure the path is correct and the file exists.")
    print(f"Error details: {e}")
    raise

# --- 3. LOAD PRE-TRAINED MODEL AND PROCESSOR ---
model_name = "kakao-enterprise/vits-ljs"
# Use VitsForConditionalGeneration for training
model = VitsForConditionalGeneration.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)

# --- 4. PREPARE THE DATA ---
# Cast the 'audio' column to the correct 'Audio' feature type.
# This will load and resample the audio on the fly.
dataset = dataset.cast_column(
    "audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate)
)

def preprocess_function(examples):
    """Process the audio and text for the model."""
    audio = [x["array"] for x in examples["audio"]]
    text = examples["text"]

    # The processor handles tokenization and feature extraction
    processed = processor(
        text=text,
        audio_target=audio,
        sampling_rate=processor.feature_extractor.sampling_rate,
        return_tensors="pt",
        padding=True # Pad here to handle variable lengths
    )
    # The VITS model's forward pass expects an argument named 'y'.
    return {"input_ids": processed["input_ids"], "y": processed["input_values"]}

# Apply the preprocessing
dataset = dataset.map(
    preprocess_function,
    is_batched=True,
    batch_size=2,
    remove_columns=dataset.column_names,
    num_proc=2,
)

# --- 5. CREATE A CUSTOM DATA COLLATOR ---
@dataclass
class DataCollatorSpeech:
    """A data collator that simply stacks the already-padded features."""
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # The features are already pre-processed and padded in our `preprocess_function`.
        # We just need to stack them into a batch.
        input_ids = torch.vstack([torch.tensor(feature["input_ids"]) for feature in features])
        y = torch.vstack([torch.tensor(feature["y"]) for feature in features])

        return {"input_ids": input_ids, "y": y}

# Instantiate the data collator
data_collator = DataCollatorSpeech(processor=processor)


# --- 6. DEFINE TRAINING ARGUMENTS ---
training_args = TrainingArguments(
    output_dir="./vits-finetuned-ljspeech",
    per_device_train_batch_size=4,  # Reduced batch size to avoid memory issues
    gradient_accumulation_steps=2,  # Accumulate gradients to simulate a larger batch size
    num_train_epochs=10,
    learning_rate=2e-5,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=20,
    save_steps=500,
    save_total_limit=2,
    fp16=True,
)

# --- 7. CREATE AND RUN THE TRAINER ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=processor,
    data_collator=data_collator,
)

print("🚀 Starting fine-tuning...")
trainer.train()

# --- 8. SAVE THE FINAL MODEL ---
trainer.save_model("./vits-finetuned-ljspeech-final")

print("🎉 Fine-tuning complete! Your custom LJSpeech model is saved.")