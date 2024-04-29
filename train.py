import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from datasets import load_dataset
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq, BertTokenizerFast, EncoderDecoderModel, EncoderDecoderConfig, BertConfig

# Function to transform text to lowercase
def lowercase_text(text):
    if isinstance(text, list):
        return [item.lower() for item in text]
    elif isinstance(text, str):
        return text.lower()
    else:
        return text

class Tokenizer(BertTokenizerFast):
    def __init__(self, DATA_HUB='atlasia/ATAM', vocab_file_path = "vocab.txt", special_tokens = ["<sos>", "<eos>", "<pad>", "<unk>"], arabizi_column='darija_arabizi', arabic_column='darija_arabic'):
        # Load the dataset
        dataset = load_dataset(DATA_HUB)['train']
        
        # Transform the column to lowercase
        print(f'[INFO] Transforming the {arabizi_column} column to lowercase')
        dataset = dataset.map(lambda example: {arabizi_column: lowercase_text(example[arabizi_column])}, batched=True)
        
        self.data = dataset.to_pandas().values.tolist()
        
        # Create a set of all unique characters in the source and target languages
        self.arabizi_chars = set(''.join([d[0] for d in self.data]))
        self.arabic_chars = set(''.join([d[1] for d in self.data]))
        
        # Create a dictionary mapping each character to a unique index
        self.char2idx_ary = {char: idx for idx, char in enumerate(self.arabizi_chars)}
        self.char2idx_ar = {char: idx for idx, char in enumerate(self.arabic_chars)}

        # Calculate the size of the vocabulary including special tokens
        self.vocab_size_src = len(self.char2idx_ary)
        self.vocab_size_tgt = len(self.char2idx_ar)
        print(f"[INFO] Vocabulary size for source language (Arabizi): {self.vocab_size_src}")
        print(f"[INFO] Vocabulary size for target language (Arabic): {self.vocab_size_tgt}")
        
        # Get all unique characters from source and target languages
        unique_chars = set(char for data_point in self.data for char in data_point[0]) | set(char for data_point in self.data for char in data_point[1])

        # Add special tokens to the unique characters
        unique_chars.update(special_tokens)

        # Sort the unique characters alphabetically
        sorted_unique_chars = sorted(list(unique_chars))

        # Write the sorted unique characters to the vocab.txt file
        with open(vocab_file_path, "w", encoding="utf-8") as vocab_file:
            # Write special tokens first
            for token in special_tokens:
                vocab_file.write(token + "\n")
            # Write sorted unique characters
            for char in sorted_unique_chars:
                vocab_file.write(char + "\n")

        # Initialize tokenizer
        self.tokenizer =  BertTokenizerFast(
            vocab_file="vocab.txt",
            bos_token="<sos>",
            eos_token="<eos>",
            pad_token="<pad>",
            unk_token="<unk>",
            trainable=True  # Set to True for learnable tokenizer
        )

        
if __name__ == "__main__":
    # learning parameters
    learning_rate = 5e-4
    batch_size = 384
    n_epochs=30
    weight_decay=0.01
    save_total_limit=3
    bf16=True
    warmup_ratio=0.01
    gradient_accumulation_steps=1
    test_size = 0.01 #(1% of the data, around 700 samples)

    # Transformer parameters
    d_model=512 
    nhead=8
    num_encoder_layers=4
    num_decoder_layers=4

    # Data paths
    DATA_PATH = 'atlasia/ATAM'

    # Set the source and target (labels)
    src='darija_arabizi'
    tgt='darija_arabic'
    
    # Saving Hub
    MODEL_HUB = 'BounharAbdelaziz/Transliteration-Moroccan-Darija'

    # Load training dataset from Hugging Face datasets
    dataset = load_dataset(DATA_PATH, split="train")
    dataset = dataset.train_test_split(test_size=test_size)
    
    # Tokenizer
    tokenizer = Tokenizer(DATA_HUB='atlasia/ATAM', vocab_file_path = "vocab.txt", special_tokens = ["<sos>", "<eos>", "<pad>", "<unk>"]).tokenizer
    
    # Define the encoder configuration
    encoder_config = BertConfig(
        hidden_size=d_model,
        num_attention_heads=nhead,
        num_hidden_layers=num_encoder_layers,
    )

    # Define the decoder configuration
    decoder_config = BertConfig(
        hidden_size=d_model,
        num_attention_heads=nhead,
        num_hidden_layers=num_decoder_layers,
    )
    # Instantiate the model configuration
    config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
    # Set the decoder_start_token_id attribute in the configuration
    config.decoder_start_token_id = tokenizer.bos_token_id
    # Set the pad_token_id attribute in the configuration
    config.pad_token_id = tokenizer.pad_token_id
    # Instantiate the model using the configuration
    model = EncoderDecoderModel(config=config)
    
    # prepare the data for training    
    def preprocess_function(examples):
        inputs = [str(example) for example in examples[src]]
        targets = [str(example) for example in examples[tgt]]
        # print(f'inputs: {inputs}')
        # print(f'targets: {targets}')
        
        input_tokenized = tokenizer.batch_encode_plus(
            inputs,
            max_length=128,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        target_tokenized = tokenizer.batch_encode_plus(
            targets,
            max_length=128,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        model_inputs = {
            "input_ids": input_tokenized["input_ids"],
            "attention_mask": input_tokenized["attention_mask"],
            "labels": target_tokenized["input_ids"],
        }
        
        return model_inputs
    
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    tokenized_data = dataset.map(preprocess_function, batched=True)

    # Define the training arguments
    training_args = TrainingArguments(
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        num_train_epochs=n_epochs,
        weight_decay=weight_decay,
        save_total_limit=save_total_limit,
        bf16=bf16,
        warmup_ratio=warmup_ratio,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_dir="logs",
        report_to="tensorboard",
        output_dir=MODEL_HUB,
        push_to_hub=False,
    )

    # Instantiate the Trainer class
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Start training
    trainer.train()
    
    # Push the model to the Hub
    trainer.push_to_hub(repo_name=MODEL_HUB)