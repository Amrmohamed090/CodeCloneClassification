import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import random
from typing import List, Tuple, Dict
from tqdm import tqdm
from sqlalchemy import create_engine, text
import numpy as np


class FullCodeCloneDataset(Dataset):
    def __init__(self, train_ratio, val_ratio, maxGroupsize= 10):
        pass

    
    def load_data(self):
        #return train_data, val_data, test_data
        pass

    def check_data_leakage(self, train_data, val_data, test_data):
        pass
    

class POJDataset(Dataset):
    def __init__(
        self,
        engine,
        data,
        batch_size: int = 32,
        tokenizer_name: str ="microsoft/codebert-base",
        max_length: int = 512,
    ):
        """
        Args:
            engine: SQLAlchemy engine to connect to the database
            batch_size: Batch size for loading data
            tokenizer_name: Name of the tokenizer to use
            max_length: Maximum length of tokenized sequences
        """
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        # Load identifiers for clone pairs and false positives from the database
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        function_id_one, function_id_two, is_clone = self.data[idx]
        code1 = self.fetch_code(function_id_one)
        code2 = self.fetch_code(function_id_two)

        # Define half the max length (minus 3 for [CLS], [SEP], [SEP] tokens)
        half_max_length = (self.max_length - 3) // 2

        # Tokenize each code snippet individually
        tokens_code1 = self.tokenizer(
            code1,
            padding="longest",
            truncation=False,
            return_tensors="pt"
        )
        tokens_code2 = self.tokenizer(
            code2,
            padding="longest",
            truncation=False,
            return_tensors="pt"
        )

        # Original token lengths before truncation
        original_length_code1 = tokens_code1["input_ids"].size(1)
        original_length_code2 = tokens_code2["input_ids"].size(1)

        # Tokenize again with truncation
        tokens_code1 = self.tokenizer(
            code1,
            max_length=half_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        tokens_code2 = self.tokenizer(
            code2,
            max_length=half_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Calculate how many tokens were truncated
        truncated_code1 = max(0, original_length_code1 - half_max_length)
        truncated_code2 = max(0, original_length_code2 - half_max_length)

        print(f"Code snippet 1 truncated by {truncated_code1} tokens.")
        print(f"Code snippet 2 truncated by {truncated_code2} tokens.")

        # Combine input_ids and attention masks with [CLS] and [SEP] tokens
        input_ids = torch.cat([
            torch.tensor([self.tokenizer.cls_token_id]),
            tokens_code1["input_ids"].squeeze(),
            torch.tensor([self.tokenizer.sep_token_id]),
            tokens_code2["input_ids"].squeeze(),
            torch.tensor([self.tokenizer.sep_token_id])
        ], dim=0)

        attention_mask = torch.cat([
            torch.tensor([1]),  # CLS token attention
            tokens_code1["attention_mask"].squeeze(),
            torch.tensor([1]),  # SEP token attention
            tokens_code2["attention_mask"].squeeze(),
            torch.tensor([1])   # SEP token attention
        ], dim=0)

        # Ensure final length is exactly self.max_length by padding or truncating
        input_ids = input_ids[:self.max_length]
        attention_mask = attention_mask[:self.max_length]

        # Pad if needed (in case combined length is less than max_length)
        padding_length = self.max_length - input_ids.size(0)
        if padding_length > 0:
            input_ids = torch.cat([input_ids, torch.zeros(padding_length, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(padding_length, dtype=torch.long)])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(is_clone, dtype=torch.float32)
        }

    def fetch_code(self, function_id: int) -> str:
        """Fetch code from the database using the function ID."""
        query = f"""
        SELECT text FROM pretty_printed_functions WHERE function_id = {function_id};
        """
        with self.engine.connect() as connection:
            result = connection.execute(text(query), {'function_id': function_id})
            code = result.scalar()  # Get the single text result
        return code
    
def get_dataloaders(
    engine,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    tokenizer_name: str = "microsoft/codebert-base",
    max_length: int = 512,
    num_workers=4,  # Disable multiprocessing
) -> Tuple[DataLoader, DataLoader, DataLoader, CodeCloneDataset, CodeCloneDataset, CodeCloneDataset]:
    """
    Create train, validation, and test dataloaders along with the original datasets
    """
    full_dataset = FullCodeCloneDataset(engine, train_ratio=train_ratio, val_ratio=val_ratio)

    train_dataset = CodeCloneDataset(engine, full_dataset.train_data, batch_size, tokenizer_name, max_length)
    val_dataset = CodeCloneDataset(engine, full_dataset.validation_data, batch_size, tokenizer_name, max_length)
    test_dataset = CodeCloneDataset(engine, full_dataset.test_data, batch_size, tokenizer_name, max_length)
    
    print(f"dataset is ready: n_sample in train = {len(train_dataset)} n_sample_test = {len(val_dataset)} n_sample_validation = {len(test_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    return train_loader, val_loader, test_loader

