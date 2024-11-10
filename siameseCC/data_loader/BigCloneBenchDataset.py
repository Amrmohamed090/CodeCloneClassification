import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import random
from typing import List, Tuple, Dict
from tqdm import tqdm
from sqlalchemy import create_engine, text
import numpy as np

class NaiveFullCodeCloneDataset(Dataset):
    def __init__(self, engine, train_ratio, val_ratio, maxGroupsize=10):
        self.engine = engine
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.maxGroupsize = maxGroupsize
        self.train_data, self.validation_data, self.test_data = self.load_data()

    def load_data(self) -> List[Tuple[int, int, int]]:
        """Load function IDs and labels from the database, then split without overlapping IDs."""
        # SQL queries to get clone and non-clone pairs
        clone_query = """
        SELECT 
            c.function_id_one,
            c.function_id_two,
            1 AS is_clone
        FROM clones c
        WHERE c.similarity_token <= 0.8
        LIMIT 10000;
        
        """
        false_positive_query = """
        SELECT
            fp.function_id_one,
            fp.function_id_two,
            0 AS is_clone
        FROM false_positives fp

        """
        
        with self.engine.connect() as connection:
            clone_result = connection.execute(text(clone_query))
            clone_pairs = [(row[0], row[1], row[2]) for row in clone_result]

            false_positive_result = connection.execute(text(false_positive_query))
            false_positives = [(row[0], row[1], row[2]) for row in false_positive_result]

        # Collect all unique function IDs
        function_ids = set([func1 for func1, _, _ in clone_pairs] + [func2 for _, func2, _ in clone_pairs])
        function_ids = list(function_ids)

        # Split function IDs for train, validation, and test sets
        num_train = int(len(function_ids) * self.train_ratio)
        num_val = int(len(function_ids) * self.val_ratio)
        train_ids = set(function_ids[:num_train])
        val_ids = set(function_ids[num_train:num_train + num_val])
        test_ids = set(function_ids[num_train + num_val:])

        # Assign pairs to train, validation, or test sets based on function IDs
        train_data, val_data, test_data = [], [], []
        for func1, func2, label in clone_pairs:
            if func1 in train_ids :
                train_data.append((func1, func2, label))
            elif func1 in val_ids :
                val_data.append((func1, func2, label))
            elif func1 in test_ids:
                test_data.append((func1, func2, label))

        train_data_false , val_data_false ,test_data_false = [], [], []

        counter = 0

        while len(train_data_false) < len(train_data) and counter < len(false_positives):
            func1, func2, label = false_positives[counter]
            if func1 not in test_ids and func2 not in test_ids and func1 not in val_ids and func2 not in val_ids:
                train_data_false.append((func1, func2, label))
            counter+=1

        while len(val_data_false) < len(val_data) and counter < len(false_positives):
            func1, func2, label = false_positives[counter]
            if func1 not in test_ids and func2 not in test_ids and func1 not in train_ids and func2 not in train_ids:
                val_data_false.append((func1, func2, label))
            counter +=1

        while len(test_data_false) < len(test_data) and counter < len(false_positives):
            func1, func2, label = false_positives[counter]
            if func1 not in train_ids and func2 not in train_ids and func1 not in val_ids and func2 not in val_ids:
                test_data_false.append((func1, func2, label))
            counter +=1

        print(len(train_data_false))
        print(len(test_data_false))
        print(len(val_data_false))
        # Print statistics for verification
        train_data += train_data_false
        test_data += test_data_false
        val_data += val_data_false
        print("\nDataset statistics:")
        for name, dataset in [("Train", train_data), ("Validation", val_data), ("Test", test_data)]:
            clones = sum(1 for _, _, label in dataset if label == 1)
            non_clones = sum(1 for _, _, label in dataset if label == 0)
            print(f"{name}: Total={len(dataset)}, Clones={clones}, Non-clones={non_clones}")

        #train_data, val_data, test_data = self.check_data_leakage( train_data, val_data, test_data)
        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)
        return train_data, val_data, test_data
    
    def check_data_leakage(self, train_data, val_data, test_data):
        # Collect all unique IDs from each set
        def create_id_set(data):
            id_set = set()
            for id1, id2, _ in data:
                id_set.add(id1)
                id_set.add(id2)
            return id_set
        
        train_ids = create_id_set(train_data)
        val_ids = create_id_set(val_data)
        test_ids = create_id_set(test_data)

        # Find overlapping IDs
        train_val_overlap = train_ids.intersection(val_ids)
        val_test_overlap = val_ids.intersection(test_ids)
        train_test_overlap = train_ids.intersection(test_ids)

        # Count original sizes
        original_train_size = len(train_data)
        original_val_size = len(val_data)
        original_test_size = len(test_data)

        # Filter out pairs where BOTH functions appear in another set
        clean_train = []
        clean_val = []
        clean_test = []

        # Clean training data
        for row in train_data:
            id1, id2, label = row
            if not (id1 in train_val_overlap and id2 in train_val_overlap) and \
            not (id1 in train_test_overlap and id2 in train_test_overlap):
                clean_train.append(row)

        # Clean validation data
        for row in val_data:
            id1, id2, label = row
            if not (id1 in train_val_overlap and id2 in train_val_overlap) and \
            not (id1 in val_test_overlap and id2 in val_test_overlap):
                clean_val.append(row)

        # Clean test data
        for row in test_data:
            id1, id2, label = row
            if not (id1 in train_test_overlap and id2 in train_test_overlap) and \
            not (id1 in val_test_overlap and id2 in val_test_overlap):
                clean_test.append(row)

        # Calculate removals
        removed_train = original_train_size - len(clean_train)
        removed_val = original_val_size - len(clean_val)
        removed_test = original_test_size - len(clean_test)

        print(f"\nWarning: Rows removed due to overlapping IDs:")
        print(f"Train set: {removed_train} rows removed")
        print(f"Validation set: {removed_val} rows removed")
        print(f"Test set: {removed_test} rows removed")

        print("\nCleaned Dataset statistics:")
        for name, dataset in [("Train", clean_train), ("Validation", clean_val), ("Test", clean_test)]:
            clones = sum(1 for _, _, label in dataset if label == 1)
            non_clones = sum(1 for _, _, label in dataset if label == 0)
            print(f"{name}: Total={len(dataset)}, Clones={clones}, Non-clones={non_clones}")

        return clean_train, clean_val, clean_test




    

class CodeCloneDataset(Dataset):
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
        self.engine = engine
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
    full_dataset = NaiveFullCodeCloneDataset(engine, train_ratio=train_ratio, val_ratio=val_ratio)

    train_dataset = CodeCloneDataset(engine, full_dataset.train_data, batch_size, tokenizer_name, max_length)
    val_dataset = CodeCloneDataset(engine, full_dataset.validation_data, batch_size, tokenizer_name, max_length)
    test_dataset = CodeCloneDataset(engine, full_dataset.test_data, batch_size, tokenizer_name, max_length)
    
    print(f"dataset is ready: n_sample in train = {len(train_dataset)} n_sample_test = {len(val_dataset)} n_sample_validation = {len(test_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    return train_loader, val_loader, test_loader

