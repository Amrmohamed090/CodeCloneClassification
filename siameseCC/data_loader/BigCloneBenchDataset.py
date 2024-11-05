import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import random
from typing import List, Tuple, Dict
from tqdm import tqdm
from sqlalchemy import create_engine, text



class CodeCloneDataset(Dataset):
    def __init__(
        self,
        engine,
        batch_size: int = 32,
        tokenizer_name: str ="unsloth/Llama-3.2-1B-Instruct-GGUF",
        max_length: int = 10000,
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
        self.data = self.load_data()


    def load_data(self) -> List[Tuple[int, int, int]]:
        """Load function IDs and labels from the database."""
        clone_query = """
        SELECT 
            c.function_id_one,
            c.function_id_two,
            1 AS is_clone
        FROM clones c
        WHERE c.min_judges >= 2
        LIMIT 10
        """
        false_positive_query = """
        SELECT
            fp.function_id_one,
            fp.function_id_two,
            0 AS is_clone
        FROM false_positives fp
        WHERE fp.min_judges >= 2
        LIMIT 10
        """
        
        with self.engine.connect() as connection:
            clone_result = connection.execute(text(clone_query))
            clone_pairs = [(row[0], row[1], row[2]) for row in clone_result]

            false_positive_result = connection.execute(text(false_positive_query))
            false_positives = [(row[0], row[1], row[2]) for row in false_positive_result]

        # Concatenate clone pairs and false positives, and shuffle the dataset
        data = clone_pairs + false_positives
        random.shuffle(data)

        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        function_id_one, function_id_two, is_clone = self.clone_pairs[idx]
        code1 = self.fetch_code(function_id_one)
        code2 = self.fetch_code(function_id_two)

        # Combine with [SEP] token and use max_length of Code LLaMA (up to 4096 tokens)
        combined_code = code1 + " [SEP] " + code2
        tokens = self.tokenizer(
            combined_code,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "label": torch.tensor(is_clone, dtype=torch.float32)
        }

    def fetch_code(self, function_id: int) -> str:
        """Fetch code from the database using the function ID."""
        query = """
        SELECT text FROM pretty_printed_functions WHERE function_id = :function_id;
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
    full_dataset = CodeCloneDataset(engine, batch_size, tokenizer_name, max_length)
    
    # Calculate indices for splitting
    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    # Split the dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    check_data_leakage(full_dataset, train_dataset, val_dataset, test_dataset)

    print(f"dataset is ready: n_sample in train = {len(train_dataset)} n_sample_test = {len(val_dataset)} n_sample_validation = {len(test_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    return train_loader, val_loader, test_loader

def check_data_leakage(full_dataset, train_dataset, val_dataset, test_dataset):
    """Check for data leakage between the train, validation, and test sets."""
    train_ids = set([full_dataset.data[idx][0] for idx in train_dataset.indices] + 
                   [full_dataset.data[idx][1] for idx in train_dataset.indices])
    val_ids = set([full_dataset.data[idx][0] for idx in val_dataset.indices] +
                 [full_dataset.data[idx][1] for idx in val_dataset.indices])
    test_ids = set([full_dataset.data[idx][0] for idx in test_dataset.indices] +
                  [full_dataset.data[idx][1] for idx in test_dataset.indices])
    overlapping_ids = train_ids.intersection(val_ids).intersection(test_ids)
    if overlapping_ids:
        print(f"Warning: Data leakage detected. The following function IDs are present in all three sets: {overlapping_ids}")
