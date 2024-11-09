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
        WHERE c.similarity_token <= 0.6
        LIMIT 10000;
        
        """
        false_positive_query = """
        SELECT
            fp.function_id_one,
            fp.function_id_two,
            0 AS is_clone
        FROM false_positives fp
        LIMIT 10000;

        """
        
        with self.engine.connect() as connection:
            clone_result = connection.execute(text(clone_query))
            clone_pairs = [(row[0], row[1], row[2]) for row in clone_result]

            false_positive_result = connection.execute(text(false_positive_query))
            false_positives = [(row[0], row[1], row[2]) for row in false_positive_result]

        # Collect all unique function IDs
        function_ids = set([func1 for func1, _, _ in clone_pairs] + [func2 for _, func2, _ in clone_pairs])
        function_ids = list(function_ids)
        random.shuffle(function_ids)

        # Split function IDs for train, validation, and test sets
        num_train = int(len(function_ids) * self.train_ratio)
        num_val = int(len(function_ids) * self.val_ratio)
        train_ids = set(function_ids[:num_train])
        val_ids = set(function_ids[num_train:num_train + num_val])
        test_ids = set(function_ids[num_train + num_val:])

        # Assign pairs to train, validation, or test sets based on function IDs
        train_data, val_data, test_data = [], [], []
        for func1, func2, label in clone_pairs:
            if func1 in train_ids and func2 in train_ids:
                train_data.append((func1, func2, label))
            elif func1 in val_ids and func2 in val_ids:
                val_data.append((func1, func2, label))
            elif func1 in test_ids and func2 in test_ids:
                test_data.append((func1, func2, label))

        # Collect all unique function IDs For negative pairs
        function_ids = set([func1 for func1, _, _ in false_positives] + [func2 for _, func2, _ in false_positives])
        function_ids = list(function_ids)
        random.shuffle(function_ids)

        # Split function IDs for train, validation, and test sets
        num_train = int(len(function_ids) * self.train_ratio)
        num_val = int(len(function_ids) * self.val_ratio)
        train_ids_false = set(function_ids[:num_train])
        val_ids_false = set(function_ids[num_train:num_train + num_val])
        test_ids_false = set(function_ids[num_train + num_val:])

        for func1, func2, label in clone_pairs:
            if func1 in train_ids_false and func2 in train_ids_false:
                train_data.append((func1, func2, label))
            elif func1 in val_ids_false and func2 in val_ids_false:
                val_data.append((func1, func2, label))
            elif func1 in test_ids_false and func2 in test_ids_false:
                test_data.append((func1, func2, label))

  
        # Print statistics for verification
        print("\nDataset statistics:")
        for name, dataset in [("Train", train_data), ("Validation", val_data), ("Test", test_data)]:
            clones = sum(1 for _, _, label in dataset if label == 1)
            non_clones = sum(1 for _, _, label in dataset if label == 0)
            print(f"{name}: Total={len(dataset)}, Clones={clones}, Non-clones={non_clones}")

        train_data, val_data, test_data = self.check_data_leakage( train_data, val_data, test_data)

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
        val_ids =  create_id_set(val_ids)
        test_ids =  create_id_set(test_ids)
        # Identify overlaps
        val_in_train = val_ids.intersection(train_ids)
        test_in_val = test_ids.intersection(val_ids)
        test_in_train = test_ids.intersection(train_ids)

        # Count original data sizes
        original_train_size = len(train_data)
        original_val_size = len(val_data)
        original_test_size = len(test_data)

        # Filter out any rows with overlapping IDs
        train_data = [row for row in train_data if row[0] not in val_in_train and row[1] not in val_in_train and row[0] not in test_in_train and row[1] not in test_in_train]
        val_data = [row for row in val_data if row[0] not in test_in_val and row[1] not in test_in_val and row[0] not in val_in_train and row[1] not in val_in_train]
        test_data = [row for row in test_data if row[0] not in test_in_val and row[1] not in test_in_val and row[0] not in test_in_train and row[1] not in test_in_train]

        # Calculate number of rows removed
        removed_train = original_train_size - len(train_data)
        removed_val = original_val_size - len(val_data)
        removed_test = original_test_size - len(test_data)

        # Print warning for removed rows
        print(f"\nWarning: Rows removed due to overlapping IDs:")
        print(f"Train set: {removed_train} rows removed")
        print(f"Validation set: {removed_val} rows removed")
        print(f"Test set: {removed_test} rows removed")

        # Print summary of the cleaned data
        print("\nCleaned Dataset statistics (after removing overlapping IDs):")
        for name, dataset in [("Train", train_data), ("Validation", val_data), ("Test", test_data)]:
            clones = sum(1 for _, _, label in dataset if label == 1)
            non_clones = sum(1 for _, _, label in dataset if label == 0)
            print(f"{name}: Total={len(dataset)}, Clones={clones}, Non-clones={non_clones}")

        return train_data, val_data, test_data




class FullCodeCloneDataset(Dataset):
    def __init__(self, engine, train_ratio, val_ratio, maxGroupsize= 10):
        self.engine = engine
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.maxGroupsize = maxGroupsize

        self.train_data, self.validation_data, self.test_data = self.load_data()

    
    def load_data(self) -> List[Tuple[int, int, int]]:
        """Load function IDs and labels from the database."""
        clone_query = """
        SELECT 
            c.function_id_one,
            c.function_id_two,
            1 AS is_clone

        
        FROM clones c
        Where c.similarity_token <= 0.6;

        """
        false_positive_query = """
        SELECT
            fp.function_id_one,
            fp.function_id_two,
            0 AS is_clone
        FROM false_positives fp;
        """
        
        with self.engine.connect() as connection:
            clone_result = connection.execute(text(clone_query))
            clone_pairs = [(row[0], row[1], row[2]) for row in clone_result]

            false_positive_result = connection.execute(text(false_positive_query))
            false_positives = [(row[0], row[1], row[2]) for row in false_positive_result]
        
        # Group the clone pairs
        # if A and B are clones
        # and B and C are clones
        # then A and C are clones
        
        func_to_group, group_sizes = self.build_cloneGroups(clone_pairs)


        train_groups, val_groups, test_groups = self.build_groups_sets(func_to_group)
        train_data_clone, val_data_clone, test_data_clone = self.groups_to_dataset(train_groups, val_groups, test_groups,func_to_group, group_sizes, clone_pairs)
        

        # Add negative pairs

        func_to_group, group_sizes = self.build_cloneGroups(false_positives)


        train_groups, val_groups, test_groups = self.build_groups_sets(func_to_group)
        train_data_nonclone, val_data_nonclone, test_data_nonclone = self.groups_to_dataset(train_groups, val_groups, test_groups,func_to_group, group_sizes, false_positives)
        train_data = train_data_clone + train_data_nonclone
        test_data = test_data_clone + test_data_nonclone
        val_data = val_data_clone + val_data_nonclone
        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)
        print(f"train data {train_data}")
        print("-"*10)
        print("-"*10)
        print("-"*10)
        # print(f"test data {test_data}")
        # print("-"*10)
        # print("-"*10)
        # print("-"*10)
        # print(f"val data {val_data}")
        # Print statistics
        print("\nDataset statistics:")
        for name, dataset in [("Train", train_data), ("Validation", val_data), ("Test", test_data)]:
            clones = sum(1 for _, _, label in dataset if label == 1)
            non_clones = sum(1 for _, _, label in dataset if label == 0)
            print(f"{name}: Total={len(dataset)}, Clones={clones}, Non-clones={non_clones}")
        
        #self.check_data_leakage(train_data, val_data, test_data)

        #print(f"train_data {len(train_data)}", train_data)
        #print(f"validation_data {len(val_data)}", val_data)
        #print(f"test data {len(test_data)}", test_data)
        return train_data, val_data, test_data


    def build_cloneGroups(self, clone_pairs):
        """
        Build groups of transitively related clones.
        Returns a dictionary mapping function_id to group_id.
        """
        group_ids = {}  # key = function_id, value = group_id
        group_sizes = {}
        i = 0
        
        # First pass: assign initial groups
        for func_id_one, func_id_two , _ in clone_pairs:
            
            if func_id_one not in group_ids and func_id_two not in group_ids:
                group_ids[func_id_one] = i
                group_ids[func_id_two] = i
                group_sizes[i] = 2
                i += 1

            elif func_id_one in group_ids and func_id_two not in group_ids:
                group_ids[func_id_two] = group_ids[func_id_one]
                group_sizes[group_ids[func_id_two]] +=1

            elif func_id_one not in group_ids and func_id_two in group_ids:
                group_ids[func_id_one] = group_ids[func_id_two]
                group_sizes[group_ids[func_id_two]] +=1

            elif func_id_one in group_ids and func_id_two in group_ids and not group_ids[func_id_one] == group_ids[func_id_two]:
                # Both are in groups - need to merge groups
                old_group_id = group_ids[func_id_two]
                new_group_id = group_ids[func_id_one]

                group_sizes[new_group_id] += group_sizes[old_group_id]
                # Update all functions in the old group to be in the new group
                for func_id, group in group_ids.items():
                    if group == old_group_id:
                        group_ids[func_id] = new_group_id

        return group_ids, group_sizes

    def build_groups_sets(self, func_to_group):
        # Get unique groups and shuffle them
        unique_groups = list(set(func_to_group.values()))

        random.shuffle(unique_groups)
        
        # Split groups into train/val/test
        train_size = int(len(unique_groups) * self.train_ratio)
        val_size = int(len(unique_groups) * self.val_ratio)
        
        train_groups = set(unique_groups[:train_size])
        val_groups = set(unique_groups[train_size:train_size + val_size])
        test_groups = set(unique_groups[train_size + val_size:])

        return train_groups, val_groups, test_groups
    
    def groups_to_dataset(self, train_groups, val_groups, test_groups ,func_to_group, group_sizes, pairs):
        # Create positive pairs based on group assignments
        train_data, val_data, test_data = [], [], []
        
        funtions_per_group_added = {}
        for group_id in group_sizes:
            funtions_per_group_added[group_id] = 0

        # Add positive pairs
        for func1, func2, is_clone in pairs:
            group_id = func_to_group[func1]  # Since both functions are in same group, can use either
            if group_id in train_groups and funtions_per_group_added[group_id] < self.maxGroupsize:
                train_data.append((func1, func2, is_clone))
                funtions_per_group_added[group_id] +=1
            elif group_id in val_groups and funtions_per_group_added[group_id] < self.maxGroupsize:
                val_data.append((func1, func2, is_clone))
                funtions_per_group_added[group_id] +=1
            elif group_id in test_groups and funtions_per_group_added[group_id] < self.maxGroupsize:
                test_data.append((func1, func2, is_clone))
                funtions_per_group_added[group_id] +=1
        return train_data, val_data, test_data
    
    def check_data_leakage(self, train_data, val_data, test_data):
        # Collect all unique IDs from each set
        unique_train_ids = {id for row in train_data for id in (row[0], row[1])}
        unique_val_ids = {id for row in val_data for id in (row[0], row[1])}
        unique_test_ids = {id for row in test_data for id in (row[0], row[1])}

        # Check for overlaps
        warnings = []

        # Validation vs Training
        val_in_train = unique_val_ids.intersection(unique_train_ids)
        for id in val_in_train:
            warnings.append(f"WARNING: ID {id} exists in both validation and training.")

        # Test vs Validation
        test_in_val = unique_test_ids.intersection(unique_val_ids)
        for id in test_in_val:
            warnings.append(f"WARNING: ID {id} exists in both test and validation.")

        # Test vs Training
        test_in_train = unique_test_ids.intersection(unique_train_ids)
        for id in test_in_train:
            warnings.append(f"WARNING: ID {id} exists in both test and training.")

        # Output warnings
        if warnings:
            print("\n".join(warnings))
        else:
            print("No data leakage detected between train, validation, and test sets.")


# class FullCodeCloneDataset(Dataset):
#     def __init__(self, engine, train_ratio, val_ratio):
#         self.engine = engine
#         self.train_ratio = train_ratio
#         self.val_ratio = val_ratio
#         self.train_data, self.validation_data, self.test_data = self.load_data()


    
#     def load_data(self) -> List[Tuple[int, int, int]]:
#         """Load function IDs and labels from the database."""
#         clone_query = """
#         SELECT 
#             c.function_id_one,
#             c.function_id_two,
#             1 AS is_clone
#         FROM clones c
#         WHERE c.min_judges >= 2
#         AND c.similarity_token <= 0.6
#         LIMIT 10000
#         """
#         false_positive_query = """
#         SELECT
#             fp.function_id_one,
#             fp.function_id_two,
#             0 AS is_clone
#         FROM false_positives fp
#         WHERE fp.min_judges >= 2
#         LIMIT 10000
#         """
        
#         with self.engine.connect() as connection:
#             clone_result = connection.execute(text(clone_query))
#             clone_pairs = [(row[0], row[1], row[2]) for row in clone_result]

#             false_positive_result = connection.execute(text(false_positive_query))
#             false_positives = [(row[0], row[1], row[2]) for row in false_positive_result]
        
#         # Concatenate clone pairs and false positives, and shuffle the dataset
#         data = clone_pairs + false_positives
#         random.shuffle(data)
#         train_data, val_data, test_data = self.solve_data_leakage(data)
#         self.check_data_leakage(train_data=train_data, val_data=val_data, test_data=test_data)
#         return train_data, val_data, test_data
    
#     def solve_data_leakage(self, data):
#         # Create a set of unique IDs
#         unique_ids = set()
#         for row in data:
#             unique_ids.add(row[0])
#             unique_ids.add(row[1])

#         unique_ids = list(unique_ids)
        
#         train_size = int(len(unique_ids) * self.train_ratio)
#         val_size = int(len(unique_ids) * self.val_ratio)

#         # Randomly select unique IDs for each set
#         train_ids = set(random.sample(unique_ids, train_size))
#         val_ids = set(random.sample([id for id in unique_ids if id not in train_ids], val_size))
#         test_ids = set(id for id in unique_ids if id not in train_ids and id not in val_ids)

#         # Filter data to ensure IDs are not in multiple sets
#         train_data, val_data, test_data = [], [], []

#         for row in data:
#             id1, id2 = row[0], row[1]

#             if id1 in train_ids and id2 in train_ids:
#                 train_data.append(row)
#             elif id1 in val_ids and id2 in val_ids:
#                 val_data.append(row)
#             elif id1 in test_ids and id2 in test_ids:
#                 test_data.append(row)
        
#         # Check the balance of clones and non-clones
#         clones_count, non_clones_count = 0, 0
#         for d in (train_data, val_data, test_data):
#             t, n = self.count_clones(d)
#             clones_count += t
#             non_clones_count += n

#         print(f"Samples in:\nTrain = {len(train_data)}\nValidation = {len(val_data)}\nTest = {len(test_data)}")
#         print(f"Clone count: {clones_count}, Non-clone count: {non_clones_count}")

#         return train_data, val_data, test_data
    
#     def count_clones(self, data):
#         clones_count = 0
#         non_clones_count = 0
#         for row in data:
#             if row[2] == 1:
#                 clones_count += row[2]  
#             else:
#                 non_clones_count += 1
#         return clones_count, non_clones_count

    

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

