from data_loader.BigCloneBenchDataset import CodeCloneDataset, get_dataloaders
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, TrainingArguments
from sqlalchemy import create_engine
from tqdm import tqdm
from typing import List, Tuple, Dict
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score


class CodeCloneModel(nn.Module):
    def __init__(self, model_name: str = "unsloth/Llama-3.2-1B-Instruct-GGUF"):
        super(CodeCloneModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)  

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_labels = []
    all_outputs = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.squeeze(), labels)
            total_loss += loss.item()

            all_labels.extend(labels.tolist())
            all_outputs.extend(torch.sigmoid(outputs).squeeze().tolist())

    avg_loss = total_loss / len(data_loader)
    f1 = f1_score(all_labels, [1 if x >= 0.5 else 0 for x in all_outputs])
    accuracy = accuracy_score(all_labels, [1 if x >= 0.5 else 0 for x in all_outputs])
    recall = recall_score(all_labels, [1 if x >= 0.5 else 0 for x in all_outputs])
    precision = precision_score(all_labels, [1 if x >= 0.5 else 0 for x in all_outputs])

    return avg_loss, f1, accuracy, recall, precision


def train_model(model, train_loader, val_loader, test_loader, optimizer, criterion, device, num_epochs=5):
    model.train()
    training_history = []

    for epoch in range(num_epochs):
        total_train_loss = 0
        print(f"Epoch {epoch + 1}:")

        for batch in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        val_loss, val_f1, val_accuracy, val_recall, val_precision = evaluate_model(model, val_loader, criterion, device)

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Training Loss: {avg_train_loss:.4f}, "
              f"Validation Loss: {val_loss:.4f}, "
              f"Validation F1: {val_f1:.4f}, "
              f"Validation Accuracy: {val_accuracy:.4f}, "
              f"Validation Recall: {val_recall:.4f}, "
              f"Validation Precision: {val_precision:.4f}")

        training_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'val_f1': val_f1,
            'val_accuracy': val_accuracy,
            'val_recall': val_recall,
            'val_precision': val_precision
        })

    test_loss, test_f1, test_accuracy, test_recall, test_precision = evaluate_model(model, test_loader, criterion, device)
    print("\nTest set metrics:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test Precision: {test_precision:.4f}")

    return training_history


def main():
    engine = create_engine("postgresql+psycopg2://postgres:123@localhost/bigclonebench")
    
    train_loader, val_loader, test_loader = get_dataloaders(
        engine,
        batch_size=4,
        train_ratio=0.8,
        val_ratio=0.1,
        tokenizer_name="unsloth/Llama-3.2-1B-Instruct-GGUF",
        max_length=4096,
        num_workers=0
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CodeCloneModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    training_history = train_model(
        model,
        train_loader,
        val_loader,
        test_loader,
        optimizer,
        criterion,
        device,
        num_epochs=5
    )

    for epoch_data in training_history:
        print(f"Epoch {epoch_data['epoch']}: "
              f"Training Loss: {epoch_data['train_loss']:.4f}, "
              f"Validation Loss: {epoch_data['val_loss']:.4f}, "
              f"Validation F1: {epoch_data['val_f1']:.4f}, "
              f"Validation Accuracy: {epoch_data['val_accuracy']:.4f}, "
              f"Validation Recall: {epoch_data['val_recall']:.4f}, "
              f"Validation Precision: {epoch_data['val_precision']:.4f}")

if __name__ == "__main__":
    main()
