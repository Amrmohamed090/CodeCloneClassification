from softwarereengineering.siameseCC.data_loader import BigCloneBenchDataset
from sqlalchemy import create_engine

def main():
    # Set up the SQLAlchemy engine for your PostgreSQL database
    engine = create_engine("postgresql+psycopg2://postgres:123@localhost/bigclonebench")

    # Get the dataloaders
    train_loader, val_loader, test_loader = BigCloneBenchDataset.get_dataloaders(
        engine,
        batch_size=32,
        train_ratio=0.8,
        val_ratio=0.1,
            num_workers=0,  # Disable multiprocessing

    )

    def inspect_dataloader(dataloader, name, num_batches=10):
        if dataloader is None:
            return
        print(f"Inspecting {name} Dataloader:")
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            print(f"Batch {i+1}:")
            print("Input IDs 1:", batch['input_ids1'][0])  # Print first example's input_ids1
            print("Attention Mask 1:", batch['attention_mask1'][0])  # Print first example's attention_mask1
            print("Input IDs 2:", batch['input_ids2'][0])  # Print first example's input_ids2
            print("Attention Mask 2:", batch['attention_mask2'][0])  # Print first example's attention_mask2
            print("Label:", batch['label'][0])  # Print first example's label
            print("-" * 40)

    # Inspect each dataloader
    inspect_dataloader(train_loader, "Train Loader")
    inspect_dataloader(val_loader, "Validation Loader")
    inspect_dataloader(test_loader, "Test Loader")

if __name__ == '__main__':
    main()
