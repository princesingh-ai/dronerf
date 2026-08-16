import torch
import torch.nn as nn
import json
from pathlib import Path

from models.cnn1d_multiclass import MultiClassDroneCNN

from utils.config import DEVICE, config

from training.data_loader import create_dataloaders
from training.train import train_one_epoch
from training.validate import validate
from training.check_points import save_checkpoint
from training.plots import plot_loss_curve


# We dynamically calculate the number of classes from our persistent mapping file.
# This makes the pipeline extremely robust—if we add more drone datasets later, 
# the architecture automatically scales without touching this code!
def get_num_classes():
    mapping_path = Path(config["dataset"]["mapping_path"])
    if not mapping_path.exists():
        raise FileNotFoundError(f"Missing {mapping_path}. Run save.py first!")
    with open(mapping_path, "r") as f:
        mapping = json.load(f)
    return len(mapping)


def main():
    """Train the multi-class RF drone classifier."""

    train_loader, validation_loader, _ = create_dataloaders(
        dataset_path=config["dataset"]["processed_path"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
    )

    num_classes = get_num_classes()
    print(f"Initializing MultiClassDroneCNN with {num_classes} classes...")

    # Instantiate the new multi-class model
    model = MultiClassDroneCNN(num_classes=num_classes).to(DEVICE)
    
    # We switch to CrossEntropyLoss, which expects logits and integer targets
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
    )

    best_validation_loss = float("inf")

    train_losses = []
    validation_losses = []

    for epoch in range(config["training"]["epochs"]):

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        validation_loss = validate(
            model,
            validation_loader,
            criterion,
        )
        
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

        print(
            f"Epoch [{epoch + 1}/{config['training']['epochs']}] "
            f"Train Loss: {train_loss:.4f} "
            f"Validation Loss: {validation_loss:.4f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                loss=validation_loss,
                path=config["training"]["model_path"],
            )

            print("✓ Best model saved.")


    plot_loss_curve(train_losses, validation_losses, output_path="docs/images/loss_curve.png")

if __name__ == "__main__":
    main()