import torch
import torch.nn as nn

from models.cnn1d import DroneCNN

from training.config import (
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    MODEL_PATH,
    DATASET_PATH,
    BATCH_SIZE,
    NUM_WORKERS,
)

from training.data_loader import create_dataloaders
from training.train import train_one_epoch
from training.validate import validate
from training.check_points import save_checkpoint
from training.plots import plot_loss_curve

def main():
    """Train the RF drone classifier."""

    train_loader, validation_loader, _ = create_dataloaders(
        dataset_path=DATASET_PATH,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    model = DroneCNN().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_validation_loss = float("inf")

    train_losses = []
    validation_losses = []

    for epoch in range(EPOCHS):

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
            f"Epoch [{epoch + 1}/{EPOCHS}] "
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
                path=MODEL_PATH,
            )

            print("✓ Best model saved.")


    plot_loss_curve(train_losses, validation_losses, output_path="docs/images/loss_curve.png")

if __name__ == "__main__":
    main()