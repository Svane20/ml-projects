from model import VisionTransformer
from data_setup import create_dataloaders
from utils import download_data, set_seeds, save_model
from engine import train

import torch
from torchvision.transforms import transforms

# Hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 5

# Vision Transformer hyperparameters
PATCH_SIZE = 16
EMBEDDING_DIM = 768
NUM_TRANSFORMER_LAYERS = 12

if __name__ == "__main__":
    # Download the data
    source_url = "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"
    destination = "pizza_steak_sushi"
    image_path = download_data(source_url, destination)

    # Setup directory paths to train and test images
    train_dir = image_path / "train"
    test_dir = image_path / "test"

    # Setup target agnostic device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device.")

    # Define the training and test data transforms
    manual_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    # Create data loaders
    train_dataloader, test_dataloader, class_names = create_dataloaders(
        train_dir=str(train_dir),
        test_dir=str(test_dir),
        train_transform=manual_transforms,
        test_transform=manual_transforms,
        batch_size=BATCH_SIZE
    )

    # Set random seeds for reproducibility
    set_seeds(seed=42)

    # Create a Vision Transformer model
    model = VisionTransformer(
        img_size=IMG_SIZE,
        in_channels=3,
        patch_size=PATCH_SIZE,
        num_transformer_layers=NUM_TRANSFORMER_LAYERS,
        embedding_dim=EMBEDDING_DIM,
        mlp_size=3072,
        num_heads=12,
        attn_dropout=0,
        embedding_dropout=0.1,
        num_classes=len(class_names)
    ).to(device)

    # Set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=LEARNING_RATE
    )

    # Train the model
    train(
        model=model,
        train_data_loader=train_dataloader,
        test_data_loader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=NUM_EPOCHS,
        device=device,
    )

    # Save the model
    save_model(
        model=model,
        target_dir="models",
        model_name="vision_transformer.pth",
    )
