import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from torchvision.io.image import read_image


class EncoderCNN(nn.Module):
    def __init__(self, embed_size: int) -> None:
        super().__init__()

        # Load a pretrained ResNet-50 model with best weights (IMAGENET1K_V2)
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Freeze the parameters of ResNet-50 to prevent them from being updated during training
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Capture the input features of the original fully connected layer
        in_features = self.resnet.fc.in_features

        # Replace the last fully connected layer with nn.Identity().
        # This effectively removes the classification layer, allowing the model to output raw feature maps instead of class scores.
        # By using nn.Identity(), we maintain the model's structure and avoid any dimensionality issues, enabling the extracted features to be passed
        # directly to the subsequent embedding layer without any modification.
        self.resnet.fc = nn.Identity()

        # Add a linear layer to project the ResNet-50 features into a lower-dimensional space (embedding space)
        self.embed = nn.Linear(in_features, embed_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract the features from the input image, flatten them, and project to embedding space
        return self.embed(self.resnet(x).flatten(1))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_path = "./data/sample.jpg"
    image = read_image(image_path)

    # Define the transformations to apply to the image
    preprocess = ResNet50_Weights.DEFAULT.transforms()

    print(preprocess)

    # Preprocess the image
    image = preprocess(image).unsqueeze(0)

    # Initialize the encoder model
    embed_size = 256
    encoder = EncoderCNN(embed_size).to(device)

    # Set encoder to evaluation mode
    encoder.eval()

    # Move the image to the target device
    image_tensor = image.to(device)

    # Forward pass the image through the encoder
    output = image_tensor
    for name, module in encoder.resnet.named_children():
        output = module(output)
        print(f"{name}: {output.shape}")

    output = output.flatten(1)
    output = encoder.embed(output)
    print(f"Embedding: {output.shape}")
