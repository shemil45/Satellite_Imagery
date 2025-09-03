import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder Part (Down-sampling)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Decoder Part (Up-sampling)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        # The part at the bottom of the 'U'
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        # Final convolution to produce the output mask
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Follow the input down the encoder path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1] # Reverse for the decoder path

        # Follow up the decoder path, using skip connections
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            # If input image size is not divisible by 16, there might be a shape mismatch
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        x = self.final_conv(x)
        # Apply sigmoid activation to get probabilities between 0 and 1
        return torch.sigmoid(x)


# --- A quick test to verify the model's output shape ---
def test():
    # Create a dummy input tensor with shape [Batch Size, Channels, Height, Width]
    x = torch.randn((4, 3, 256, 256))
    model = UNET(in_channels=3, out_channels=1)
    preds = model(x)
    
    print("--- Model Sanity Check ---")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {preds.shape}")
    # The output shape must match the input height and width
    assert preds.shape == (4, 1, 256, 256)
    print("Success! The model output shape is correct.")


if __name__ == "__main__":
    test()
