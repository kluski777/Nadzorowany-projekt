import torch
from src.models.autoencoder import AutoEncoder


model = AutoEncoder(
    input_channels=3,
    latent_channels=32,
    learning_rate=1e-3,
    loss_type="ssim"
)

test_cases = [
    (1, 3, 256, 256),
    (4, 3, 256, 256),
    (8, 3, 256, 256),
]

model.eval()
with torch.no_grad():
    for shape in test_cases:
        input_tensor = torch.randn(*shape)
        output = model(input_tensor)
        
        assert input_tensor.shape == output.shape, \
            f"Shape mismatch! Input: {input_tensor.shape}, Output: {output.shape}"

print("All tests passed!")
