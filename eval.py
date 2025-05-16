import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from Input_Layer import VitInputLayer
from MHSA_Layer import MultiHeadSelfAttention
from Encoder_Brock import VitEncoderBlock
from Vit import Vit

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# 利用可能なアクセラレーターの確認
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


# # モデルの読み込み
model = Vit().to(device)
model.load_state_dict(torch.load("model.pth", weights_only=True))

# モデルの予測
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# 推論モードに切り替え
model.eval()
for i in range(10):
    x, y = test_data[i][0], test_data[i][1]
    with torch.no_grad():
        # バッチ次元 (dim=0) を追加して 4D テンソルにする
        x = x.unsqueeze(0).to(device)   # x.shape == [1, 1, 28, 28]
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')