import torch

# 1. Check if a GPU (which PyTorch usually calls 'cuda' even for ROCm) is available
if torch.cuda.is_available():
    print(f"current gpu device count: {torch.cuda.device_count()}")
    print(f"current gpu device name: {torch.cuda.get_device_name(0)}")

    # 2. Try to put a tensor on the gpu device
    device = torch.device("cuda:0")
    x = torch.rand(3, 3).to(device)
    y = torch.rand(3, 3).to(device)
    result = x + y
    print(f"result device: {result.device}")

else:
    print("awww, no GPU available or pytorch can't find it. -_-")
    print("make sure your rocm installation in the container is good!")
