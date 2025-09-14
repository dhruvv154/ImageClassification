import torch, platform
print("Python:", platform.python_version())
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))
    # tiny test
    x = torch.randn(1024, 1024, device="cuda")
    y = torch.mm(x, x.t())
    print("Matmul ok, mean:", y.mean().item())
else:
    print("Running on CPU. If you have an NVIDIA GPU, ensure the correct CUDA wheel is installed and drivers are up-to-date.")
