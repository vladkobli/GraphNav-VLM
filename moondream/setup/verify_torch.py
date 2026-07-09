import torch

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version used by PyTorch:", torch.version.cuda)
    print("GPU count:", torch.cuda.device_count())
    print("GPU name:", torch.cuda.get_device_name(0))
    x = torch.randn(512, 512, device="cuda")
    y = x @ x
    print("CUDA tensor test:", float(y[0, 0]))
else:
    print("WARNING: CUDA is not available to PyTorch.")
