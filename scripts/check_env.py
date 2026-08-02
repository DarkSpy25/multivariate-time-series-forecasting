import torch, platform, sys

cude_available = torch.cuda.is_available()
mps_available = torch.backends.mps.is_available()

print("python  ", sys.version.split()[0])
print("torch   ", torch.__version__)
print("machine ", platform.machine())
print("cude available?: ", cude_available)
print("mps available?: ", mps_available)

if cude_available:
    print("GPU: ", torch.cuda.get_device_name(0))

if mps_available:
    x = torch.randn(2000, 2000, device="mps")
    print("matmul  ", float((x @ x).sum()))
    print("GPU(mps) is working.")
else:
    print("No MPS. You'll run on CPU(slower)")

#  make MPS fall back gracefully for unsupported operations (like torch.fft.fft2)
# echo 'export PYTORCH_ENABLE_MPS_FALLBACK=1' >> ~/.zshrc
# source ~/.zshrc