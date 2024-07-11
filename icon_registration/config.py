import torch

if torch.cuda.is_available():
    device = torch.device("cuda:2")
else:
    device = torch.device("cpu")
