import torch

# Helper function to get a gpu if one is available
def get_device():
    return torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
