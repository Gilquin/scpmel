# global import
import torch

def _to_tensor(x, device='cpu'):
    """
    Utility function to broadcast an array to a torch tensor (float 32)
    and move it to the specified device (default to 'cpu').
    """

    if not torch.is_tensor(x):
        x = torch.tensor(
            x, device=device, dtype=torch.get_default_dtype())

    return x
