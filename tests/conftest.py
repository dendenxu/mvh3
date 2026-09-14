"""Keep CPU regression work within the same thread budget as training."""

import torch

torch.set_num_threads(1)
