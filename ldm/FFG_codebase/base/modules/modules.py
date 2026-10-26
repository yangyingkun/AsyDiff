"""
Original code: https://github.com/NVlabs/FUNIT/blob/master/trainer.py
"""

import torch.nn as nn

def spectral_norm(module):
    """ init & apply spectral norm """
    nn.init.xavier_uniform_(module.weight, 2 ** 0.5)
    if hasattr(module, 'bias') and module.bias is not None:
        module.bias.data.zero_()

    return nn.utils.spectral_norm(module)
