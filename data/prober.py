try:
    from datasets import load_dataset
    import torch.nn as nn
    import torch 
except:
    import os
    os.system('pip install datasets')

