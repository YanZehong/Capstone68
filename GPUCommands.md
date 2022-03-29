# Commands

### View the graphics card information

'''
nvidia-smi
'''

### Typical setting up
'''
sudo apt update
sudo apt install build-essential
sudo apt install zip
sudo apt install python3.8
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.11.0-Linux-x86_64.sh
bash Miniconda3-py38_4.11.0-Linux-x86_64.sh
bash
pip install jupyter torch torchvision d2l zehong
'''

### Map remote jupyter notebook to local
'''
jupyter notebook
ssh -L8888:localhost:8888 zehong@172.27.82.2
'''

### Download file from cloud
- Right-click on the file you are interested in download (from web interface), and choose Embed.

- Press "Generate HTML code to embed this file".

- Copy the part contained in the "" of src is your link.

- Replace 'embed' with 'download'

'''
wget --no-check-certificate "https://onedrive.live.com/download?cid=82847715CFF24FC2&resid=82847715CFF24FC2%21655&authkey=AJKEstD7ElbnHVU"
'''

### Computing Devices

Install the PyTorch version that supports CUDA 10.0 via:
'''
pip install torch-cu100.
'''

In PyTorch, the CPU and GPU can be indicated by torch.device('cpu') and torch.device('cuda').
'''
import torch
from torch import nn
torch.device('cpu')
torch.device('cuda') # default 0
torch.device('cuda:1')
torch.cuda.device_count()
'''

Two convenient functions that allow us to run code even if the requested GPUs do not exist.
'''
def try_gpu(i=0):  #@save
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():  #@save
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

try_gpu(), try_gpu(10), try_all_gpus()
'''
