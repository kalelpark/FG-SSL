from typing import List
from torch.utils.data import DataLoader

from .isiceight import *
from .isicseven import *
from .aptos import *

def get_loader(args):
    if args.dataset in ["isic2017", "ISIC-2017", "ISIC2017"]:
        return isicseven_loader(args)      
    elif args.dataset in ["isic2018", "ISIC-2018", "ISIC2018"]:
        return isiceight_loader(args)      
    elif args.dataset in ["aptos", "APTOS", "eyes"]:
        return aptos_moco_loader(args)
