import os
import random
import numpy as np
import torch

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def wd_groupping(model):
    lst_wd = []
    lst_wd_no = []
    for name, parm in model.named_parameters():
        if 'linear' in name and 'bias' not in name:
            lst_wd.append(parm)
        else:
            lst_wd_no.append(parm)

    params_wd = {'params' : lst_wd, 'weight_decay' : model.weight_decay}

    if (model.SkipConnection_encoder or model.SkipConnection_decoder):
        # in this case we apply weight decay only to linear weights
        params_wd_no = {'params' : lst_wd_no, 'weight_decay' : 0}
    else: 
        params_wd_no = {'params' : lst_wd_no, 'weight_decay' : model.weight_decay}

    return [params_wd, params_wd_no]