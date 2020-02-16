import numpy as np
import random
import torch
from datetime import datetime


def print_time(msg: str):
    assert isinstance(msg, str), 'Argument "msg" must be a string.'
    now = datetime.now()
    current_time = now.strftime("%A, %B %d %Y at %I:%M%p")
    print('\n%s: %s\n' % (msg ,current_time))


def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



