import torch


def load_dict(model, key, path):
    ckpt = torch.load(path, map_location='cpu')
    state_dict = ckpt['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if key + '.' in k:
            k = str(k)
            key = str(key)
            index = k.find(key)
            new_key = k[index+len(key+'.'):]
            new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)

