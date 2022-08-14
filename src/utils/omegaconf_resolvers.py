from git import Repo
from omegaconf import OmegaConf


def _exp_gamma_finder(start_lr, end_lr, steps):
    import numpy as np
    gamma = np.power(end_lr / start_lr, 1 / steps)
    return float(gamma)

def _const_after_cosine_finder(start_lr, eta_min):
    factor = eta_min / start_lr
    return float(factor)

def _onecycle_warmup_epoches(epochs, warmup_epoches):
    return warmup_epoches / epochs

def _invert(value):
    return 1 / value

def _get_one_cycle_final_div_factor(div_factor, final_div_lr):
    return final_div_lr / div_factor  # We should divide by div_factor to get 1 lr


def _get_git_commit(repo_dir, num_values=7):
    repo = Repo(repo_dir)
    hcommit = repo.head.commit
    return str(hcommit)[:num_values]

def register_resolvers():
    OmegaConf.register_new_resolver(
                'exp_gamma',
                _exp_gamma_finder,
                use_cache=False
            )

    OmegaConf.register_new_resolver(
        'const_after_cosine_finder',
        _const_after_cosine_finder,
        use_cache=False
    )

    OmegaConf.register_new_resolver(
        'onecycle_warmup_epoches',
        _onecycle_warmup_epoches,
        use_cache=False
    )
    OmegaConf.register_new_resolver(
        'invert',
        _invert,
        use_cache=False
    ),
    OmegaConf.register_new_resolver(
        'get_one_cycle_final_div_factor',
        _get_one_cycle_final_div_factor,
        use_cache=False
    )

    OmegaConf.register_new_resolver(
        'git_commit',
        _get_git_commit,
        use_cache=False
    )