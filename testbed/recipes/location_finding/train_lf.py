import os
import argparse
from omegaconf import OmegaConf
import torch


from testbed.tasks import LocationFinding
from testbed.design_objectives import SPCE

from testbed.designers import PolicyDesigner
from testbed.design_nets import *
from testbed.recipes.train import Trainer
from testbed.core.manager import Manager

def build_policy_net(cfg):
    POLICY_NETS = {
        "DADNet": DADNet,
        "DADNet_v2": DADNet_v2,
        "DADNet_v3": DADNet_v3,
        "AttentiveDAD": AttentiveDAD,
        "AttentiveDAD_v2": AttentiveDAD_v2,
        "AttentiveDAD_v3": AttentiveDAD_v3,
        "AttentiveDAD_v4": AttentiveDAD_v4,
    }
    
    net_cls = POLICY_NETS[cfg.policy_net_name]
    net = net_cls()   # later you can pass args via cfg too
    return net


if __name__ == '__main__':
    conf_cli = OmegaConf.from_cli()
    cfg = OmegaConf.load(conf_cli.config)
    # sync cli args with config
    cfg = OmegaConf.merge(cfg, conf_cli)

    # Manager setup
    run_name = f"{os.environ.get('SLURM_ARRAY_JOB_ID', '0')}_{os.environ.get('SLURM_ARRAY_TASK_ID', '0')}"
    manager = Manager(cfg, root=cfg.save_dir, name=run_name, timestamp=False)

    # Set seed
    torch.manual_seed(cfg.seed)

    task = LocationFinding(cfg.n_sources, cfg.dim_theta, device=cfg.device)

    policy_net = build_policy_net(cfg).to(cfg.device)
    designer = PolicyDesigner(task, cfg.T, policy_net)

    objective = SPCE(task, n_contrastive=cfg.n_inner)

    optimizer = torch.optim.Adam(
        policy_net.parameters(),
        lr=cfg.lr,
    )

    trainer = Trainer(
        task=task,
        designer=designer,
        objective=objective,
        optimizer=optimizer,
        cfg=cfg,
        manager=manager
    )

    trainer.train()
    