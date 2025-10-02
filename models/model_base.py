"""

Do not Modify
-Connor

"""

import os
import torch
import torch.nn as nn
from utils.utils_bnorm import merge_bn, tidy_sequential
from torch.nn.parallel import DataParallel, DistributedDataParallel


class ModelBase:
    def __init__(self, opt):
        self.opt = opt  # opt
        self.save_dir = opt["path"]["models"]  # save models
        self.device = torch.device("cuda" if opt["gpu_ids"] is not None else "cpu")
        self.is_train = opt["is_train"]  # training or not
        self.schedulers = []  # schedulers

    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    def init_train(self):
        pass

    def load(self):
        pass

    def save(self, label):
        pass

    def define_loss(self):
        pass

    def define_optimizer(self):
        pass

    def define_scheduler(self):
        pass

    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def current_visuals(self):
        pass

    def current_losses(self):
        pass

    def update_learning_rate(self, n):
        for scheduler in self.schedulers:
            scheduler.step(n)

    def current_learning_rate(self):
        return self.schedulers[0].get_lr()[0]

    def requires_grad(self, model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    """
    # ----------------------------------------
    # Information of net
    # ----------------------------------------
    """

    def print_network(self):
        pass

    def info_network(self):
        pass

    def print_params(self):
        pass

    def info_params(self):
        pass

    def get_bare_model(self, network):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(network, (DataParallel, DistributedDataParallel)):
            network = network.module
        return network

    def model_to_device(self, network):
        """Model to device. It also warps models with DistributedDataParallel
        or DataParallel.
        Args:
            network (nn.Module)
        """
        network = network.to(self.device)
        if self.opt["dist"]:
            find_unused_parameters = self.opt.get("find_unused_parameters", True)
            use_static_graph = self.opt.get("use_static_graph", False)
            network = DistributedDataParallel(
                network,
                device_ids=[torch.cuda.current_device()],
                find_unused_parameters=find_unused_parameters,
            )
            if use_static_graph:
                print(
                    'Using static graph. Make sure that "unused parameters" will not change during training loop.'
                )
                network._set_static_graph()
        else:
            network = DataParallel(network)
        return network

    # ----------------------------------------
    # network name and number of parameters
    # ----------------------------------------
    def describe_network(self, network):
        network = self.get_bare_model(network)
        msg = "\n"
        msg += "Networks name: {}".format(network.__class__.__name__) + "\n"
        msg += (
            "Params number: {}".format(
                sum(map(lambda x: x.numel(), network.parameters()))
            )
            + "\n"
        )
        msg += "Net structure:\n{}".format(str(network)) + "\n"
        return msg

    # ----------------------------------------
    # parameters description
    # ----------------------------------------
    def describe_params(self, network):
        network = self.get_bare_model(network)
        msg = "\n"
        msg += (
            " | {:^6s} | {:^6s} | {:^6s} | {:^6s} || {:<20s}".format(
                "mean", "min", "max", "std", "shape", "param_name"
            )
            + "\n"
        )
        for name, param in network.state_dict().items():
            if not "num_batches_tracked" in name:
                v = param.data.clone().float()
                msg += (
                    " | {:>6.3f} | {:>6.3f} | {:>6.3f} | {:>6.3f} | {} || {:s}".format(
                        v.mean(), v.min(), v.max(), v.std(), v.shape, name
                    )
                    + "\n"
                )
        return msg

    """
    # ----------------------------------------
    # Save prameters
    # Load prameters
    # ----------------------------------------
    """

    # ----------------------------------------
    # save the state_dict of the network
    # ----------------------------------------
    def save_network(self, save_dir, network, network_label, iter_label):
        save_filename = "{}_{}.pth".format(iter_label, network_label)
        save_path = os.path.join(save_dir, save_filename)
        network = self.get_bare_model(network)
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    # ----------------------------------------
    # load the state_dict of the network
    # ----------------------------------------
    def load_network(self, load_path, network, strict=True, param_key="params"):
        network = self.get_bare_model(network)
        if strict:
            state_dict = torch.load(load_path)
            if param_key in state_dict.keys():
                state_dict = state_dict[param_key]
            network.load_state_dict(state_dict, strict=strict)
        else:
            # Flexible loading: load matching parameters, initialize the rest
            checkpoint = torch.load(load_path)
            if param_key in checkpoint.keys():
                checkpoint = checkpoint[param_key]

            model_dict = network.state_dict()

            # Filter out parameters that don't match in size
            matched_dict = {}
            mismatched_keys = []
            missing_keys = []

            for k, v in model_dict.items():
                if k in checkpoint:
                    if checkpoint[k].shape == v.shape:
                        matched_dict[k] = checkpoint[k]
                    else:
                        mismatched_keys.append(k)
                        print(
                            f"Size mismatch for {k}: checkpoint {checkpoint[k].shape} vs model {v.shape}, using random initialization"
                        )
                else:
                    missing_keys.append(k)

            # Report what was not in checkpoint
            if missing_keys:
                print(f"\nKeys in model but not in checkpoint ({len(missing_keys)}):")
                for k in missing_keys[:10]:  # Show first 10
                    print(f"  - {k}")
                if len(missing_keys) > 10:
                    print(f"  ... and {len(missing_keys) - 10} more")

            # Update model dict with matched parameters
            model_dict.update(matched_dict)
            network.load_state_dict(model_dict, strict=True)

            print(
                f"\nLoaded {len(matched_dict)}/{len(model_dict)} parameters from checkpoint"
            )
            print(f"Mismatched: {len(mismatched_keys)}, Missing: {len(missing_keys)}")

            del checkpoint, model_dict, matched_dict

    # ----------------------------------------
    # save the state_dict of the optimizer
    # ----------------------------------------
    def save_optimizer(self, save_dir, optimizer, optimizer_label, iter_label):
        save_filename = "{}_{}.pth".format(iter_label, optimizer_label)
        save_path = os.path.join(save_dir, save_filename)
        torch.save(optimizer.state_dict(), save_path)

    # ----------------------------------------
    # load the state_dict of the optimizer
    # ----------------------------------------
    def load_optimizer(self, load_path, optimizer):
        optimizer.load_state_dict(
            torch.load(
                load_path,
                map_location=lambda storage, loc: storage.cuda(
                    torch.cuda.current_device()
                ),
            )
        )

    def update_E(self, decay=0.999):
        netG = self.get_bare_model(self.netG)
        netG_params = dict(netG.named_parameters())
        netE_params = dict(self.netE.named_parameters())
        for k in netG_params.keys():
            netE_params[k].data.mul_(decay).add_(netG_params[k].data, alpha=1 - decay)

    """
    # ----------------------------------------
    # Merge Batch Normalization for training
    # Merge Batch Normalization for testing
    # ----------------------------------------
    """

    # ----------------------------------------
    # merge bn during training
    # ----------------------------------------
    def merge_bnorm_train(self):
        merge_bn(self.netG)
        tidy_sequential(self.netG)
        self.define_optimizer()
        self.define_scheduler()

    # ----------------------------------------
    # merge bn before testing
    # ----------------------------------------
    def merge_bnorm_test(self):
        merge_bn(self.netG)
        tidy_sequential(self.netG)
