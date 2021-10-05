from torch import multiprocessing
from transformers import BertTokenizer, BertForMaskedLM
import torch
import torch_xla
from typing import Union
from transformers import AdamW
import torch_xla.core.xla_model as xm
import torch_xla.test.test_utils as test_utils
import time
import argparse
import os
import torch_xla.distributed.parallel_loader as pl


def _train_update(
    device: Union[xm.xla_device, torch.device],
    step: int,
    loss: torch.Tensor,
    tracker: xm.RateTracker,
    epoch: int,
    writer=None,
):
    """
    This function prints the loss and rate(in samples per sec) per training step.

    Args:
        device (Union[xm.xla_device, torch.device]): The device where these statistics come from
        step (int): Step number for the update
        loss (torch.Tensor): Loss tensor value
        tracker (xm.RateTracker): Rate Tracker for training measurement
        epoch (int): Current epoch number
        summary_writer (SummaryWriter, optional): If provided, this method will
      write some of the provided statistics to Tensorboard.
    """
    test_utils.print_training_update(
        device,
        step,
        loss,
        tracker.rate(),
        tracker.global_rate(),
        epoch,
        summary_writer=writer,
    )


class MeditationsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


def get_dataset_loader(batch_size: int, seq_len: int):
    """
    This function is used to get the dataset loader. It uses the Meditations dataset raw text with the
    given seq len and batch size to generate it.

    Args:
        batch_size (int): Batch size of input data
        seq_len (int): Sequence Length of input data

    Returns:
        loader (torch.utils.data.DataLoader): Returns a torch Dataloader with the appropriate configurations
    """
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    cwd = os.getcwd()
    dataset_path = os.path.join(cwd, "clean.txt")

    os.system(
        f"wget -N https://raw.githubusercontent.com/jamescalam/transformers/main/data/text/meditations/clean.txt && mv clean.txt {dataset_path}"
    )

    with open(dataset_path, "r") as fp:
        text = fp.read().split("\n")

    print(text[:5])

    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=seq_len,
        truncation=True,
        padding="max_length",
    )

    inputs["labels"] = inputs.input_ids.detach().clone()
    print(f"Input Keys:{inputs.keys()}")

    rand = torch.rand(inputs.input_ids.shape)
    # create mask array
    mask_arr = (
        (rand < 0.15)
        * (inputs.input_ids != 101)
        * (inputs.input_ids != 102)
        * (inputs.input_ids != 0)
    )

    selection = []

    for i in range(inputs.input_ids.shape[0]):
        selection.append(torch.flatten(mask_arr[i].nonzero()).tolist())

    for i in range(inputs.input_ids.shape[0]):
        inputs.input_ids[i, selection[i]] = 103

    dataset = MeditationsDataset(inputs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader


def loop_with_amp(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim,
    autocast,
    scaler,
    xla_enabled: bool,
):
    """
    This function runs a training iteration in Automatic Mixed Precision(AMP) mode.

    Args:
        model (torch.nn.Module): PyTorch Model
        input_ids (torch.Tensor): Similar to token indices, numerical representations of tokens building the sequences
        that will be used as input by the model.
        attention_mask (torch.Tensor): This argument indicates to the model which tokens should be attended to,
        and which should not. It is used when batching sequences together.
        labels (torch.Tensor): Expected prediction of the model
        optimizer (torch.optim): PyTorch Optimizer
        autocast (Union[torch_xla.amp, torch.cuda.amp]): For AMP, autocasting
        automatically chooses the precision for GPU operations to improve performance while maintaining accuracy.
        scaler (Union[torch_xla.amp, torch.cuda.amp]): For AMP, this helps perform the steps of gradient scaling conveniently.

    Returns:
        loss(torch.Tensor) : Tensor representing the loss value
        optimizer(torch.optim) : Model Optimizer
    """
    with autocast():
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

    if xla_enabled:
        scaler.scale(loss).backward()
        gradients = xm._fetch_gradients(optimizer)
        xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
        scaler.step(optimizer)
        scaler.update()
        # xm.mark_step()
    else:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return loss, optimizer


def loop_without_amp(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim,
    xla_enabled: bool,
):
    """
    This function runs a training iteration in non-AMP/fp32 mode
    Args:
        model (torch.nn.Module): PyTorch Model
        input_ids (torch.Tensor): Similar to token indices, numerical representations of tokens building the sequences
        that will be used as input by the model.
        attention_mask (torch.Tensor): This argument indicates to the model which tokens should be attended to,
        and which should not. It is used when batching sequences together.
        labels (torch.Tensor): Expected prediction of the model
        optimizer (torch.optim): PyTorch Optimizer
        xla_enabled (bool): Whether xla should be used or not

    Returns:
        loss(torch.Tensor) : Tensor representing the loss value
        optimizer(torch.optim) : Model Optimizer
    """
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    if xla_enabled:
        xm.optimizer_step(optimizer)
    else:
        optimizer.step()
    return loss, optimizer


def get_device(xla_enabled):
    """Get the appropriate device for running the model (CUDA / XLA). This is for single-GPU only
    Args:
        xla_enabled (bool): Indicating whether xla is enabled or not

    Returns:
        Union[xm.xla_device, torch.device]: CUDA / XLA Device
    """
    if xla_enabled:
        return xm.xla_device()
    else:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_model(model_name: str, device: Union[xm.xla_device, torch.device]) -> torch.nn.Module:
    """
    This function gets the pretrained model , loads it on the given device and returns it.
    Args:
        model_name (str): The name of the model
        device (Union[xm.xla_device, torch.device]): The device on which the model should be loaded on

    Returns:
        model (torch.nn.Module): Returns the model loaded on the given device
    """
    model = BertForMaskedLM.from_pretrained(model_name)
    model.to(device)
    model.train()
    return model


def get_autocast_and_scaler(args):
    """This function returns the appropriate autocast and GradScaler depending
    upon amp, xla-flags and syncfree optimizer flags.

    Returns:
        autocast (Union[torch_xla.amp, torch.cuda.amp, None]): For AMP, autocasting
        automatically chooses the precision for GPU operations to improve performance while maintaining accuracy.
        scaler (Union[torch_xla.amp, torch.cuda.amp, None]): For AMP, this helps perform the steps of gradient scaling conveniently.
    """
    autocast, scaler = None, None
    if not args.fp32:
        if args.xla_enabled:
            from torch_xla.amp import autocast, GradScaler, syncfree

            if args.use_syncfree_optimizer:
                return autocast, syncfree.GradScaler()
            return autocast, GradScaler()

        from torch.cuda.amp import autocast, GradScaler

        return autocast, GradScaler()

    return autocast, scaler


def train(
    args: argparse.ArgumentParser,
    loader: torch.utils.data.DataLoader,
    device: Union[xm.xla_device, torch.device],
    model: torch.nn.Module,
    optimizer: torch.optim,
    autocast,
    scaler,
):
    """This function trains the model for the given number of epochs in XLA/native-PT.
    It can also do profiling and other logging depending upon which flags are activated.
    It prints out the time taken by each epoch at the end of each epoch.
    Args:
        args (argparse.ArgumentParser): User-defined arguments
        loader (torch.utils.data.DataLoader): Dataloader for model
        device (Union[xm.xla_device, torch.device]): Device on which the model will run
        model (torch.nn.Module): PyTorch Model
        optimizer (torch.optim): PyTorch optimizer
        autocast (Union[torch_xla.amp, torch.cuda.amp]): For AMP, autocasting
        automatically chooses the precision for GPU operations to improve performance while maintaining accuracy.
        scaler (Union[torch_xla.amp, torch.cuda.amp]): For AMP, this helps perform the steps of gradient scaling conveniently.
        Gradient scaling improves convergence for networks with float16 gradients by minimizing gradient underflow
    """
    if args.xla_tb_profile:
        server = xp.start_server(port_number)
    for epoch in range(args.num_epochs):
        if epoch == 1 and args.xla_tb_profile:
            training_started.set()
        start_time = time.time()
        for step, batch in enumerate(loader):
            # initialize calculated gradients (from prev step)
            optimizer.zero_grad()
            # pull all tensor batches required for training
            tracker = xm.RateTracker()  # Placing the tracker here frees it of I/O time.
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            if not args.xla_enabled:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

            attention_mask = None
            if not args.fp32:
                loss, optimizer = loop_with_amp(
                    model,
                    input_ids,
                    attention_mask,
                    labels,
                    optimizer,
                    autocast,
                    scaler,
                    args.xla_enabled,
                )
            else:
                loss, optimizer = loop_without_amp(
                    model,
                    input_ids,
                    attention_mask,
                    labels,
                    optimizer,
                    args.xla_enabled,
                )

            tracker.add(input_ids.size(0))
            # Skip printing the train update for better performance
            # _train_update(device, step, loss, tracker, epoch, None)

        num_steps = step + 1
        end_time = time.time()
        print("Epoch ", epoch, (end_time - start_time) / num_steps)
        # Skip printing the train update for better performance
        # _train_update(device, step, loss, tracker, epoch, None)


def main(args: argparse.ArgumentParser):
    loader = get_dataset_loader(args.batch_size, args.seq_len)
    device = get_device(args.xla_enabled)
    model = get_model(args.model_name, device)

    if args.use_syncfree_optimizer:
        from torch_xla.amp.syncfree import Adam

        optimizer = Adam(model.parameters(), lr=1e-5)
    else:
        optimizer = AdamW(model.parameters(), lr=1e-5)

    autocast, scaler = get_autocast_and_scaler(args)
    if args.dlprof_enabled and not args.xla_enabled:
        with torch.autograd.profiler.emit_nvtx():
            train(args, loader, device, model, optimizer, autocast, scaler)
    else:
        if args.xla_enabled:
            train(
                args,
                pl.MpDeviceLoader(loader, device),
                device,
                model,
                optimizer,
                autocast,
                scaler,
            )
        else:
            train(args, loader, device, model, optimizer, autocast, scaler)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--xla_enabled", action="store_true", default=False)
    arg_parser.add_argument("--fp32", action="store_true", default=False)
    arg_parser.add_argument(
        "--use_syncfree_optimizer", action="store_true", default=False
    )

    arg_parser.add_argument("--dlprof_enabled", action="store_true", default=False)
    arg_parser.add_argument(
        "--xla_debug_metrics_enabled", action="store_true", default=False
    )
    arg_parser.add_argument("--xla_tb_profile", action="store_true", default=False)
    arg_parser.add_argument("--port_number", default=8192, type=int)
    arg_parser.add_argument("--xla_tb_folder", default="/tmp/bert_tb_profile", type=str)
    arg_parser.add_argument("--model_name", default="bert-base-uncased", type=str)
    arg_parser.add_argument("--batch_size", default=12, type=int)
    arg_parser.add_argument("--seq_len", default=512, type=int)
    arg_parser.add_argument("--num_epochs", default=100, type=int)
    args = arg_parser.parse_args()

    if args.dlprof_enabled and not args.xla_enabled:
        import nvidia_dlprof_pytorch_nvtx

        nvidia_dlprof_pytorch_nvtx.init()

    if args.xla_enabled:
        dataset_path = "/pytorch/xla/test/clean.txt"
        os.system(
            f"wget -N https://raw.githubusercontent.com/jamescalam/transformers/main/data/text/meditations/clean.txt && mv clean.txt {dataset_path}"
        )
        if args.xla_tb_profile:
            import torch_xla.debug.profiler as xp

            port_number = args.port_number
            training_started = multiprocessing.Event()
            p = multiprocessing.Process(target=main, args=())
            p.start()
            training_started.wait()
            xp.trace(f"localhost:{port_number}", args.xla_tb_folder)
        else:
            main(args)
        if args.xla_debug_metrics_enabled:
            import torch_xla.debug.metrics as met

            print(met.metrics_report())
    else:
        dataset_path = "data/clean.txt"
        os.system(
            f"wget -N https://raw.githubusercontent.com/jamescalam/transformers/main/data/text/meditations/clean.txt && mv clean.txt {dataset_path}"
        )
        main(args)
