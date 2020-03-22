import pytorch_lightning as pl
from torchvision import models
from torch import nn
from nt_xent import NT_Xent
import torch
from pytorch_lamb import Lamb
import math
from torch.optim.lr_scheduler import LambdaLR


class SimCLR(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.encoder = getattr(models, hparams.model)(
            pretrained=hparams.from_pretrained
        )

        self.n_features = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        self.embedder = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, self.hparams.embedding_size, bias=False),
        )

        mask = mask_correlated_samples(hparams)
        self.criterion = NT_Xent(hparams.batch_size, hparams.temperature, mask)

    def forward(self, x):
        h = self.encoder(x)
        z = self.embedder(h)

        if self.hparams.normalize:
            z = nn.functional.normalize(z, dim=1)

        return h, z

    def _encode_batch(self, batch):
        (x_i, x_j), _ = batch

        h_i, z_i = self.forward(x_i)
        h_j, z_j = self.forward(x_j)

        loss = self.criterion(z_i, z_j)
        return loss

    def training_step(self, batch, idx):
        loss = self._encode_batch(batch)
        tensorboard_logs = {"train_loss": loss}

        # log learning rate.
        scheduler = self.trainer.lr_schedulers[0]["scheduler"]
        for i, lr in enumerate(scheduler.get_lr()):
            tensorboard_logs[f"lr_{i}"] = lr

        return {"loss": loss, "tensorboard_logs": tensorboard_logs}

    def validation_step(self, batch, idx):
        loss = self._encode_batch(batch)
        tensorboard_logs = {"val_loss": loss}
        return {"val_loss": loss, "tensorboard_logs": tensorboard_logs}

    def validation_epoch_end(self, outputs):
        mean_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        return {"val_loss": mean_val_loss}

    def configure_optimizers(self):
        no_decay = ["batch_normalization", "bias"]
        weight_decay = 10e-6
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        learning_rate = 0.3 * self.hparams.batch_size / 256
        optim = Lamb(optimizer_grouped_parameters, lr=learning_rate)

        total_steps = int(self.hparams.max_epochs * self.hparams.batch_size)
        warmup_steps = int(0.1 * total_steps)

        lr_scheduler = get_cosine_schedule_with_warmup(optim, warmup_steps, total_steps)

        scheduler_config = {"scheduler": lr_scheduler, "interval": "step"}

        return [optim], [scheduler_config]

    def add_model_specifc_args(parser):
        parser.add_argument(
            "--model", type=str, default="resnet18", choices=["resnet18", "resnet50"]
        )
        parser.add_argument("--from_pretrained", action="store_true")
        parser.add_argument("--embedding_size", type=int, default=128)
        parser.add_argument("--normalize", action="store_true")
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--temperature", type=float, default=0.5)
        return parser


def mask_correlated_samples(args):
    mask = torch.ones((args.batch_size * 2, args.batch_size * 2), dtype=bool)
    mask = mask.fill_diagonal_(0)
    for i in range(args.batch_size):
        mask[i, args.batch_size + i] = 0
        mask[args.batch_size + i, i] = 0
    return mask


def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1
):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.

    From github.com/huggingface/transformers
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)
