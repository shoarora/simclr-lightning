from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torchvision

from model import SimCLR
from transforms import TransformsSimCLR

from datasets import ImagePathsDataset


def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--gpus", type=eval, default=0)
    parser.add_argument("--num_tpu_cores", type=int, default=None)
    parser.add_argument("--distributed_backend", type=str, default=None)

    parser.add_argument(
        "--dataset_type",
        type=str,
        default="cifar10",
        choices=["cifar10", "folder", "paths"],
    )
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    return parser


def get_dataloaders(args):
    if args.dataset_type == "cifar10":
        train = torchvision.datasets.CIFAR10(
            args.dataset_path, train=True, download=True, transform=TransformsSimCLR()
        )
    elif args.dataset_type == "folder":
        train = torchvision.datasets.ImageFolder(
            args.dataset_path, transform=TransformsSimCLR()
        )
    elif args.dataset_type == "paths":
        train = ImagePathsDataset(args.dataset_path, transforms=TransformsSimCLR())

    loader = torch.utils.data.DataLoader(
        train, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True
    )
    return loader


def main():
    parser = get_parser()
    parser = SimCLR.add_model_specifc_args(parser)
    args = parser.parse_args()
    print(args)
    if args.num_tpu_cores:
        args.gpus = None

    model = SimCLR(args)
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        gpus=args.gpus,
        distributed_backend=args.distributed_backend,
        num_tpu_cores=args.num_tpu_cores,
        val_check_percent=0.01,
    )

    if not args.num_tpu_cores:
        args.num_tpu_cores = 0
    if not args.gpus:
        args.gpus = 0
    if not args.distributed_backend:
        del args.distributed_backend

    train_loader = get_dataloaders(args)

    trainer.fit(model, train_loader, train_loader)


if __name__ == "__main__":
    main()
