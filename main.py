import argparse
import json
from pathlib import Path

import torch

from data_loader import data_loader
from model import Model
from train import train, save
from utils import seed_everything, wd_groupping


CONFIG_DIR = Path(__file__).resolve().parent / "configs"
SUPPORTED_DATASETS = ("normal", "exp", "mix", "copula", "cortex", "VerbAgg")
SUPPORTED_FORWARDS = ("VAE", "IWAE", "IWRCL")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train VAE/IWAE/IWRCL with dataset-specific fixed settings."
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU number used when device is CUDA (default: 0).",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="exp",
        help="Dataset: normal, exp, mix, copula, cortex, VerbAgg (default: exp).",
    )
    parser.add_argument(
        "--forward",
        type=str,
        default="IWRCL",
        help="Forward/training method: VAE, IWAE, IWRCL (default: IWRCL).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of latent/importance samples M (default: 100).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cpu, cuda, cuda:N, mps, ... (default: auto).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    return parser.parse_args()


def normalize_dataset_name(name: str) -> str:
    lookup = {dataset.lower(): dataset for dataset in SUPPORTED_DATASETS}
    key = name.strip().lower()
    if key not in lookup:
        raise ValueError(
            f"Unsupported dataset '{name}'. Choose from: {', '.join(SUPPORTED_DATASETS)}"
        )
    return lookup[key]


def normalize_forward_name(name: str) -> str:
    lookup = {method.lower(): method for method in SUPPORTED_FORWARDS}
    key = name.strip().lower()
    if key not in lookup:
        raise ValueError(
            f"Unsupported forward method '{name}'. "
            f"Choose from: {', '.join(SUPPORTED_FORWARDS)}"
        )
    return lookup[key]


def resolve_device(device_arg: str, gpu: int) -> torch.device:
    requested = device_arg.strip().lower()

    if requested == "auto":
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{gpu}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    elif requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available.")
        device = torch.device(f"cuda:{gpu}")

    else:
        device = torch.device(requested)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"--device {device_arg} was requested, but CUDA is not available."
                )
            if device.index is None:
                device = torch.device(f"cuda:{gpu}")

    if device.type == "cuda":
        n_gpu = torch.cuda.device_count()
        if device.index is None or device.index < 0 or device.index >= n_gpu:
            raise ValueError(
                f"Invalid CUDA device {device}. Available CUDA device count: {n_gpu}."
            )
        torch.cuda.set_device(device)

    return device


def load_dataset_config(data_name: str) -> dict:
    config_path = CONFIG_DIR / f"{data_name}.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    required_keys = {
        "x_dim",
        "h_dims",
        "z_dim",
        "u_dim",
        "LayerNorm",
        "SkipConnection_encoder",
        "SkipConnection_decoder",
        "batch_size",
        "epochs",
        "kl_weight",
        "learning_rate",
        "weight_decay",
    }
    missing = required_keys.difference(config)
    if missing:
        raise KeyError(
            f"Missing keys in {config_path.name}: {', '.join(sorted(missing))}"
        )

    return config


def build_save_path(data_name: str, forward: str, num_samples: int, seed: int) -> Path:
    return Path("./save") / data_name / forward / f"M{num_samples}" / f"seed{seed}"


def main():
    args = parse_args()

    data_name = normalize_dataset_name(args.data)
    forward = normalize_forward_name(args.forward)

    if data_name == "cortex" and forward == "VAE":
        raise ValueError(
            "VAE is not supported for the Cortex experiment. "
            "Please use IWAE or IWRCL."
        )

    if args.num_samples <= 0:
        raise ValueError("--num_samples must be a positive integer.")
    if args.gpu < 0:
        raise ValueError("--gpu must be a non-negative integer.")

    device = resolve_device(args.device, args.gpu)
    config = load_dataset_config(data_name)

    print("=" * 72)
    print(f"dataset       : {data_name}")
    print(f"forward       : {forward}")
    print(f"num_samples   : {args.num_samples}")
    print(f"seed          : {args.seed}")
    print(f"device        : {device}")
    print(f"config        : {CONFIG_DIR / f'{data_name}.json'}")
    print("=" * 72)

    # 1. Fix random seed.
    seed_everything(args.seed)

    # 2. Load data using the fixed batch size for the selected dataset.
    x, u, train_loader = data_loader(
        data_name=data_name,
        batch_size=config["batch_size"],
    )

    # 3. Build model. Runtime-selectable settings are added here;
    #    all remaining settings come only from the dataset config.
    model_kwargs = dict(config)
    model_kwargs.update(
        {
            "data_name": data_name,
            "model": forward,
            "num_samples": args.num_samples,
            "device": device,
            "seed": args.seed,
        }
    )

    model = Model(
        std=train_loader.std,
        **model_kwargs,
    ).to(device)

    # 4. Optimizer.
    optimizer = torch.optim.Adam(
        wd_groupping(model),
        lr=model.learning_rate,
    )

    # 5. Train.
    model = train(train_loader, model, optimizer)

    # 6. Save.
    save_path = build_save_path(
        data_name=data_name,
        forward=forward,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    save(model, train_loader, path=str(save_path))

    print(f"Saved to: {save_path.resolve()}")


if __name__ == "__main__":
    main()
