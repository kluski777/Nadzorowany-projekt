import comet_ml  # noqa: F401 (import comet_ml before pytorch)
import argparse

from utils import load_config
from data.generate_splits import generate_splits
from training import train_autoencoder


def parse_args():
    parser = argparse.ArgumentParser(
        description="WikiArt AutoEncoder Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    parser_generate = subparsers.add_parser(
        "generate_splits",
        help="Generate train/val/test splits"
    )
    parser_generate.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file (default: config.yaml)",
    )
    parser_generate.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for split generation (overrides config value)",
    )
    
    parser_train = subparsers.add_parser(
        "train_autoencoder",
        help="Train AutoEncoder model"
    )
    parser_train.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file (default: config.yaml)",
    )
    parser_train.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file (.ckpt) to resume training from",
    )
    
    return parser.parse_args(), parser


def cmd_generate_splits(args):
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    seed = args.seed if args.seed is not None else config["splits"]["seed"]
    
    generate_splits(
        seed=seed,
        total_samples=config["splits"]["total_samples"],
        val_split=config["splits"]["val_split"],
        test_split=config["splits"]["test_split"],
        data_dir=config["data"]["data_dir"],
        splits_dir=config["data"]["splits_dir"],
    )


def cmd_train_autoencoder(args):
    train_autoencoder(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
    )


def run():
    args, parser = parse_args()
    match args.command:
        case "generate_splits":
            cmd_generate_splits(args)
        case "train_autoencoder":
            cmd_train_autoencoder(args)
        case _:
            parser.print_help()


if __name__ == "__main__":
    run()
