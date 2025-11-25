import comet_ml  # noqa: F401 (import comet_ml before pytorch)
import argparse

from utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="WikiArt AutoEncoder Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    parser_generate = subparsers.add_parser(
        "generate_splits", help="Generate train/val/test splits"
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
        "train_autoencoder", help="Train AutoEncoder model"
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

    parser_generate_latent = subparsers.add_parser(
        "generate_latent_spaces",
        help="Generate latent spaces for images in dataset splits",
    )
    parser_generate_latent.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file (default: config.yaml)",
    )
    parser_generate_latent.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (.ckpt) to load model from",
    )
    parser_generate_latent.add_argument(
        "--output-dir",
        type=str,
        default="data/latent_spaces",
        help="Output directory for latent spaces (default: data/latent_spaces)",
    )
    parser_generate_latent.add_argument(
        "--cutting-seed",
        type=int,
        default=None,
        help="Seed for cutting operations (default: from config)",
    )
    parser_generate_latent.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for processing (default: 1)",
    )

    parser_umap = subparsers.add_parser(
        "visualize_umap", help="Visualize UMAP embeddings of latent spaces"
    )
    parser_umap.add_argument(
        "--input-dir",
        type=str,
        default="data/latent_spaces",
        help="Input directory containing latent space files for UMAP visualization",
    )

    return parser.parse_args(), parser


def cmd_generate_splits(args):
    from data.generate_splits import generate_splits

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
    from training import train_autoencoder

    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    train_autoencoder(
        config=config,
        checkpoint_path=args.checkpoint,
    )


def cmd_generate_latent_spaces(args):
    from data.generate_latent_spaces import generate_latent_spaces

    generate_latent_spaces(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        cutting_seed=args.cutting_seed,
        batch_size=args.batch_size,
    )

def cmd_visualize_umap(args):
    from utils.visualize import visualize_umap

    visualize_umap(
        input_dir=args.input_dir,
    )

def run():
    args, parser = parse_args()
    match args.command:
        case "generate_splits":
            cmd_generate_splits(args)
        case "train_autoencoder":
            cmd_train_autoencoder(args)
        case "generate_latent_spaces":
            cmd_generate_latent_spaces(args)
        case "visualize_umap":
            cmd_visualize_umap(args)
        case _:
            parser.print_help()


if __name__ == "__main__":
    run()
