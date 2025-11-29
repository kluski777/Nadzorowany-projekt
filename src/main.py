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
        "--architecture",
        type=str,
        default=None,
        help="Name of an autoencoder's architecture",
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

    parser_fit_feature_extractor = subparsers.add_parser(
        "fit_feature_extractor", help="Fit Feature Extractor on latent spaces"
    )
    parser_fit_feature_extractor.add_argument(
        "--input-dir",
        type=str,
        default="data/latent_spaces",
        help="Input directory containing latent space files for Feature Extractor fitting",
    )
    parser_fit_feature_extractor.add_argument(
        "--output-dir",
        type=str,
        default="data/models",
        help="Output directory for Feature Extractor fitting",
    )
    parser_fit_feature_extractor.add_argument(
        "--n-components",
        type=int,
        default=50,
        help="Number of principal components for Feature Extractor (default: 50)",
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
        architecture=args.architecture,
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
    from utils import visualize_umap

    print(f"Visualizing UMAP embeddings from latent spaces in: {args.input_dir}")

    visualize_umap(
        input_dir=args.input_dir,
    )

    print("UMAP visualization completed")

def cmd_fit_feature_extractor(args):
    from training import fit_feature_extractor

    print(f"Fitting Feature Extractor on latent spaces from: {args.input_dir}")

    fit_feature_extractor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        n_components=args.n_components,
    )

    print("Feature Extractor fitting completed and saved")

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
        case "fit_feature_extractor":
            cmd_fit_feature_extractor(args)
        case _:
            parser.print_help()


if __name__ == "__main__":
    run()
