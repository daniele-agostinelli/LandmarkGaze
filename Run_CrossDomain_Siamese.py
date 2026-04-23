import argparse

from cross_domain_utils import run_cross_domain_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cross-domain validation for a Siamese landmark model."
    )
    parser.add_argument(
        "--model-path",
        default="models/blender_siameseMLP.pth",
        help="Path to Siamese checkpoint (.pth).",
    )
    parser.add_argument(
        "--source-dataset",
        default="blender",
        help="Source dataset name used to infer default target datasets.",
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        default=None,
        help="Optional explicit target datasets. Default: all other real datasets.",
    )
    parser.add_argument(
        "--config-path",
        default="configs/default_config.yaml",
        help="Base OmegaConf yaml file.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help='Inference device, e.g. "cuda", "cuda:0", or "cpu".',
    )
    parser.add_argument(
        "--output-dir",
        default="models/stats/cross_domain/blender_siamese",
        help="Output directory for detailed rows and summary CSV.",
    )
    parser.add_argument(
        "--tag",
        default="blender_siamese",
        help="Tag used in output filenames.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_cross_domain_validation(
        model_path=args.model_path,
        model_type="normalized_siamese",
        config_path=args.config_path,
        device=args.device,
        output_dir=args.output_dir,
        tag=args.tag,
        source_dataset=args.source_dataset,
        targets=args.targets,
    )


if __name__ == "__main__":
    main()
