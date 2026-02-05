import argparse
from pathlib import Path
from typing import Any, Dict


def load_yaml(path: Path) -> Dict[str, Any]:
    import yaml  # type: ignore

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML must be a mapping: {path}")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO inference runner.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--weights", type=str, default="")
    parser.add_argument("--source", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(Path(args.config).resolve())

    weights = args.weights or cfg.get("weights", "")
    source = args.source or cfg.get("source", "")
    if not weights or not source:
        raise ValueError("weights and source must be provided.")

    conf = float(cfg.get("conf", 0.25))
    save_pic = bool(cfg.get("save_pic", True))
    project = Path(cfg.get("project", "experiments")).resolve()
    name = cfg.get("name", "infer")

    from ultralytics import YOLO

    model = YOLO(weights)
    model.predict(
        source=source,
        save=save_pic,
        conf=conf,
        project=str(project),
        name=str(name),
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
