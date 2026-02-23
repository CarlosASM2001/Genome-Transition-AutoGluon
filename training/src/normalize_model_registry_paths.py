import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "training/reports/model_registry.json"


def extract_training_relative(path_value: str) -> str | None:
    normalized = str(path_value).replace("\\", "/")
    marker = "training/"
    idx = normalized.lower().find(marker)
    if idx == -1:
        return None
    return normalized[idx:]


def to_relative_path(path_value: str) -> str:
    raw = str(path_value).strip()
    if not raw:
        return raw

    normalized = raw.replace("\\", "/")
    if normalized.startswith("training/"):
        return normalized

    path_obj = Path(raw)
    if path_obj.is_absolute():
        try:
            return path_obj.resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            extracted = extract_training_relative(raw)
            return extracted if extracted else normalized

    extracted = extract_training_relative(raw)
    return extracted if extracted else normalized


def normalize_registry_paths(registry_path: Path = DEFAULT_REGISTRY_PATH) -> dict:
    if not registry_path.exists():
        raise FileNotFoundError(f"Registry not found: {registry_path}")

    with registry_path.open("r", encoding="utf-8") as f:
        registry = json.load(f)

    models = registry.get("models", {})
    updated = 0
    for transition, cfg in models.items():
        old_path = cfg.get("path")
        if old_path is None:
            continue
        new_path = to_relative_path(old_path)
        if new_path != old_path:
            models[transition]["path"] = new_path
            updated += 1

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)

    print(f"Normalized {updated} path(s) in: {registry_path}")
    return registry


if __name__ == "__main__":
    normalize_registry_paths()
