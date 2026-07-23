from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CONFIG_PATH = Path(__file__).with_name("encoder_config.json")


@dataclass(frozen=True)
class EncoderConfig:
    language_model: str
    siglip2_model: str
    dinov2_model: str
    device: str = "auto"
    auto_download: bool = True

    @classmethod
    def load(cls, path=None) -> "EncoderConfig":
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        required = ("language_model", "siglip2_model", "dinov2_model")
        missing = [name for name in required if not data.get(name)]
        if missing:
            raise ValueError(f"Encoder config {path} is missing: {', '.join(missing)}")
        device = str(data.get("device", "auto"))
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("encoder config device must be auto, cpu, or cuda")
        return cls(
            language_model=str(data["language_model"]),
            siglip2_model=str(data["siglip2_model"]),
            dinov2_model=str(data["dinov2_model"]),
            device=device,
            auto_download=bool(data.get("auto_download", True)),
        )

    @property
    def models(self):
        return {
            "language": self.language_model,
            "siglip2": self.siglip2_model,
            "dinov2": self.dinov2_model,
        }

    def save(self, path):
        path = Path(path)
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("w", encoding="utf-8") as file:
            json.dump({
                "language_model": self.language_model,
                "siglip2_model": self.siglip2_model,
                "dinov2_model": self.dinov2_model,
                "device": self.device,
                "auto_download": self.auto_download,
            }, file, indent=2)
            file.write("\n")
        temporary.replace(path)


def resolve_config_path(bank_dir, config_path=None) -> Path:
    if config_path:
        return Path(config_path)
    bank_config = Path(bank_dir) / "encoder_config.json"
    return bank_config if bank_config.exists() else DEFAULT_CONFIG_PATH
