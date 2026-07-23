from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

import numpy as np

from .config import EncoderConfig


def _local_model(model_name: str) -> str:
    cache = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
    snapshots = cache / ("models--" + model_name.replace("/", "--")) / "snapshots"
    if snapshots.is_dir():
        choices = sorted((p for p in snapshots.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)
        if choices:
            return str(choices[-1])
    return model_name


def _unit(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)
    return vector if norm == 0 else vector / norm


class KeyEncoder:
    """Lazy encoder compatible with keys produced by Back2Distribution."""

    def __init__(
        self,
        language_model: str = "BAAI/bge-large-en-v1.5",
        siglip2_model: str = "google/siglip2-base-patch16-224",
        dinov2_model: str = "facebook/dinov2-base",
        device: str | None = None,
        local_files_only: bool = True,
    ):
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.local_files_only = local_files_only
        self.names = {
            "language": language_model,
            "siglip2": siglip2_model,
            "dinov2": dinov2_model,
        }
        self._text = self._text_tokenizer = None
        self._text_kind = None
        self._vision: dict[str, tuple[object, object]] = {}

    @classmethod
    def from_config(cls, config: EncoderConfig):
        return cls(
            language_model=config.language_model,
            siglip2_model=config.siglip2_model,
            dinov2_model=config.dinov2_model,
            device=None if config.device == "auto" else config.device,
            local_files_only=not config.auto_download,
        )

    def _model_load_error(self, model_name: str, error: Exception):
        mode = "local cache" if self.local_files_only else "local cache and automatic source download"
        raise RuntimeError(
            f"Could not load model {model_name!r} from the {mode}. "
            f"Download it manually with: hf download {model_name}. "
            "Then retry, or set HF_HUB_CACHE to the cache containing the model. "
            f"Original error: {error}"
        ) from error

    def _ensure_text(self):
        if self._text is not None:
            return
        source = _local_model(self.names["language"])
        try:
            try:
                from sentence_transformers import SentenceTransformer

                self._text = SentenceTransformer(
                    source, device=self.device, local_files_only=self.local_files_only
                )
                self._text_kind = "sentence_transformer"
            except Exception:
                from transformers import AutoModel, AutoTokenizer

                self._text_tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=self.local_files_only)
                self._text = AutoModel.from_pretrained(source, local_files_only=self.local_files_only).to(self.device).eval()
                self._text_kind = "transformers"
        except Exception as error:
            self._model_load_error(self.names["language"], error)

    def _ensure_vision(self, name: str):
        if name in self._vision:
            return
        from transformers import AutoImageProcessor, AutoModel

        source = _local_model(self.names[name])
        try:
            processor = AutoImageProcessor.from_pretrained(source, local_files_only=self.local_files_only)
            model = AutoModel.from_pretrained(source, local_files_only=self.local_files_only).to(self.device).eval()
        except Exception as error:
            self._model_load_error(self.names[name], error)
        self._vision[name] = (processor, model)

    def encode_text(self, text: str) -> np.ndarray:
        self._ensure_text()
        if self._text_kind == "sentence_transformer":
            return np.asarray(
                self._text.encode(str(text), normalize_embeddings=True), dtype=np.float32
            )
        inputs = self._text_tokenizer(str(text), return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            hidden = self._text(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return _unit(pooled[0].float().cpu().numpy())

    def encode_image(self, image, model_name: str) -> np.ndarray:
        from PIL import Image

        self._ensure_vision(model_name)
        processor, model = self._vision[model_name]
        if isinstance(image, (str, os.PathLike)):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image).astype(np.uint8))
        inputs = processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            if hasattr(model, "get_image_features"):
                output = model.get_image_features(**inputs)
            else:
                output = model(**inputs)
        if not self.torch.is_tensor(output):
            output = output.pooler_output if getattr(output, "pooler_output", None) is not None else output.last_hidden_state[:, 0]
        return _unit(output[0].float().cpu().numpy())

    def encode(self, images: Mapping[str, object], text: str) -> dict:
        visual = {}
        for view, image in images.items():
            view = str(view).lower().replace("-", "_").removesuffix("_rgb")
            visual[f"{view}_siglip2"] = self.encode_image(image, "siglip2")
            visual[f"{view}_dinov2"] = self.encode_image(image, "dinov2")
        return {"action_emb": self.encode_text(text), "visual": visual}
