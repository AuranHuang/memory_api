from __future__ import annotations

import heapq
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

from .encoder import KeyEncoder
from .config import EncoderConfig, resolve_config_path
from .paths import DEFAULT_BANK_DIR


DEFAULT_FIELDS = ("images", "text", "robot_state")


def _cosine(left, right) -> float:
    left, right = np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else float("-inf")


def _decode(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


@dataclass(frozen=True)
class RetrievalResult:
    """A match contains only its opaque embedding-key ID and requested values."""

    key_id: str
    language_score: float
    visual_score: float
    visual_scores: dict[str, float]
    values: dict[str, Any]


class MemoryRetriever:
    """Retrieve from the flat key/value memory-bank format.

    ``keys.hdf5`` contains only opaque key records and embedding vectors.
    ``values.hdf5`` is not opened until matching finishes, and then only the
    requested datasets under the winning opaque key IDs are read.
    """

    def __init__(self, bank_dir=None, *, encoder: KeyEncoder | None = None, config_path=None):
        self.bank_dir = Path(bank_dir) if bank_dir is not None else DEFAULT_BANK_DIR
        self.keys_path = self.bank_dir / "keys.hdf5"
        self.values_path = self.bank_dir / "values.hdf5"
        self.config_path = resolve_config_path(self.bank_dir, config_path)
        self.config = EncoderConfig.load(self.config_path)
        self._managed_encoder = encoder is None
        self.encoder = encoder or KeyEncoder.from_config(self.config)
        self._config_signature = self._file_signature(self.config_path)
        self._validate_model_metadata()

    @staticmethod
    def _file_signature(path):
        stat = Path(path).stat()
        return stat.st_mtime_ns, stat.st_size

    def _refresh_encoder_config(self):
        signature = self._file_signature(self.config_path)
        if signature != self._config_signature:
            new_config = EncoderConfig.load(self.config_path)
            if new_config != self.config:
                if not self._managed_encoder:
                    raise ValueError(
                        "encoder_config.json changed but MemoryRetriever was given an explicit encoder; "
                        "construct a new retriever with a matching encoder"
                    )
                # Drop the old managed encoder. New models remain lazy and load
                # only when the next query needs them.
                self.encoder = KeyEncoder.from_config(new_config)
                self.config = new_config
            self._config_signature = signature
        self._validate_model_metadata()

    def _validate_model_metadata(self):
        if not self.keys_path.exists():
            return
        with h5py.File(self.keys_path, "r") as file:
            recorded = {
                "language": file.attrs.get("language_model"),
                "siglip2": file.attrs.get("siglip2_model"),
                "dinov2": file.attrs.get("dinov2_model"),
            }
        mismatches = [
            f"{name}: config={self.config.models[name]!r}, keys={recorded[name]!r}"
            for name in recorded
            if recorded[name] not in (None, "custom") and recorded[name] != self.config.models[name]
        ]
        if mismatches:
            raise ValueError(
                "Encoder config does not match keys.hdf5 metadata. Regenerate keys or use the matching config. "
                + "; ".join(mismatches)
            )

    def retrieve(
        self,
        image,
        text: str,
        *,
        image_view: str = "agentview",
        fields: Sequence[str] = DEFAULT_FIELDS,
        top_k: int = 1,
        language_top_k: int = 256,
        view_weights: Mapping[str, float] | None = None,
        model_weights: Mapping[str, float] | None = None,
        verbose: bool = False,
    ) -> list[RetrievalResult]:
        self._refresh_encoder_config()
        images = image if isinstance(image, Mapping) else {image_view: image}
        images = {
            str(view).lower().replace("-", "_").removesuffix("_rgb"): value
            for view, value in images.items()
        }
        if view_weights is not None:
            view_weights = {
                str(view).lower().replace("-", "_").removesuffix("_rgb"): float(weight)
                for view, weight in view_weights.items()
            }
            if any(weight <= 0 for weight in view_weights.values()):
                raise ValueError("all supplied view_weights must be positive")
        extra_views = set(images) - {"agentview"}
        if extra_views and view_weights is None:
            raise ValueError(
                "Non-default query views require explicit view_weights; missing weights for "
                + ", ".join(sorted(extra_views))
            )
        if view_weights is not None:
            missing = set(images) - set(view_weights)
            if missing:
                raise ValueError("view_weights is missing query views: " + ", ".join(sorted(missing)))
        return self.retrieve_embeddings(
            self.encoder.encode(images, text), fields=fields, top_k=top_k,
            language_top_k=language_top_k, view_weights=view_weights,
            model_weights=model_weights, verbose=verbose,
        )

    def retrieve_embeddings(
        self,
        query: Mapping[str, Any],
        *,
        fields: Sequence[str] = DEFAULT_FIELDS,
        top_k: int = 1,
        language_top_k: int = 256,
        view_weights: Mapping[str, float] | None = None,
        model_weights: Mapping[str, float] | None = None,
        verbose: bool = False,
    ) -> list[RetrievalResult]:
        if top_k < 1 or language_top_k < 1:
            raise ValueError("top_k and language_top_k must be positive")
        query_views = {name.rsplit("_", 1)[0] for name in query.get("visual", {})}
        if query_views - {"agentview"} and view_weights is None:
            raise ValueError("Non-default query views require explicit view_weights")
        if view_weights is not None and query_views - set(view_weights):
            raise ValueError("view_weights must include every query view")
        with h5py.File(self.keys_path, "r") as file:
            keys = file["keys"]
            key_modalities = sorted({name for group in keys.values() for name in group.keys()})
            language_matches = self._language_stage(keys, query["action_emb"], language_top_k)
            visual_matches = self._visual_stage(
                keys, language_matches, query["visual"], top_k,
                dict(view_weights or {}), dict(model_weights or {}),
            )
            if verbose:
                self._print_verbose(
                    file, query, fields, top_k, language_top_k,
                    view_weights, model_weights, key_modalities,
                    len(keys), len(language_matches), len(visual_matches),
                )

        # Payload access starts here, after both matching stages have completed.
        values_by_id = self._read_values([item[1] for item in visual_matches], fields)
        language_scores = {key_id: score for score, key_id in language_matches}
        return [
            RetrievalResult(
                key_id=key_id,
                language_score=language_scores[key_id],
                visual_score=visual_score,
                visual_scores=visual_scores,
                values=values_by_id[key_id],
            )
            for visual_score, key_id, visual_scores in visual_matches
        ]

    @staticmethod
    def _print_verbose(
        file, query, fields, top_k, language_top_k, view_weights,
        model_weights, key_modalities, record_count, language_count, visual_count,
    ):
        def attr(name, default=None):
            value = file.attrs.get(name, default)
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            try:
                return json.loads(value) if isinstance(value, str) and value[:1] in "[{" else value
            except json.JSONDecodeError:
                return value

        query_modalities = sorted(query.get("visual", {}))
        effective_views = dict(view_weights or {"agentview": 1.0})
        effective_models = {"siglip2": 1.0, "dinov2": 1.0}
        effective_models.update(model_weights or {})
        report = {
            "bank": {
                "format": attr("format"),
                "record_count": record_count,
                "included_views": attr("included_views", sorted({m.rsplit('_', 1)[0] for m in key_modalities if m != 'language'})),
                "key_modalities": key_modalities,
                "models": {
                    "language": attr("language_model", "not recorded"),
                    "siglip2": attr("siglip2_model", "not recorded"),
                    "dinov2": attr("dinov2_model", "not recorded"),
                },
            },
            "query": {"visual_modalities": query_modalities, "requested_fields": list(fields)},
            "hyperparameters": {
                "language_top_k": language_top_k,
                "visual_top_k": top_k,
                "view_weights": effective_views,
                "model_weights": effective_models,
            },
            "retrieval": {
                "language_candidates": language_count,
                "visual_results": visual_count,
            },
        }
        print("[memory_api] effective retrieval configuration")
        print(json.dumps(report, indent=2, sort_keys=True))

    @staticmethod
    def _language_stage(keys, query, count):
        heap = []
        for key_id, group in keys.items():
            score = _cosine(query, group["language"][...])
            entry = (score, key_id)
            if len(heap) < count:
                heapq.heappush(heap, entry)
            elif entry > heap[0]:
                heapq.heapreplace(heap, entry)
        return sorted(heap, reverse=True)

    @staticmethod
    def _visual_stage(keys, candidates, query, count, view_weights, model_weights):
        heap = []
        for _, key_id in candidates:
            group = keys[key_id]
            scores, total, total_weight = {}, 0.0, 0.0
            for modality, vector in query.items():
                if modality not in group:
                    continue
                score = _cosine(vector, group[modality][...])
                view, model = modality.rsplit("_", 1)
                weight = float(view_weights.get(view, 1.0)) * float(model_weights.get(model, 1.0))
                scores[modality] = score
                total += score * weight
                total_weight += weight
            if not total_weight:
                continue
            entry = (total / total_weight, key_id, scores)
            comparable = entry[:2]
            if len(heap) < count:
                heapq.heappush(heap, entry)
            elif comparable > heap[0][:2]:
                heapq.heapreplace(heap, entry)
        return sorted(heap, reverse=True)

    def _read_values(self, key_ids: Sequence[str], fields: Sequence[str]):
        result = {}
        with h5py.File(self.values_path, "r") as file:
            records = file["values"]
            for key_id in key_ids:
                record = records[key_id]
                selected = {}
                for field in fields:
                    if field in ("image", "images"):
                        selected[field] = {
                            name.removeprefix("image__"): dataset[...]
                            for name, dataset in record.items()
                            if name.startswith("image__")
                        }
                    else:
                        if field not in record:
                            raise KeyError(f"Value {field!r} is not stored for key {key_id}")
                        selected[field] = _decode(record[field][()])
                result[key_id] = selected
        return result
