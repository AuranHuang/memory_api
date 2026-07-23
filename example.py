"""Production-style integration example for a larger application."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from threading import Lock
from typing import Mapping, Sequence

from memory_api import MemoryRetriever
from memory_api.paths import DEFAULT_BANK_DIR


class MemoryService:
    """Long-lived application service that owns and reuses one retriever.

    Create this once during process/worker startup. The lock makes lazy model
    initialization and config refresh safe when callers share the service.
    For high-throughput GPU serving, create one service per inference worker.
    """

    def __init__(self, bank_dir, *, config_path=None):
        self.bank_dir = Path(bank_dir)
        self._retriever = MemoryRetriever(
            self.bank_dir,
            config_path=config_path,
        )
        self._lock = Lock()

    def retrieve(
        self,
        *,
        text: str,
        images: Mapping[str, object],
        fields: Sequence[str] = ("text", "robot_state"),
        language_top_k: int = 100,
        visual_top_k: int = 1,
        view_weights: Mapping[str, float] | None = None,
        model_weights: Mapping[str, float] | None = None,
        minimum_language_score: float | None = None,
        minimum_visual_score: float | None = None,
        verbose: bool = False,
    ) -> list[dict]:
        """Return JSON-like records; image fields remain NumPy arrays."""
        with self._lock:
            matches = self._retriever.retrieve(
                image=images,
                text=text,
                fields=fields,
                language_top_k=language_top_k,
                top_k=visual_top_k,
                view_weights=view_weights,
                model_weights=model_weights,
                verbose=verbose,
            )

        accepted = []
        for match in matches:
            if minimum_language_score is not None and match.language_score < minimum_language_score:
                continue
            if minimum_visual_score is not None and match.visual_score < minimum_visual_score:
                continue
            accepted.append(asdict(match))
        return accepted


def create_memory_service(settings: Mapping[str, object]) -> MemoryService:
    """Example dependency-injection factory called at application startup."""
    return MemoryService(
        bank_dir=settings["memory_bank_dir"],
        config_path=settings.get("memory_encoder_config"),
    )


# Example application setup. Do this once, not inside a control-loop iteration.
if __name__ == "__main__":
    import h5py

    service = create_memory_service({
        "memory_bank_dir": DEFAULT_BANK_DIR,
        "memory_encoder_config": DEFAULT_BANK_DIR / "encoder_config.json",
    })

    # Use one existing value only to make this file directly runnable.
    with h5py.File(DEFAULT_BANK_DIR / "values.hdf5", "r") as file:
        key_id = next(iter(file["values"]))
        record = file[f"values/{key_id}"]
        query_text = record["text"][()].decode("utf-8")
        query_image = record["image__agentview"][...]

    results = service.retrieve(
        text=query_text,
        images={"agentview": query_image},
        fields=("text", "robot_state"),
        language_top_k=100,
        visual_top_k=3,
        model_weights={"siglip2": 1.0, "dinov2": 1.0},
        verbose=True,
    )
    print(results[0])
