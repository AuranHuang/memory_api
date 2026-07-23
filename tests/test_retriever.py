import json
import pytest

import h5py
import numpy as np

from memory_api import MemoryRetriever


class NoEncoder:
    def encode(self, images, text):
        raise AssertionError("embedding-level test must not load models")


def test_flat_hierarchy_and_requested_values(tmp_path):
    with h5py.File(tmp_path / "keys.hdf5", "w") as h5:
        keys = h5.create_group("keys")
        for key_id, language, visual in (
            ("opaque-a", [1, 0], [1, 0]),
            ("opaque-b", [1, 0], [0, 1]),
            ("opaque-c", [0, 1], [0, 1]),
        ):
            group = keys.create_group(key_id)
            group.create_dataset("language", data=language)
            group.create_dataset("agentview_siglip2", data=visual)

    with h5py.File(tmp_path / "values.hdf5", "w") as h5:
        values = h5.create_group("values")
        for key_id in ("opaque-a", "opaque-b", "opaque-c"):
            group = values.create_group(key_id)
            group.create_dataset("text", data=key_id)
            group.create_dataset("robot_state", data=json.dumps({"key": key_id}))
            group.create_dataset("future_value", data=json.dumps({"expandable": True}))

    api = MemoryRetriever(tmp_path, encoder=NoEncoder())
    result = api.retrieve_embeddings(
        {"action_emb": np.array([1, 0]), "visual": {"agentview_siglip2": np.array([0, 1])}},
        fields=("text", "robot_state", "future_value"), language_top_k=2,
    )[0]
    assert result.key_id == "opaque-b"
    assert result.values == {
        "text": "opaque-b", "robot_state": {"key": "opaque-b"},
        "future_value": {"expandable": True},
    }


def test_non_default_view_requires_weight(tmp_path):
    api = MemoryRetriever(tmp_path, encoder=NoEncoder())
    with pytest.raises(ValueError, match="explicit view_weights"):
        api.retrieve_embeddings({
            "action_emb": np.array([1, 0]),
            "visual": {"eye_in_hand_siglip2": np.array([1, 0])},
        })
