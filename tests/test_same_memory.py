"""Opt-in test: a real flat-bank value retrieves its own opaque key."""
import os
import h5py
import pytest

from memory_api import MemoryRetriever
from memory_api.paths import DEFAULT_BANK_DIR

BANK = DEFAULT_BANK_DIR


@pytest.mark.skipif(os.environ.get("RUN_REAL_MEMORY_TEST") != "1", reason="large model integration test")
def test_same_image_and_text_retrieve_same_flat_memory():
    with h5py.File(BANK / "values.hdf5", "r") as values:
        key_id = next(iter(values["values"]))
        record = values[f"values/{key_id}"]
        text = record["text"][()].decode()
        images = {
            name.removeprefix("image__"): dataset[...]
            for name, dataset in record.items() if name.startswith("image__")
        }
        images = {"agentview": images["agentview"]}
    api = MemoryRetriever(BANK)
    match = api.retrieve(images, text, fields=("text", "robot_state"), language_top_k=100)[0]
    assert match.key_id == key_id
