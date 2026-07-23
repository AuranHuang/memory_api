import h5py
import numpy as np

from memory_api.generate_keys import generate_keys_from_values


class RecordingEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, images, text):
        self.calls.append((set(images), text))
        return {
            "action_emb": np.array([1.0, 2.0], dtype=np.float32),
            "visual": {
                f"{view}_mock": np.array([3.0, 4.0], dtype=np.float32)
                for view in images
            },
        }


def test_generate_keys_reads_encodable_values_only(tmp_path):
    with h5py.File(tmp_path / "values.hdf5", "w") as file:
        record = file.create_group("values/opaque-id")
        record.create_dataset("text", data="pick up the bowl")
        record.create_dataset("image__agentview", data=np.zeros((2, 2, 3), dtype=np.uint8))
        record.create_dataset("robot_state", data="must not be decoded or encoded")

    encoder = RecordingEncoder()
    generated = generate_keys_from_values(tmp_path, encoder=encoder)
    assert encoder.calls == [({"agentview"}, "pick up the bowl")]
    with h5py.File(generated, "r") as file:
        record = file["keys/opaque-id"]
        np.testing.assert_array_equal(record["language"][...], [1, 2])
        np.testing.assert_array_equal(record["agentview_mock"][...], [3, 4])
        assert "robot_state" not in record
