# Flat Memory API: setup and usage guide

This package provides two operations:

1. **Key processing:** encode the text and RGB images in `values.hdf5` and
   generate `keys.hdf5`.
2. **Retrieval:** use query text for first-stage filtering, use query images for
   second-stage ranking, and then read only requested values for the matches.

The input is assumed to already use the flat value format. There is no task,
demo, step, conversion, or sampling logic in this package.

## 1. Installation

Python 3.10 or newer is recommended. Run commands from the directory containing
`memory_api` or add that directory to `PYTHONPATH`.

```bash
python -m pip install numpy h5py Pillow torch transformers sentence-transformers
```

A CUDA GPU is strongly recommended for processing a large bank. Retrieval
scans stored embeddings on CPU but query encoding can use either CUDA or CPU.

The default models are:

| Purpose | Default model | Output name |
|---|---|---|
| Language | `BAAI/bge-large-en-v1.5` | `language` |
| Semantic vision | `google/siglip2-base-patch16-224` | `<view>_siglip2` |
| Structural vision | `facebook/dinov2-base` | `<view>_dinov2` |

Model selection is controlled by `encoder_config.json`:

```json
{
  "language_model": "BAAI/bge-large-en-v1.5",
  "siglip2_model": "google/siglip2-base-patch16-224",
  "dinov2_model": "facebook/dinov2-base",
  "device": "auto",
  "auto_download": true
}
```

Put this file inside the bank directory. If it is absent, the packaged
[default config](encoder_config.json) is used. `device="auto"` selects CUDA
when available and CPU otherwise.

Loading follows this sequence:

1. Read the configured model IDs.
2. Check the local Hugging Face snapshot cache.
3. Reuse models already loaded in the `MemoryRetriever` encoder instance.
4. If a model is absent and `auto_download=true`, pull it from its configured
   Hugging Face source.
5. If downloading fails, raise an error containing the exact manual command,
   for example `hf download BAAI/bge-large-en-v1.5`.

Set `auto_download=false` for a strictly offline deployment. In that mode all
models must already be cached. Models are lazy: language loads on the first
text encoding, while SigLIP2 and DINOv2 load on the first image encoding.

## 2. Bank directory

One bank is one directory containing two HDF5 files:

```text
my_memory_bank/
├── encoder_config.json # authoritative model/device/download configuration
├── values.hdf5       # supplied by the application
└── keys.hdf5         # generated from values.hdf5
```

The included/default bank is located at `memory_api/memory_bank_flat`. Calling
`MemoryRetriever()` without a path uses it automatically:

```python
from memory_api import MemoryRetriever

api = MemoryRetriever()
```

Likewise, `python -m memory_api.generate_keys` processes that default bank when
`--bank-dir` is omitted. Pass an explicit bank directory for production data.

Do not manually assign meaning to a key ID. It is only an opaque, unique link
between one key record and one value record. IDs must not contain `/`, because
HDF5 interprets `/` as a path separator. UUIDs or hexadecimal hashes work well.

## 3. Required `values.hdf5` structure

```text
values.hdf5
└── values/
    ├── <opaque-id-1>/
    │   ├── text                    REQUIRED
    │   ├── image__agentview        REQUIRED by default
    │   ├── image__eye_in_hand      optional additional view
    │   ├── robot_state             optional
    │   └── ...                     any other optional values
    └── <opaque-id-2>/
        └── ...
```

### Required values

Every record must contain:

- `text`: a scalar UTF-8 string describing the memory. This is encoded into
  the language key.
- `image__agentview`: an RGB image with shape `(H, W, 3)` for the default
  configuration.
  `uint8` with values from 0 through 255 is recommended. The text after
  `image__` is the view name, such as `agentview`, `eye_in_hand`, or `front`.

Other `image__<view>` datasets are optional. They are included in keys only
when their weights are explicitly supplied during processing. All records must
contain every view selected for key generation.

### Optional values

Everything except `text` and the views explicitly selected for key generation
is optional. With default settings, `image__agentview` is required. Examples of
optional payloads include:

```text
robot_state
logical_state
object_state
object_target
object_target_pose
raw_state
action
reward
force_torque
point_cloud
next_robot_state
```

Optional values do not participate in matching. They are payloads retrieved
after an opaque key has matched. New value names require no code change.

Values may be HDF5 numeric scalars, arrays, strings, or JSON strings. JSON is
convenient for dictionaries and nested structures: the API automatically
decodes a retrieved JSON string into Python dictionaries/lists. Ordinary
non-image NumPy arrays are returned as lists. Images are returned as NumPy
arrays.

### Example: create `values.hdf5`

```python
import json
from pathlib import Path

import h5py
import numpy as np

bank_dir = Path("my_memory_bank")
bank_dir.mkdir(parents=True, exist_ok=True)

utf8 = h5py.string_dtype(encoding="utf-8")

with h5py.File(bank_dir / "values.hdf5", "w") as file:
    values = file.create_group("values")

    # Use a UUID/hash generated by your producer. It has no task semantics.
    record = values.create_group("8b04fc51f45d4f63a6bf")
    record.create_dataset("text", data="pick up the red bowl", dtype=utf8)
    record.create_dataset(
        "image__agentview",
        data=np.asarray(agent_rgb, dtype=np.uint8),
        compression="gzip",
    )
    record.create_dataset(
        "image__eye_in_hand",
        data=np.asarray(wrist_rgb, dtype=np.uint8),
        compression="gzip",
    )
    record.create_dataset(
        "robot_state",
        data=json.dumps({"eef_pose": [0.1, 0.2, 0.3], "gripper": 0.0}),
        dtype=utf8,
    )
    record.create_dataset("reward", data=np.float32(0.75))
```

## 4. Generate keys from values

By default, the processor reads only `text` and `image__agentview`. It does not
load other views, robot state, or other payload fields.

### Command line

```bash
python -m memory_api.generate_keys \
  --bank-dir my_memory_bank \
  --config my_memory_bank/encoder_config.json \
  --device cuda
```

To include additional views, explicitly assign every included view a positive
weight. Include `agentview` explicitly when using any `--view-weight` option:

```bash
python -m memory_api.generate_keys \
  --bank-dir my_memory_bank \
  --device cuda \
  --view-weight agentview=1.0 \
  --view-weight eye_in_hand=1.5
```

If `keys.hdf5` already exists, generation stops rather than overwriting it.
Pass `--overwrite` only when values changed and keys must be rebuilt:

```bash
python -m memory_api.generate_keys \
  --bank-dir my_memory_bank \
  --device cuda \
  --overwrite
```

Key generation copies the normalized configuration to
`my_memory_bank/encoder_config.json` and records all three model IDs as HDF5
metadata in `keys.hdf5`.

### Python API and custom encoder

```python
from memory_api.encoder import KeyEncoder
from memory_api.config import EncoderConfig
from memory_api.generate_keys import generate_keys_from_values

config = EncoderConfig.load("my_memory_bank/encoder_config.json")
encoder = KeyEncoder.from_config(config)

generate_keys_from_values(
    "my_memory_bank",
    encoder=encoder,
    config_path="my_memory_bank/encoder_config.json",
    overwrite=True,
    view_weights={"agentview": 1.0, "eye_in_hand": 1.5},
)
```

Generation writes `keys.hdf5.tmp` first and atomically moves it to
`keys.hdf5` only after every record succeeds. The generated structure is:

```text
keys.hdf5
└── keys/
    └── <same-opaque-id>/
        ├── language
        ├── agentview_siglip2
        ├── agentview_dinov2
        ├── eye_in_hand_siglip2
        └── eye_in_hand_dinov2
```

If values are added, removed, or their text/images change, regenerate the key
file. Changing only an optional payload such as `robot_state` does not affect
embeddings, although rebuilding is harmless.

**The encoder configuration used for retrieval must match key generation.** A
different language or vision model produces incompatible vector spaces and
invalid similarity scores.

## 5. Basic retrieval

```python
from memory_api import MemoryRetriever

api = MemoryRetriever("my_memory_bank")

matches = api.retrieve(
    image={
        "agentview": current_agent_rgb,
        "eye_in_hand": current_wrist_rgb,
    },
    text="pick up the red bowl",
    view_weights={"agentview": 1.0, "eye_in_hand": 1.5},
    fields=("images", "text", "robot_state"),
    language_top_k=100,
    top_k=5,
    verbose=True,
)

for match in matches:
    print(match.key_id)
    print(match.language_score)
    print(match.visual_score)
    print(match.visual_scores)
    print(match.values)
```

For a single image, provide the image directly and identify its view:

```python
matches = api.retrieve(
    image=current_agent_rgb,
    image_view="agentview",
    text="pick up the red bowl",
)
```

`MemoryRetriever` reads `encoder_config.json` automatically and verifies its
three model IDs against metadata in `keys.hdf5`. A mismatch stops immediately
with instructions to use the matching config or regenerate keys. The encoder
object and its loaded models are retained by the API instance and reused across
calls; create one retriever and keep it alive instead of constructing one for
every query. Before each `retrieve(...)` call, the API checks the config file.
If a managed configuration changed, it discards the old managed encoder and
creates a new lazy encoder from the new metadata. It then checks that metadata
against `keys.hdf5`; changed models require matching regenerated keys.

Images may be NumPy arrays, PIL images, or filesystem paths accepted by PIL.
Dictionary view names are normalized to lowercase, hyphens become underscores,
and a trailing `_rgb` is removed.

## 6. Retrieval algorithm

```text
query text
   │ encode
   ▼
compare with every keys/<id>/language using cosine similarity
   │ retain language_top_k
   ▼
query image view(s)
   │ encode with SigLIP2 and DINOv2
   ▼
weighted visual cosine similarity over shared modalities
   │ retain top_k
   ▼
close keys.hdf5, open values.hdf5
   │
read only requested fields for matching IDs
```

Cosine similarity is:

```text
cosine(a, b) = dot(a, b) / (norm(a) * norm(b))
```

Language similarity filters candidates; it is not mixed numerically into the
visual score. Visual score is the weighted mean:

```text
modality_weight = view_weight × model_weight

visual_score = sum(modality_cosine × modality_weight)
               / sum(modality_weight)
```

Only modalities shared by the query and stored key participate.

The default key includes only `language`, `agentview_siglip2`, and
`agentview_dinov2`. Additional views exist in the key only if selected with
explicit weights during key generation. When a retrieval query supplies any
non-`agentview` view, `view_weights` is mandatory and must name every query
view. This prevents additional views from silently receiving unintended equal
weight.

## 7. Hyperparameters

### `language_top_k`

Maximum number of memories passed from language filtering to visual ranking.
Default: `256`.

- Smaller values are faster but may exclude a visually correct memory.
- Larger values improve visual recall but require more visual comparisons.
- Effective count is `min(language_top_k, number_of_records)`.
- A useful starting point is 50–200 for small banks and 200–1000 for larger or
  linguistically repetitive banks.

### `top_k`

Maximum number of final visual matches returned. Default: `1`.

- Use `1` for a single best memory.
- Use larger values for reranking, inspection, voting, or downstream planning.
- The result count cannot exceed the language candidate count or the number of
  candidates sharing visual modalities with the query.

### `image_view`

View name assigned when `image` is a single image instead of a dictionary.
Default: `agentview`. It must match the suffix used in `image__<view>`.

### `fields`

Payload datasets to load after matching. Default:

```python
("images", "text", "robot_state")
```

`"images"` or `"image"` loads every `image__*` dataset under the matched value
record. Any other field reads only the dataset with that exact name. To minimize
memory and I/O, request only what the caller needs:

```python
fields=("text", "reward")
```

### `verbose`

Default: `False`. Set `verbose=True` to print a JSON report containing:

- bank record count and key format;
- every modality included in stored keys;
- recorded encoder model names;
- query modalities and requested payload fields;
- effective language and visual top-K values;
- effective view and model weights;
- actual language candidate and final result counts.

```python
matches = api.retrieve(image, text, verbose=True)
```

### `view_weights`

Weights camera viewpoints in visual ranking. `agentview` defaults to 1.0 when
it is the only query view. For multi-view queries, provide every view:

```python
view_weights={
    "agentview": 1.0,
    "eye_in_hand": 2.0,
}
```

This makes each wrist-camera modality twice as influential as the corresponding
agent-view modality.

### `model_weights`

Weights the two vision encoders. Unspecified models default to 1.0:

```python
model_weights={
    "siglip2": 2.0,
    "dinov2": 1.0,
}
```

Combined example:

```python
matches = api.retrieve(
    image={"agentview": agent_rgb, "eye_in_hand": wrist_rgb},
    text="place the cup in the drawer",
    language_top_k=200,
    top_k=3,
    view_weights={"agentview": 1.0, "eye_in_hand": 1.5},
    model_weights={"siglip2": 2.0, "dinov2": 1.0},
    fields=("text", "robot_state"),
)
```

The resulting modality weights are:

| Modality | Weight |
|---|---:|
| `agentview_siglip2` | 1.0 × 2.0 = 2.0 |
| `agentview_dinov2` | 1.0 × 1.0 = 1.0 |
| `eye_in_hand_siglip2` | 1.5 × 2.0 = 3.0 |
| `eye_in_hand_dinov2` | 1.5 × 1.0 = 1.5 |

Use non-negative weights. At least one shared modality must have positive total
weight. Setting a view/model weight to zero disables its contribution.

## 8. Precomputed query embeddings

If another component already caches query embeddings, avoid model inference:

```python
query = encoder.encode(
    {"agentview": agent_rgb, "eye_in_hand": wrist_rgb},
    "pick up the red bowl",
)

matches = api.retrieve_embeddings(
    query,
    fields=("text", "robot_state"),
    language_top_k=100,
    top_k=5,
)
```

Expected query structure:

```python
{
    "action_emb": np.ndarray,  # language vector
    "visual": {
        "agentview_siglip2": np.ndarray,
        "agentview_dinov2": np.ndarray,
        # additional views...
    },
}
```

## 9. Returned result

Each result is a `RetrievalResult`:

```python
RetrievalResult(
    key_id="8b04fc51f45d4f63a6bf",
    language_score=0.91,
    visual_score=0.87,
    visual_scores={
        "agentview_siglip2": 0.90,
        "agentview_dinov2": 0.84,
    },
    values={
        "text": "pick up the red bowl",
        "robot_state": {"eef_pose": [0.1, 0.2, 0.3]},
    },
)
```

`values.hdf5` is opened only after both matching stages finish. HDF5 group
lookup does not load its children: only datasets named in `fields` are read.

## 10. Integrating into a larger system

See the runnable [application service example](example.py). The recommended
pattern is to create one long-lived service during application or worker
startup and inject it into controllers, planners, request handlers, or robot
runtime components:

```python
from memory_api.example import create_memory_service

# Application startup: models have not loaded yet; they remain lazy.
memory = create_memory_service({
    "memory_bank_dir": "/data/robot_memory",
    "memory_encoder_config": "/data/robot_memory/encoder_config.json",
})

# Later, inside a controller/request handler. The first call loads models;
# subsequent calls reuse them.
matches = memory.retrieve(
    text=current_instruction,
    images={"agentview": observation["agent_rgb"]},
    fields=("text", "robot_state", "action"),
    language_top_k=200,
    visual_top_k=5,
    minimum_visual_score=0.70,
)

if matches:
    prior_robot_state = matches[0]["values"]["robot_state"]
    prior_action = matches[0]["values"]["action"]
else:
    # Application-specific no-match fallback.
    prior_robot_state = None
    prior_action = None
```

The wrapper:

- owns one `MemoryRetriever` and reuses its loaded encoder models;
- serializes shared access during lazy loading/config refresh;
- maps the public `visual_top_k` name to the core API's `top_k` argument;
- optionally applies application-level minimum score thresholds;
- returns dataclasses as ordinary dictionaries for easier system integration;
- does not initialize a retriever as a side effect of importing the module.

For multi-process inference, create one service per worker. For multi-view
queries, supply all required view weights exactly as described above.

Run the example against the included bank:

```bash
python -m memory_api.example
```

## 11. Recommended tuning procedure

1. Start with equal view and model weights.
2. Set `language_top_k` large enough that known correct memories survive stage
   one; evaluate recall rather than guessing.
3. Choose `top_k` based on the downstream consumer.
4. Inspect `language_score`, `visual_score`, and per-modality `visual_scores`.
5. Increase the weight of views/models that are reliable for your environment.
6. Validate on held-out queries before reducing `language_top_k` for speed.

The current API does not apply a minimum similarity threshold. If an
application must reject weak matches, inspect returned scores and apply its own
validated threshold.

## 12. Common errors

- **A model cannot be loaded/downloaded:** run the `hf download <model-id>`
  command included in the exception, then retry. Confirm `HF_HUB_CACHE` points
  to that cache. In networked environments, set `auto_download=true`.
- **Missing `text`:** every value record needs a scalar `text` dataset.
- **Missing `image__agentview`:** the default processor requires this RGB view.
  With explicit view weights, every record must contain each selected
  `image__<view>` dataset.
- **Empty result:** query and stored keys probably have no common view names, or
  all shared modality weights are zero.
- **Missing requested field:** remove it from `fields` or add that dataset to
  every record that may be returned.
- **Poor matches after changing a model:** regenerate all keys using exactly the
  same encoder configuration used for queries.
- **Stale keys after changing text/images:** rerun key generation with
  `overwrite=True` or `--overwrite`.

## 13. Tests

Fast processing and retrieval tests:

```bash
python -m pytest -q \
  memory_api/tests/test_generate_keys.py \
  memory_api/tests/test_retriever.py
```

Real-bank model and round-trip test:

```bash
RUN_REAL_MEMORY_TEST=1 python -m pytest -q -s \
  memory_api/tests/test_same_memory.py
```
