"""Generate embedding keys from a flat ``values.hdf5`` bank."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py

from .encoder import KeyEncoder
from .config import EncoderConfig, resolve_config_path
from .paths import DEFAULT_BANK_DIR


def generate_keys_from_values(
    bank_dir,
    *,
    encoder: KeyEncoder | None = None,
    output_path=None,
    overwrite: bool = False,
    view_weights=None,
    config_path=None,
):
    """Encode text and images for every flat value record.

    Only ``text`` and datasets named ``image__<view>`` are read. Other values,
    such as robot state or logical state, never enter memory during key creation.
    The output is written atomically so an interrupted run does not damage an
    existing key store.
    """
    bank_dir = Path(bank_dir)
    values_path = bank_dir / "values.hdf5"
    output_path = Path(output_path) if output_path else bank_dir / "keys.hdf5"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists; pass overwrite=True")

    config_file = resolve_config_path(bank_dir, config_path)
    config = EncoderConfig.load(config_file)
    encoder = encoder or KeyEncoder.from_config(config)
    view_weights = {"agentview": 1.0} if view_weights is None else {
        str(name).lower().replace("-", "_").removesuffix("_rgb"): float(weight)
        for name, weight in view_weights.items()
    }
    if not view_weights or any(weight <= 0 for weight in view_weights.values()):
        raise ValueError("view_weights must contain at least one view and all weights must be positive")
    temporary = output_path.with_name(output_path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()

    try:
        with h5py.File(values_path, "r") as values_file, h5py.File(temporary, "w") as keys_file:
            values = values_file["values"]
            keys = keys_file.create_group("keys", track_order=True)
            for key_id, record in values.items():
                if "text" not in record:
                    raise KeyError(f"Flat value {key_id} has no 'text' dataset")
                raw_text = record["text"][()]
                text = raw_text.decode("utf-8") if isinstance(raw_text, bytes) else str(raw_text)
                images = {}
                for view in view_weights:
                    dataset_name = f"image__{view}"
                    if dataset_name not in record:
                        raise KeyError(f"Flat value {key_id} has no {dataset_name!r} dataset")
                    images[view] = record[dataset_name][...]

                embeddings = encoder.encode(images, text)
                key_record = keys.create_group(key_id)
                key_record.create_dataset("language", data=embeddings["action_emb"])
                for modality, vector in sorted(embeddings["visual"].items()):
                    key_record.create_dataset(modality, data=vector)

            keys_file.attrs["format"] = "flat-embedding-key-v1"
            keys_file.attrs["record_count"] = len(keys)
            keys_file.attrs["generated_from"] = "values.hdf5:text,image__*"
            keys_file.attrs["included_views"] = json.dumps(sorted(view_weights))
            keys_file.attrs["view_weights"] = json.dumps(view_weights, sort_keys=True)
            names = getattr(encoder, "names", {})
            keys_file.attrs["language_model"] = names.get("language", "custom")
            keys_file.attrs["siglip2_model"] = names.get("siglip2", "custom")
            keys_file.attrs["dinov2_model"] = names.get("dinov2", "custom")
        os.replace(temporary, output_path)
        config.save(bank_dir / "encoder_config.json")
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-dir", default=str(DEFAULT_BANK_DIR))
    parser.add_argument("--output")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--config", help="encoder config JSON; defaults to BANK/encoder_config.json")
    parser.add_argument(
        "--view-weight", action="append", default=[], metavar="VIEW=WEIGHT",
        help="view to encode and its retrieval weight; default: agentview=1.0",
    )
    args = parser.parse_args()
    view_weights = None
    if args.view_weight:
        view_weights = {}
        for item in args.view_weight:
            try:
                name, raw_weight = item.rsplit("=", 1)
                view_weights[name] = float(raw_weight)
            except ValueError as error:
                parser.error(f"invalid --view-weight {item!r}; expected VIEW=WEIGHT")
    config = EncoderConfig.load(resolve_config_path(args.bank_dir, args.config))
    if args.device:
        config = EncoderConfig(
            config.language_model, config.siglip2_model, config.dinov2_model,
            device=args.device, auto_download=config.auto_download,
        )
    encoder = KeyEncoder.from_config(config)
    path = generate_keys_from_values(
        args.bank_dir, encoder=encoder, output_path=args.output,
        overwrite=args.overwrite, view_weights=view_weights,
        config_path=args.config,
    )
    print(path)


if __name__ == "__main__":
    main()
