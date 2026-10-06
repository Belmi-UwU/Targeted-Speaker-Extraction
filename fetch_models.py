"""Restore pinned model weights excluded from Git; verify every SHA-256."""

import argparse
import hashlib
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parent
MODELS = {
    "sepformer": {
        "repo": "speechbrain/sepformer-libri3mix",
        "revision": "1f2d824e2333aec9e60d0030fb144885c1afc426",
        "directory": ROOT / "evaluation/sepformer_libr3mix_checkpoint",
        "files": {
            "decoder.ckpt": "9afdda4e4ecb9eefb5d49329193c7a9215ae7443aab206f2fe4eb0f50988ca53",
            "encoder.ckpt": "e949e80c1a96965b723460209bb155eac1fdb2f6d2cf4e7b3e1d3152716f38d0",
            "masknet.ckpt": "416af7b09d912c486261dd824788dd92d3b21373c8ec4d029209792522f4dbcb",
        },
    },
    "ecapa": {
        "repo": "speechbrain/spkrec-ecapa-voxceleb",
        "revision": "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286",
        "directory": ROOT / "evaluation/ecapa_checkpoint",
        "files": {
            "mean_var_norm_emb.ckpt": "cd70225b05b37be64fc5a95e24395d804231d43f74b2e1e5a513db7b69b34c33",
            "embedding_model.ckpt": "0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2",
            "classifier.ckpt": "fd9e3634fe68bd0a427c95e354c0c677374f62b3f434e45b78599950d860d535",
        },
    },
    "speakerbeam": {
        "repo": "helloidea/OpenSpeakerBeam-SS-dataset",
        "repo_type": "dataset",
        "revision": "82e6d7646126d1fa33b30c0b44f1d076496cca71",
        "directory": ROOT / "speakerbeam_ss_verification/checkpoints",
        "files": {
            "best_model.pth": "41c250587af2e947dc3e265cd6e7aa7de230048d408e93c53d8348d63cf75896",
        },
        "subfolder": "checkpoints",
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_model(name, spec, offline_check):
    directory = spec["directory"]
    for filename, expected in spec["files"].items():
        destination = directory / filename
        if not destination.exists():
            if offline_check:
                raise FileNotFoundError(destination)
            directory.mkdir(parents=True, exist_ok=True)
            downloaded = Path(hf_hub_download(
                repo_id=spec["repo"],
                repo_type=spec.get("repo_type", "model"),
                revision=spec["revision"],
                filename=filename,
                subfolder=spec.get("subfolder"),
                local_dir=directory if "subfolder" not in spec else None,
            ))
            if spec.get("subfolder"):
                shutil.copyfile(downloaded, destination)
        actual = sha256(destination)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {destination}: {actual}")
        print(f"{name}: verified {destination.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speakerbeam", action="store_true",
                        help="also restore the optional OpenSpeakerBeam-SS baseline")
    parser.add_argument("--offline-check", action="store_true",
                        help="verify local files without network access")
    args = parser.parse_args()
    names = ["sepformer", "ecapa"] + (["speakerbeam"] if args.speakerbeam else [])
    for name in names:
        ensure_model(name, MODELS[name], args.offline_check)


if __name__ == "__main__":
    main()
