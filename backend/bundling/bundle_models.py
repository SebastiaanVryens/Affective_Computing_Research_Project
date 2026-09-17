"""Collect every model the sidecar needs into one directory, for shipping.

    python -m packaging.bundle_models
    python -m packaging.bundle_models --dest ../desktop/resources/models --no-voice

Nothing in app/ ships its weights. asr.py, text_model.py and audio_model.py all
call into the HuggingFace cache and download on first use — fine on a dev
machine where that happened months ago, fatal in a packaged app, where the
first launch would sit there silently pulling ~2GB before the voice channel
worked.

This walks the same model names the app reads from settings, downloads each
into a self-contained cache, and copies the local MELD checkpoint alongside.
Point HF_HOME at the result and set HF_HUB_OFFLINE=1 and the sidecar loads with
no network at all — which is also how you find out you missed one, because it
raises instead of quietly downloading.

Run this *before* packaging, on a machine with network. The output directory is
the thing the installer carries.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List

# Must be set before huggingface_hub is imported anywhere. The hub cache
# symlinks blobs into snapshots by default, which on Windows needs Developer
# Mode or admin and otherwise dies with WinError 1314 partway through a
# download. A bundle wants real files regardless — symlinks would not survive
# being packaged into an installer.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings  # noqa: E402

# faster-whisper resolves a bare size like "base.en" to this repo. Mirrors the
# mapping inside faster_whisper.utils; if that ever changes, a bundled build
# fails loudly at startup rather than silently downloading, which is the point.
WHISPER_REPO = "Systran/faster-whisper-{size}"


def whisper_repo_id(model: str) -> str:
    """Map a faster-whisper size to its hub repo, passing through full ids."""
    return model if "/" in model else WHISPER_REPO.format(size=model)


#: Metadata and tokeniser files, small and always wanted.
SUPPORT_PATTERNS = ["*.json", "*.txt", "*.model", "*.tiktoken"]


def weight_patterns(repo_id: str) -> List[str]:
    """Pick one weight format per repo instead of taking every copy.

    Most hub repos carry the same tensors twice — `pytorch_model.bin` and
    `model.safetensors`. Downloading both doubled the wav2vec2 head to 2.4GB
    for no benefit, since transformers loads whichever it finds and prefers
    safetensors anyway.

    `model.bin` is a separate case and must always come through: for
    faster-whisper that name is a CTranslate2 model, not a torch pickle, and
    it is the only weights file in the repo.
    """
    from huggingface_hub import HfApi

    files = set(HfApi().list_repo_files(repo_id))
    has_safetensors = any(f.endswith(".safetensors") for f in files)

    if has_safetensors:
        # Skip torch pickles; keep model.bin, which is not one.
        return ["*.safetensors", "model.bin"]
    return ["*.bin"]


def fetch(repo_id: str, cache_dir: Path) -> bool:
    """Download one repo into cache_dir. Returns False on failure."""
    from huggingface_hub import snapshot_download

    print(f"  {repo_id}")
    try:
        patterns = SUPPORT_PATTERNS + weight_patterns(repo_id)
        snapshot_download(
            repo_id=repo_id,
            cache_dir=str(cache_dir),
            allow_patterns=patterns,
        )
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"    !! failed: {exc}")
        return False


def directory_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024 / 1024


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "models" / "bundle",
        help="where to write the self-contained cache",
    )
    parser.add_argument(
        "--no-voice",
        action="store_true",
        help="skip the ~1.2GB wav2vec2 SER head; the app falls back to the "
        "prosody heuristic, which is what MINDSCAPE_VOICE_EMOTION=false does",
    )
    args = parser.parse_args()

    hub = args.dest / "hub"
    hub.mkdir(parents=True, exist_ok=True)

    repos = [
        whisper_repo_id(settings.whisper_model),
        settings.fallback_text_model,
    ]
    if not args.no_voice:
        repos.append(settings.voice_emotion_model)

    print(f"Destination: {args.dest}")
    print(f"Downloading {len(repos)} repos into the bundle cache:")

    failures = [r for r in repos if not fetch(r, hub)]

    # The MELD checkpoint is not on the hub - train_text.py wrote it locally,
    # and it is the model the evaluation actually used. Without it the app
    # silently falls back to the general head (see config.py), which would ship
    # a different system than the one in the report.
    meld_src = settings.meld_text_model
    meld_dest = args.dest / "meld-text"
    if meld_src.is_dir():
        print(f"\nCopying MELD checkpoint from {meld_src}")
        if meld_dest.exists():
            shutil.rmtree(meld_dest)
        shutil.copytree(meld_src, meld_dest)
    else:
        print(f"\n  !! no MELD checkpoint at {meld_src}")
        print("     The packaged app will fall back to the general text model.")
        print("     Run `python -m training.train_text` first if that is not what you want.")
        failures.append("meld-text")

    print(f"\nBundle size: {directory_size_mb(args.dest):.0f} MB")

    if failures:
        print(f"\nIncomplete - {len(failures)} missing: {', '.join(failures)}")
        raise SystemExit(1)

    print("\nComplete. To verify it loads with no network:")
    print(f"  HF_HOME={args.dest} HF_HUB_OFFLINE=1 MINDSCAPE_MODEL_BUNDLE={args.dest} \\")
    print("    python -m uvicorn app.main:app --port 8000")


if __name__ == "__main__":
    main()
