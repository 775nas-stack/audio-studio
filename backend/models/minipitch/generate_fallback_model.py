"""Utility to create randomly initialized TinyMiniPitchNet weights."""
from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file

from backend.models.minipitch.model_definition import MODEL_FILENAME, TinyMiniPitchNet


def main() -> None:
    model = TinyMiniPitchNet()
    state_dict = model.state_dict()
    model_dir = Path(__file__).parent
    output_path = model_dir / MODEL_FILENAME
    # Ensure directory exists (it should, but be defensive for manual runs).
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, str(output_path))
    print("Fallback model generated successfully.")


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
