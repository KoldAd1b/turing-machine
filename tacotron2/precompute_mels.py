import argparse
import csv
import os
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from dataset import AudioMelConversions, load_wav


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_manifest", type=str, nargs="+", required=True)
    parser.add_argument("--path_to_save", type=str, required=True)
    parser.add_argument("--output_suffix", type=str, default="_mels")
    parser.add_argument("--sampling_rate", type=int, default=22050)
    parser.add_argument("--num_mels", type=int, default=80)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--window_size", type=int, default=1024)
    parser.add_argument("--hop_size", type=int, default=256)
    parser.add_argument("--min_db", type=float, default=-100.0)
    parser.add_argument("--max_scaled_abs", type=float, default=4.0)
    parser.add_argument("--fmin", type=int, default=0)
    parser.add_argument("--fmax", type=int, default=8000)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    save_dir = Path(args.path_to_save)
    save_dir.mkdir(parents=True, exist_ok=True)

    converter = AudioMelConversions(
        num_mels=args.num_mels,
        sampling_rate=args.sampling_rate,
        n_fft=args.n_fft,
        window_size=args.window_size,
        hop_size=args.hop_size,
        fmin=args.fmin,
        fmax=args.fmax,
        min_db=args.min_db,
        max_scaled_abs=args.max_scaled_abs,
    )
    target_dtype = torch.float16 if args.dtype == "float16" else torch.float32

    for manifest in args.path_to_manifest:
        manifest_path = Path(manifest)
        metadata = pd.read_csv(manifest_path)
        mel_paths = []

        for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc=manifest_path.name):
            audio_path = row["file_path"]
            file_stem = Path(audio_path).stem
            mel_path = save_dir / f"{file_stem}.pt"

            if args.overwrite or not mel_path.exists():
                audio = load_wav(audio_path, sr=args.sampling_rate)
                mel = converter.audio2mel(audio, do_norm=True).squeeze(0).to(target_dtype).cpu()
                torch.save(mel, mel_path)

            mel_paths.append(str(mel_path))

        metadata["mel_path"] = mel_paths
        out_path = manifest_path.with_name(f"{manifest_path.stem}{args.output_suffix}{manifest_path.suffix}")
        metadata.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
