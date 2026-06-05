import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import torchaudio
from tqdm import tqdm


SPLITS = [
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
    "dev-clean",
    "test-clean",
]


def iter_audio_files(path_to_librispeech_data, splits):
    for split in splits:
        path_to_split = os.path.join(path_to_librispeech_data, split)
        if not os.path.isdir(path_to_split):
            continue

        for speaker in os.listdir(path_to_split):
            path_to_speaker = os.path.join(path_to_split, speaker)
            if not os.path.isdir(path_to_speaker):
                continue

            for chapter in os.listdir(path_to_speaker):
                path_to_chapter = os.path.join(path_to_speaker, chapter)
                if not os.path.isdir(path_to_chapter):
                    continue

                for file_name in os.listdir(path_to_chapter):
                    if file_name.endswith(".flac"):
                        yield os.path.join(path_to_chapter, file_name)


def validate_audio_file(args):
    path, num_frames = args
    try:
        try:
            waveform, sample_rate = torchaudio.load(path, num_frames=num_frames)
        except Exception:
            waveform, sample_rate = torchaudio.load(path)
            waveform = waveform[..., :num_frames]
        if waveform.numel() == 0:
            return path, "empty waveform"
        if sample_rate <= 0:
            return path, f"invalid sample rate: {sample_rate}"
    except Exception as error:
        return path, repr(error)

    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Validate LibriSpeech FLAC files can be loaded by torchaudio.")
    parser.add_argument("--path_to_librispeech_data", default="../../data/LibriSpeech")
    parser.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    parser.add_argument("--max_audio_duration", default=15.0, type=float)
    parser.add_argument("--sampling_rate", default=16000, type=int)
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument("--error_log", default="bad_librispeech_audio.txt")
    return parser.parse_args()


def main():
    args = parse_args()
    num_frames = int(args.max_audio_duration * args.sampling_rate)
    audio_files = list(iter_audio_files(args.path_to_librispeech_data, args.splits))
    failures = []

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(validate_audio_file, (path, num_frames)) for path in audio_files]
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result is not None:
                failures.append(result)

    if failures:
        with open(args.error_log, "w") as f:
            for path, error in failures:
                f.write(f"{path}\t{error}\n")
        print(f"Found {len(failures)} bad audio files. Wrote {args.error_log}")
        raise SystemExit(1)

    print(f"Validated {len(audio_files)} audio files with no failures.")


if __name__ == "__main__":
    main()
