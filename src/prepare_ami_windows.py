"""Build cached AMI 30s pause-stitched windows (flac + trans)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from ami import SR, load_ami_ihm_test, stitch_pause_windows

OUT = Path("/workspace/SpeechPTQ/data/ami_ihm_stitch30")


def main():
    print("loading AMI IHM test chunks...")
    items = load_ami_ihm_test()
    print(f"chunks={len(items)}")
    windows = stitch_pause_windows(items, target_sec=30.0)
    print(f"windows={len(windows)}")
    OUT.mkdir(parents=True, exist_ok=True)
    by_meet: dict[str, list] = {}
    for w in windows:
        by_meet.setdefault(w["meeting_id"], []).append(w)
    n = 0
    audio_sec = 0.0
    for meet, rows in by_meet.items():
        d = OUT / meet
        d.mkdir(parents=True, exist_ok=True)
        lines = []
        for w in rows:
            uid = w["id"]
            path = d / f"{uid}.flac"
            sf.write(path, w["audio"], SR)
            text = " ".join(w["text"].split())
            lines.append(f"{uid} {text}")
            audio_sec += w["audio"].shape[0] / SR
            n += 1
        (d / f"{meet}.trans.txt").write_text("\n".join(lines) + "\n")
        print(f"  {meet}: {len(rows)} windows")
    print(f"wrote {n} windows, audio_sec={audio_sec:.1f} to {OUT}")


if __name__ == "__main__":
    main()
