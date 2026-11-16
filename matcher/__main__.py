"""python -m matcher RESUME.txt JOB.txt"""
from __future__ import annotations

import argparse
import pathlib

from .core import match


def main() -> None:
    ap = argparse.ArgumentParser(prog="matcher", description="Score a resume against a job description.")
    ap.add_argument("resume", type=pathlib.Path)
    ap.add_argument("job", type=pathlib.Path)
    args = ap.parse_args()
    m = match(args.resume.read_text(), args.job.read_text())

    print(f"Match score: {m.score}/100   (text similarity {m.text_similarity})\n")
    print(f"Required skills   ✓ {', '.join(m.required_hit) or '-'}")
    print(f"                  ✗ {', '.join(m.required_missing) or 'none missing'}")
    print(f"Preferred skills  ✓ {', '.join(m.preferred_hit) or '-'}")
    print(f"                  ✗ {', '.join(m.preferred_missing) or 'none missing'}")
    if m.required_missing:
        print("\nIf you have these, say so explicitly. If not, they're the gap to close first:")
        for s in m.required_missing:
            print(f"  - {s}")
    if m.unquantified_bullets:
        print(f"\n{len(m.unquantified_bullets)} bullet(s) have no numbers. Add scale or impact where you can:")
        for b in m.unquantified_bullets:
            print(f"  - {b[:90]}{'…' if len(b) > 90 else ''}")


if __name__ == "__main__":
    main()
