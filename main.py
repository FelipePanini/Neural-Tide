"""Neural Tide — interactive generative art driven by your hands.

    python main.py            # webcam
    python main.py --mock     # the mouse becomes a hand (no webcam needed)
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace

from config import CONFIG, Config

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mock", action="store_true", help="use the mouse as a hand instead of the webcam")
    parser.add_argument("--camera", type=int, metavar="N", help="webcam index (default: config.py)")
    parser.add_argument("--size", metavar="WxH", help="window size, e.g. 1920x1080")
    parser.add_argument("--fullscreen", action="store_true", help="start in fullscreen")
    parser.add_argument("--no-debug", action="store_true", help="start with the debug overlay hidden")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace, base: Config = CONFIG) -> Config:
    """Apply command-line overrides on top of ``config.py``."""
    cfg = base
    if args.camera is not None:
        cfg = replace(cfg, camera=replace(cfg.camera, index=args.camera))
    if args.size:
        width, height = (int(v) for v in args.size.lower().split("x"))
        cfg = replace(cfg, window=replace(cfg.window, width=width, height=height))
    if args.fullscreen:
        cfg = replace(cfg, window=replace(cfg.window, fullscreen=True))
    if args.no_debug:
        cfg = replace(cfg, debug=replace(cfg.debug, start_visible=False))
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = build_config(args)
    if not args.mock and not cfg.models.hand_landmarker.path.exists():
        print("Modelo do MediaPipe não encontrado. Rode primeiro:\n    python tools/download_model.py", file=sys.stderr)
        return 1

    from app import App  # imported late so --help stays instant

    print("Neural Tide — F1 debug · C fundo · ESC sair" + ("  (modo mock: mouse = mão)" if args.mock else ""))
    App(cfg, mock=args.mock).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
