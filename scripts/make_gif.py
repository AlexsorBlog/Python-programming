"""Screenshots of the running service, glued into docs/search.gif.

    findex serve --index data/index.pkl  (with EMBEDDINGS_PATH set)
    uv run --group bench python scripts/make_gif.py http://127.0.0.1:8000

Real pages of the real app, taken by headless Edge one after another.
"""

import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlencode

from PIL import Image

EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
SIZE = (1100, 820)

# (path, query params, how long the frame stays, ms)
FRAMES: list[tuple[str, dict[str, str], int]] = [
    ("/", {}, 1600),
    ("/", {"q": "chernobyl disaster"}, 2600),
    ("/", {"q": "kyiv AND (ukraine OR dnipro) NOT river"}, 2600),
    ("/", {"q": "worst nuclear accident in history", "mode": "semantic"}, 3000),
    ("/", {"q": "worst nuclear accident in history", "mode": "hybrid"}, 3000),
    ("/doc/7283", {"q": "chernobyl disaster"}, 2400),
]


def shot(url: str, out: Path) -> None:
    subprocess.run(
        [
            str(EDGE),
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--force-color-profile=srgb",
            f"--window-size={SIZE[0]},{SIZE[1]}",
            f"--screenshot={out}",
            url,
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    base = sys.argv[1].rstrip("/")
    out = Path("docs/search.gif")
    frames: list[Image.Image] = []
    durations: list[int] = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, (path, params, ms) in enumerate(FRAMES):
            url = base + path + (f"?{urlencode(params)}" if params else "")
            png = Path(tmp) / f"{i}.png"
            shot(url, png)
            frame = Image.open(png).convert("RGB").resize((880, 656), Image.LANCZOS)
            frames.append(frame.quantize(colors=128, method=Image.Quantize.MEDIANCUT))
            durations.append(ms)
            print(f"кадр {i + 1}/{len(FRAMES)}: {url}", flush=True)

    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    print(f"{out}: {out.stat().st_size / 1024:.0f} КБ")


if __name__ == "__main__":
    main()
