"""Build a compact, browsable review bundle from completed handoff outputs."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


COLORS = {
    "gt": (255, 45, 35),
    "native": (30, 210, 80),
    "direct": (225, 55, 190),
    "hybrid": (45, 135, 255),
}


def overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    result = image.astype(np.float32).copy()
    result[mask] = result[mask] * 0.4 + np.asarray(color) * 0.6
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def panel(image: Image.Image, title: str, score: float | None) -> Image.Image:
    result = Image.new("RGB", (image.width, image.height + 34), "white")
    result.paste(image, (0, 34))
    label = title if score is None else f"{title}   GT J&F {score:.3f}"
    ImageDraw.Draw(result).text((8, 10), label, fill="black")
    return result


def load_scores(path: Path) -> dict[int, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(row["frame"]): float(row["J_and_F"]) for row in payload["frames"]}


def mask(path: Path, object_id: int | None = None) -> np.ndarray:
    array = np.asarray(Image.open(path))
    return array == object_id if object_id is not None else array > 0


def write_chart(series: dict[str, dict[int, float]], output: Path) -> None:
    width, height, margin = 1100, 420, 55
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.line((margin, margin, margin, height - margin), fill=(80, 80, 80), width=2)
    draw.line((margin, height - margin, width - margin, height - margin), fill=(80, 80, 80), width=2)
    frames = sorted(set().union(*(values.keys() for values in series.values())))
    palette = {"Large-native": COLORS["native"], "Direct": COLORS["direct"], "Hybrid": COLORS["hybrid"]}
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = height - margin - tick * (height - 2 * margin)
        draw.line((margin, y, width - margin, y), fill=(225, 225, 225), width=1)
        draw.text((10, y - 7), f"{tick:.2f}", fill="black")
    for name, values in series.items():
        points = []
        for frame in frames:
            if frame not in values:
                continue
            x = margin + (frame - frames[0]) / (frames[-1] - frames[0]) * (width - 2 * margin)
            y = height - margin - values[frame] * (height - 2 * margin)
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=palette[name], width=3)
    x = margin
    for name in series:
        draw.rectangle((x, 12, x + 20, 27), fill=palette[name])
        draw.text((x + 28, 13), name, fill="black")
        x += 180
    draw.text((width - 170, height - 35), "Frame", fill="black")
    draw.text((10, 30), "J&F", fill="black")
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument("--annotation-dir", required=True, type=Path)
    parser.add_argument("--native-mask-dir", required=True, type=Path)
    parser.add_argument("--direct-mask-dir", required=True, type=Path)
    parser.add_argument("--hybrid-mask-dir", required=True, type=Path)
    parser.add_argument("--native-metrics", required=True, type=Path)
    parser.add_argument("--direct-metrics", required=True, type=Path)
    parser.add_argument("--hybrid-metrics", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--object-id", type=int, default=1)
    parser.add_argument("--start-frame", type=int, default=7)
    parser.add_argument("--end-frame", type=int, default=89)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = args.output_dir / "frames"
    frame_dir.mkdir(exist_ok=True)
    scores = {
        "Large-native": load_scores(args.native_metrics),
        "Direct": load_scores(args.direct_metrics),
        "Hybrid": load_scores(args.hybrid_metrics),
    }
    records = []
    for frame in range(args.start_frame, args.end_frame + 1):
        stem = f"{frame:05d}"
        rgb = np.asarray(Image.open(args.video_dir / f"{stem}.jpg").convert("RGB"))
        gt = mask(args.annotation_dir / f"{stem}.png", args.object_id)
        native = mask(args.native_mask_dir / f"{stem}.png")
        direct = mask(args.direct_mask_dir / f"{stem}.png")
        hybrid = mask(args.hybrid_mask_dir / f"{stem}.png")
        panels = [
            panel(overlay(rgb, gt, COLORS["gt"]), "DAVIS GT", None),
            panel(overlay(rgb, native, COLORS["native"]), "Large-native", scores["Large-native"].get(frame)),
            panel(overlay(rgb, direct, COLORS["direct"]), "Direct", scores["Direct"].get(frame)),
            panel(overlay(rgb, hybrid, COLORS["hybrid"]), "Ridge hybrid", scores["Hybrid"].get(frame)),
        ]
        canvas = Image.new("RGB", (sum(p.width for p in panels), panels[0].height), "white")
        x = 0
        for item in panels:
            canvas.paste(item, (x, 0))
            x += item.width
        filename = f"frame_{stem}.jpg"
        canvas.save(frame_dir / filename, quality=80, optimize=True)
        records.append({"frame": frame, "image": f"frames/{filename}", **{name: values.get(frame) for name, values in scores.items()}})

    write_chart(scores, args.output_dir / "jf_by_frame.png")
    (args.output_dir / "frames.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    html = """<!doctype html><meta charset='utf-8'><title>CMMT handoff review</title>
<style>body{font:16px system-ui;margin:24px;background:#111;color:#eee}main{max-width:1500px;margin:auto}img{max-width:100%;display:block;margin:16px 0}input{width:100%}.scores{font-family:monospace}.note{color:#bbb}</style>
<main><h1>CMMT: bmx-bumps handoff review</h1><p class='note'>Red=GT, green=Large-native, pink=Direct, blue=Ridge hybrid. Frame 89 is visual-only and excluded from official aggregate metrics.</p>
<video controls loop muted style='width:100%' src='handoff_review.mp4'></video><img src='jf_by_frame.png' alt='J&F by frame'>
<h2 id='frame-title'></h2><input id='slider' type='range' min='0' value='0'><div class='scores' id='scores'></div><img id='frame-image'>
<p class='note'>Dataset: <a href='https://davischallenge.org/'>DAVIS 2017</a>, licensed <a href='https://creativecommons.org/licenses/by-nc/4.0/'>CC BY-NC 4.0</a>. Used for non-commercial research evaluation; please cite the DAVIS benchmark papers.</p>
<script>const rows=__FRAME_ROWS__;const s=document.querySelector('#slider'),im=document.querySelector('#frame-image'),t=document.querySelector('#frame-title'),v=document.querySelector('#scores');s.max=rows.length-1;function show(){const r=rows[+s.value];t.textContent=`Frame ${r.frame}`;im.src=r.image;v.textContent=`Large-native ${fmt(r['Large-native'])} | Direct ${fmt(r.Direct)} | Hybrid ${fmt(r.Hybrid)}`};function fmt(x){return x==null?'visual only':x.toFixed(3)}s.oninput=show;show()</script></main>"""
    html = html.replace("__FRAME_ROWS__", json.dumps(records, separators=(",", ":")))
    (args.output_dir / "index.html").write_text(html, encoding="utf-8")
    subprocess.run([
        "ffmpeg", "-y", "-framerate", "6", "-start_number", str(args.start_frame),
        "-i", str(frame_dir / "frame_%05d.jpg"), "-vf", "scale=1920:-2",
        "-c:v", "libx264", "-crf", "25", "-pix_fmt", "yuv420p",
        str(args.output_dir / "handoff_review.mp4"),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(json.dumps({"frames": len(records), "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
