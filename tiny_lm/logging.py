"""Lightweight CSV and SVG experiment logging."""
from __future__ import annotations
import csv
from pathlib import Path

def write_history_csv(path: str | Path, history: list[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["step", "train_loss", "validation_loss", "learning_rate", "elapsed_seconds"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in history)

def write_loss_curve_svg(path: str | Path, history: list[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    points = [(float(row["step"]), float(row["train_loss"])) for row in history]
    validation = [(float(row["step"]), float(row["validation_loss"])) for row in history if row.get("validation_loss") is not None]
    all_values = [value for _, value in points + validation]
    if not all_values:
        raise ValueError("history must contain loss values")
    width, height, margin = 800, 480, 50
    minimum, maximum = min(all_values), max(all_values)
    span = max(maximum - minimum, 1e-8)
    max_step = max(step for step, _ in points) or 1
    def map_point(step: float, value: float) -> str:
        x = margin + (step / max_step) * (width - 2 * margin)
        y = height - margin - ((value - minimum) / span) * (height - 2 * margin)
        return f"{x:.1f},{y:.1f}"
    train_points = " ".join(map_point(step, value) for step, value in points)
    validation_points = " ".join(map_point(step, value) for step, value in validation)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/><line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="black"/><line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="black"/>
<polyline fill="none" stroke="#2563eb" stroke-width="2" points="{train_points}"/><polyline fill="none" stroke="#dc2626" stroke-width="2" points="{validation_points}"/>
<text x="{margin}" y="25" font-family="sans-serif" font-size="16">TinyLM Loss Curve</text><text x="{width-180}" y="30" fill="#2563eb" font-family="sans-serif">train</text><text x="{width-100}" y="30" fill="#dc2626" font-family="sans-serif">validation</text>
</svg>'''
    output.write_text(svg, encoding="utf-8")