"""Generate the activation-function SVG used by the CNN study note."""

from math import exp, tanh
from pathlib import Path


WIDTH, HEIGHT = 760, 480
LEFT, RIGHT, TOP, BOTTOM = 70, 710, 55, 400


def x_map(x: float) -> float:
    return LEFT + (x + 4) / 8 * (RIGHT - LEFT)


def y_map(y: float) -> float:
    return TOP + (1.2 - y) / 2.4 * (BOTTOM - TOP)


def path_for(fn) -> str:
    points = []
    for i in range(161):
        x = -4 + i * 8 / 160
        points.append((x_map(x), y_map(fn(x))))
    command = f"M {points[0][0]:.2f} {points[0][1]:.2f}"
    return command + " " + " ".join(
        f"L {x:.2f} {y:.2f}" for x, y in points[1:]
    )


def sigmoid(x: float) -> float:
    return 1 / (1 + exp(-x))


def relu(x: float) -> float:
    return max(0, x)


def build_svg() -> str:
    grid = []
    for y in range(55, 401, 57):
        grid.append(f'<line class="grid" x1="70" y1="{y}" x2="710" y2="{y}"/>')
    for x in range(70, 711, 80):
        grid.append(f'<line class="grid" x1="{x}" y1="55" x2="{x}" y2="400"/>')
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">
  <title id="title">Activation functions: Sigmoid, Tanh, and ReLU</title>
  <desc id="desc">Continuous curves comparing Sigmoid, Tanh, and ReLU.</desc>
  <rect width="100%" height="100%" fill="#ffffff"/>
  <style>
    .grid {{ stroke: #e5e7eb; stroke-width: 1; }}
    .axis {{ stroke: #374151; stroke-width: 1.5; }}
    .tick {{ fill: #6b7280; font: 12px system-ui, sans-serif; }}
    .label {{ fill: #374151; font: 14px system-ui, sans-serif; }}
    .legend {{ fill: #111827; font: 14px system-ui, sans-serif; }}
  </style>
  {''.join(grid)}
  <line class="axis" x1="70" y1="227.5" x2="710" y2="227.5"/>
  <line class="axis" x1="390" y1="55" x2="390" y2="400"/>
  <path d="{path_for(sigmoid)}" fill="none" stroke="#2563eb" stroke-width="3.5" stroke-linecap="round"/>
  <path d="{path_for(tanh)}" fill="none" stroke="#dc2626" stroke-width="3.5" stroke-linecap="round"/>
  <path d="{path_for(relu)}" fill="none" stroke="#059669" stroke-width="3.5" stroke-linecap="round"/>
  <g class="tick">
    <text x="58" y="231.5" text-anchor="end">0</text>
    <text x="70" y="420" text-anchor="middle">-4</text>
    <text x="230" y="420" text-anchor="middle">-2</text>
    <text x="390" y="420" text-anchor="middle">0</text>
    <text x="550" y="420" text-anchor="middle">2</text>
    <text x="710" y="420" text-anchor="middle">4</text>
  </g>
  <text class="label" x="390" y="452" text-anchor="middle">input z</text>
  <text class="label" x="16" y="230" text-anchor="middle" transform="rotate(-90 16 230)">output f(z)</text>
  <g transform="translate(92 22)">
    <line x1="0" y1="0" x2="27" y2="0" stroke="#2563eb" stroke-width="3.5"/><text class="legend" x="35" y="5">Sigmoid</text>
    <line x1="115" y1="0" x2="142" y2="0" stroke="#dc2626" stroke-width="3.5"/><text class="legend" x="150" y="5">Tanh</text>
    <line x1="220" y1="0" x2="247" y2="0" stroke="#059669" stroke-width="3.5"/><text class="legend" x="255" y="5">ReLU</text>
  </g>
</svg>
'''


if __name__ == "__main__":
    Path(__file__).with_name("activation-functions.svg").write_text(build_svg(), encoding="utf-8")
