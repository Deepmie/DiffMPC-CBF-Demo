from pathlib import Path

from PIL import Image, ImageDraw


source = Path("imgs/tvto_0.gif")
target = Path("cursor-try/tvto_0_contact.png")

with Image.open(source) as gif:
    count = getattr(gif, "n_frames", 1)
    indices = sorted(set(round(i * (count - 1) / 11) for i in range(12)))
    frames = []
    for index in indices:
        gif.seek(index)
        frame = gif.convert("RGB").resize((400, 240))
        ImageDraw.Draw(frame).text((8, 8), f"frame {index}/{count - 1}", fill="black")
        frames.append(frame)

sheet = Image.new("RGB", (4 * 400, 3 * 240), "white")
for i, frame in enumerate(frames):
    sheet.paste(frame, ((i % 4) * 400, (i // 4) * 240))
sheet.save(target)
print(f"{count} frames -> {target}")
