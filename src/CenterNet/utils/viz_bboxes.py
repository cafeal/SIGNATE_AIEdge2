import numpy as np
from PIL import Image
from PIL import ImageDraw


def viz_bboxes(img, bboxes, categories, color, alpha=None, width=3):
    pil_img = Image.fromarray(img).convert("RGBA")
    boxes = Image.new("RGBA", pil_img.size, color=0)
    draw = ImageDraw.Draw(boxes)

    if alpha is None:
        alpha = np.ones(len(categories))
    alpha = (alpha * 255).astype(np.uint8)

    order = np.argsort(alpha)
    bboxes = [bboxes[i] for i in order]
    categories = [categories[i] for i in order]
    alpha = [alpha[i] for i in order]

    for b, c, a in zip(bboxes, categories, alpha):
        x1, y1, x2, y2 = b
        _color = color + (a,)  # RGBA
        draw.rectangle((x1, y1, x2, y2), outline=_color, width=width)
        draw.text((x1, y1 - 10), c, fill=_color)

    pil_img = Image.alpha_composite(pil_img, boxes).convert("RGB")

    return np.array(pil_img)
