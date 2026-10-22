# FIBO 2 support for Forge Neo
# The prompt as FIBO 2.0 was trained on it, and the pictures Qwen3-VL sees, ported from ComfyUI's
# comfy/text_encoders/fibo2.py and comfy_extras/nodes_fibo2.py (COMFYUI_COMMIT in native/__init__.py), which follow
# Bria's reference pipeline.
#
# A prompt is a structured JSON caption (Bria's VGL) or plain text. The caption is cleaned as in training (empty
# fields and scores dropped) and re-serialized compactly. Each text_render entry gets a marker where Qwen3-VL sees a
# picture of its text, drawn black on white. An edit wraps the caption, or the plain text as the instruction, in
# {"image": <marker>, ..., "structured_caption": ..., "edit_instruction": ...}. An empty prompt is the null caption
# training used for classifier-free guidance.
#
# No Forge imports: the CPU tests run it as is.

import json
import math

import numpy as np
import torch

VISION_MARKER = "<|vision_start|><|image_pad|><|vision_end|>"
TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
# the null caption, a whole chat turn, taken as it is (no cleaning, no template)
NULL_CAPTION = TEMPLATE.format('{"short_description":"","objects":[],"background_setting":"","lighting":{},'
                               '"aesthetics":{},"photographic_characteristics":{},"style_medium":"","context":"",'
                               '"artistic_style":""}')
SCORES = ("aesthetic_score", "preference_score")
# Qwen3-VL sees each image to edit at most this many pixels a side, as in training
THUMBNAIL = 768
FONT_SIZE = 20

_font = None


def dumps(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def empty(value):
    if isinstance(value, float):
        return math.isnan(value)
    return value is None or (isinstance(value, (str, dict, list)) and not value)


def clean(value):
    if isinstance(value, dict):
        value = {k: clean(v) for k, v in value.items()}
        return {k: v for k, v in value.items() if not empty(v)}
    if isinstance(value, list):
        return [v for v in map(clean, value) if not empty(v)]
    return value


def parse_prompt(prompt):
    """(caption, edit instruction): the caption as training saw it; a prompt that isn't a JSON object comes back as
    it is, with no instruction."""
    try:
        caption = json.loads(prompt)
    except json.JSONDecodeError:
        return prompt, None
    if not isinstance(caption, dict):
        return prompt, None
    instruction = caption.pop("edit_instruction", None)
    if "structured_caption" in caption:
        caption = caption.pop("structured_caption")
        if not isinstance(caption, dict):
            return caption or None, instruction

    keys = set(caption)
    caption = clean({key: {k: v for k, v in value.items() if k not in SCORES} if isinstance(value, dict) else value
                     for key, value in caption.items() if key not in SCORES})
    # exactly a short_description and a text_render list keeps both, the list even when empty: the image has no text
    if keys == {"short_description", "text_render"}:
        caption = {"short_description": caption.get("short_description") or "",
                   "text_render": caption.get("text_render") or []}
    return caption or None, instruction


def build(prompt: str, edit_images: int = 0, text_images: bool = True) -> tuple[str, list[str]]:
    """The chat turn for a prompt, and the texts to draw as pictures, in the order of their markers after the
    edit images'. text_images=False leaves each text entry's marker empty: the caption a generation starts from."""
    if prompt.strip() == "":
        return NULL_CAPTION, []
    caption, instruction = parse_prompt(prompt)
    if edit_images > 0 and instruction is None:
        caption, instruction = None, prompt
    entries = []
    if isinstance(caption, dict):
        entries = [e for e in caption.get("text_render") or []
                   if isinstance(e, dict) and isinstance(e.get("text"), str) and e["text"]]
    for entry in entries:
        entry["image_tokens"] = VISION_MARKER if text_images else ""

    if edit_images > 0:
        edit = ({"image": VISION_MARKER} if edit_images == 1
                else {f"image_{k}": VISION_MARKER for k in range(1, edit_images + 1)})
        if caption:
            edit["structured_caption"] = caption
        edit["edit_instruction"] = instruction
        text = dumps(edit)
    elif not isinstance(caption, str):
        text = dumps(caption or {})
    else:
        text = caption
    texts = [entry["text"] for entry in entries] if text_images else []
    return (text if text.startswith("<|im_start|>") else TEMPLATE.format(text)), texts


def has_text_render(prompt: str) -> bool:
    return bool(build(prompt, text_images=True)[1])


# pictures: (1, H, W, 3) float tensors in [0, 1], as ComfyUI's images


def font():
    global _font
    if _font is None:
        from PIL import ImageFont

        _font = ImageFont.load_default(FONT_SIZE)
    return _font


def has_glyph(font, notdef, character):
    # a font without the glyph draws its notdef box, the one it draws for U+FFFE, which no font has
    try:
        mask = font.getmask(character)
    except (OSError, ValueError):  # as in training, a font that can't say is taken to have the glyph
        return True
    return not (mask.size == notdef.size and bytes(mask) == bytes(notdef))


def draw_text_image(text, font):
    """Pillow's built-in font, black on white, whitespace collapsed, split every 256 characters (training's non-ASCII
    fallback fonts aren't shipped)."""
    from PIL import Image, ImageDraw

    line = " ".join(text.split())
    lines = [line[i:i + 256] for i in range(0, len(line), 256)] or [""]
    notdef = font.getmask(chr(0xFFFE))
    if all(ord(c) < 128 or has_glyph(font, notdef, c) for c in set(line)):
        # each line placed by its bounding box, at least as tall as "Ag"
        min_box = font.getbbox("Ag")
        boxes = [font.getbbox(line or " ") for line in lines]
        heights = [max(1, min_box[3] - min_box[1], box[3] - box[1]) for box in boxes]
        width = max([1] + [box[2] - box[0] for box in boxes])
        origins, anchor = [(2 - box[0], -box[1]) for box in boxes], None
    else:
        # a character the font lacks (drawn as its notdef box) puts every line on the baseline, as tall as the font
        ascent, descent = font.getmetrics()
        lines = [line or " " for line in lines]
        heights = [max(1, ascent + descent)] * len(lines)
        width = max(max(1, round(font.getlength(line))) for line in lines)
        origins, anchor = [(2.0, ascent)] * len(lines), "ls"

    image = Image.new("RGB", (width + 4, sum(heights) + 4 * len(lines)), "white")
    draw = ImageDraw.Draw(image)
    y = 2
    for line, (x, dy), height in zip(lines, origins, heights):
        draw.text((x, y + dy), line, font=font, fill=(0, 0, 0), anchor=anchor)
        y += height + 4
    return torch.from_numpy(np.array(image)).float().div(255.0).unsqueeze(0)


def resize_8bit(image, height, width):
    # antialiased bicubic rounded to 8 bit, as PIL and Qwen3-VL's processor resize
    image = torch.nn.functional.interpolate(image.movedim(-1, 1), size=(height, width), mode="bicubic", antialias=True)
    return image.mul(255.0).round().clamp(0, 255).div(255.0).movedim(1, -1)


def thumbnail(image, size=THUMBNAIL):
    scale = size / max(image.shape[1], image.shape[2])
    if scale >= 1:
        return image
    return resize_8bit(image, max(1, round(image.shape[1] * scale)), max(1, round(image.shape[2] * scale)))


def to_qwen_grid(image):
    # the processor's 32 pixel grid, so Forge's own bilinear resize to it is a no-op
    h, w = image.shape[1], image.shape[2]
    height, width = round(h / 32) * 32, round(w / 32) * 32
    if height * width > 1605632:
        beta = math.sqrt(h * w / 1605632)
        height, width = max(32, math.floor(h / beta / 32) * 32), max(32, math.floor(w / beta / 32) * 32)
    elif height * width < 3136:
        beta = math.sqrt(3136 / (h * w))
        height, width = math.ceil(h * beta / 32) * 32, math.ceil(w * beta / 32) * 32
    return resize_8bit(image, height, width)


def trained_size(aspect_ratio):
    """Bria's (width, height) for an aspect ratio: of the multiples of 16 within 10% of 1 MP, the least cropping,
    then the closest to 1 MP."""
    area = 1024 * 1024
    sizes = [(w, h) for w in range(64, 2 * area // 64, 16)
             for h in range(max(64, 16 * math.ceil(0.9 * area / w / 16)), 16 * math.floor(1.1 * area / w / 16) + 1, 16)]
    return min(sizes, key=lambda s: (round(1 - min(s[0] / s[1] / aspect_ratio, aspect_ratio * s[1] / s[0]), 9),
                                     abs(s[0] * s[1] - area), -s[0]))


def fit(image, width, height):
    """Scaled and center cropped to width x height with Lanczos, as ComfyUI's common_upscale(..., "lanczos",
    "center"); unchanged when it already is that size."""
    if (image.shape[2], image.shape[1]) == (width, height):
        return image
    from PIL import Image

    old_height, old_width = image.shape[1], image.shape[2]
    old_aspect, new_aspect = old_width / old_height, width / height
    x = y = 0
    if old_aspect > new_aspect:
        x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
    elif old_aspect < new_aspect:
        y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
    cropped = image[0, y:old_height - y, x:old_width - x]
    pil = Image.fromarray(np.clip(255.0 * cropped.cpu().numpy(), 0, 255).astype(np.uint8))
    pil = pil.resize((width, height), resample=Image.Resampling.LANCZOS)
    return torch.from_numpy(np.array(pil).astype(np.float32) / 255.0).unsqueeze(0)


def edit_images(images, width=None, height=None):
    """The images to edit at the sizes upstream gives them: a single one scaled and center cropped to the result's
    size (multiples of 16), since the edit lines up with it pixel for pixel; several, each to its own trained size."""
    images = [image[..., :3] for image in images]
    if len(images) == 1:
        h, w = images[0].shape[1], images[0].shape[2]
        width, height = width or w, height or h
        return [fit(images[0], max(16, width // 16 * 16), max(16, height // 16 * 16))]
    return [fit(image, *trained_size(image.shape[2] / image.shape[1])) for image in images]
