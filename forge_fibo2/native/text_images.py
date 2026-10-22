# FIBO 2 support for Forge Neo
# The pictures of a caption's text_render texts, drawn the way Fibo 2.0's training drew them: each text on its own,
# in one line, black on white. Pillow's built-in font draws the text; a character it lacks (é, ã, ç and every other
# accented letter among them) is drawn with the first fallback font that has it: Aileron, the full version of the
# built-in font, then DejaVu Sans, both shipped in fonts/. Ported from diffusers' BriaFibo2Pipeline
# (src/diffusers/pipelines/bria_fibo2/text_images.py, DIFFUSERS_COMMIT in native/__init__.py), Copyright 2026 Bria AI
# and The HuggingFace Team, Apache License 2.0. ComfyUI draws with the built-in font only.
#
# No Forge imports: the CPU tests run it as is.

import functools
import os

FONT_SIZE = 20
# texts longer than this are split into lines of it
MAX_LINE_CHARS = 256
PADDING = 2
LINE_SPACING = 4
FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
FALLBACK_FONTS = ("Aileron-Regular.otf", "DejaVuSans.ttf")


@functools.lru_cache(maxsize=None)
def default_font():
    from PIL import ImageFont

    try:
        font = ImageFont.load_default(FONT_SIZE)
    except TypeError:  # Pillow before 10.1 can't size its built-in font
        font = None
    if not isinstance(font, ImageFont.FreeTypeFont):
        raise ImportError("FIBO 2's text pictures need Pillow 10.1 or later, built with FreeType")
    return font


@functools.lru_cache(maxsize=None)
def fallback_fonts() -> tuple:
    from PIL import ImageFont

    # the basic layout, as the built-in font's, so the text doesn't set differently where libraqm is installed
    return tuple(ImageFont.truetype(os.path.join(FONTS_DIR, name), FONT_SIZE, layout_engine=ImageFont.Layout.BASIC)
                 for name in FALLBACK_FONTS)


@functools.lru_cache(maxsize=4096)
def has_glyph(font, character: str) -> bool:
    # a font without a glyph for the character draws its notdef box, as it does for U+FFFE, which no font has
    try:
        notdef = font.getmask(chr(0xFFFE))
        mask = font.getmask(character)
    except Exception:  # as in training, a font that can't say is taken to have the glyph
        return True
    return not (mask.size == notdef.size and bytes(mask) == bytes(notdef))


def font_runs(text: str, fonts) -> list:
    """The text in runs of characters drawn with the same font: the first font with a glyph for the character, or the
    built-in font, the first, when none has one."""
    runs = []
    for character in text:
        font = fonts[0]
        if not (ord(character) < 128 or has_glyph(fonts[0], character)):
            font = next((candidate for candidate in fonts[1:] if has_glyph(candidate, character)), fonts[0])
        if runs and runs[-1][1] is font:
            runs[-1][0] += character
        else:
            runs.append([character, font])
    return [(run, font) for run, font in runs]


def draw_one_font(lines: list, font):
    from PIL import Image, ImageDraw

    # each line placed by its bounding box, at least as tall as "Ag"
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1), "white"))
    min_box = draw.textbbox((0, 0), "Ag", font=font)
    min_line_height = max(1, min_box[3] - min_box[1])
    line_sizes, max_width, total_height = [], 1, 0
    for line in lines:
        box = draw.textbbox((0, 0), line or " ", font=font)
        line_height = max(min_line_height, box[3] - box[1])
        line_sizes.append((box, line_height))
        max_width = max(max_width, box[2] - box[0])
        total_height += line_height
    total_height += LINE_SPACING * (len(lines) - 1)

    image = Image.new("RGB", (max_width + PADDING * 2, total_height + PADDING * 2), "white")
    draw = ImageDraw.Draw(image)
    y = PADDING
    for line, (box, line_height) in zip(lines, line_sizes):
        draw.text((PADDING - box[0], y - box[1]), line, font=font, fill=(0, 0, 0))
        y += line_height + LINE_SPACING
    return image


def draw_mixed(lines: list, fonts):
    from PIL import Image, ImageDraw

    # each run drawn with its own font on the line's baseline, each line as tall as the built-in font's ascent and
    # descent
    ascent, descent = fonts[0].getmetrics()
    line_height = max(1, ascent + descent)
    runs_per_line = [font_runs(line or " ", fonts) for line in lines]
    width = max(max(1, round(sum(font.getlength(run) for run, font in runs))) for runs in runs_per_line)
    height = line_height * len(lines) + LINE_SPACING * (len(lines) - 1)

    image = Image.new("RGB", (width + PADDING * 2, height + PADDING * 2), "white")
    draw = ImageDraw.Draw(image)
    y = PADDING
    for runs in runs_per_line:
        x = float(PADDING)
        for run, font in runs:
            draw.text((x, y + ascent), run, font=font, fill=(0, 0, 0), anchor="ls")
            x += font.getlength(run)
        y += line_height + LINE_SPACING
    return image


def draw_text_image(text: str):
    """The picture of one text_render text, as a PIL image: its whitespace collapsed, one line (or lines of 256
    characters), black on white with 2 pixels of padding."""
    font = default_font()
    line = " ".join(text.split())
    lines = [line[i:i + MAX_LINE_CHARS] for i in range(0, len(line), MAX_LINE_CHARS)] or [""]
    if all(ord(character) < 128 or has_glyph(font, character) for character in line):
        return draw_one_font(lines, font)
    return draw_mixed(lines, [font, *fallback_fonts()])
