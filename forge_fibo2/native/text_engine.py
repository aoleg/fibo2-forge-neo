# FIBO 2 support for Forge Neo
# The FIBO 2.0 text encoder: Qwen3-VL 4B over the prompt's chat turn (caption.py), with the pictures of the images to
# edit and of the caption's text_render entries in its vision tower. Fifteen hidden layers are tapped, each scaled
# to unit RMS and rounded to bfloat16, and grouped into six bundles of three (7680 features): one for the
# Perceiver, one for each injection block (ComfyUI's Fibo2TEModel, COMFYUI_COMMIT in native/__init__.py).
#
# Each prompt is encoded on its own and unpadded; the DiT reads every token. A caption with text_render entries and
# no image to edit gives two encodings, without and with the text pictures, packed together (phases.py).

import torch
import torch.nn.functional as F
from backend.args import dynamic_args
from backend.text_processing import emphasis
from backend.text_processing._comfy import INF, SDClipModel, SDTokenizer

from . import caption, phases

PAD = 151643  # <|endoftext|>: never part of a prompt, so Forge's end-of-text masking never cuts one
IMAGE_PAD = 151655  # <|image_pad|>, where a picture goes
# Perceiver, then the injection blocks 6, 14, 22, 30 and 38; Forge's taps are transformers' hidden_states[i]
BUNDLE_LAYERS = ((9, 20, 31), (5, 16, 27), (7, 18, 29), (9, 20, 31), (11, 22, 33), (13, 24, 35))
TAP_LAYERS = sorted({layer for bundle in BUNDLE_LAYERS for layer in bundle})
BUNDLE_INDEX = [TAP_LAYERS.index(layer) for bundle in BUNDLE_LAYERS for layer in bundle]
RMS_EPS = 1e-6


class Fibo2TextEngine:
    def __init__(self, text_encoder, tokenizer):
        self.text_encoder = SDClipModel(text_encoder, layer=TAP_LAYERS, layer_idx=None, special_tokens={"pad": PAD},
                                        layer_norm_hidden_state=False, enable_attention_masks=True,
                                        return_attention_masks=True)
        self.tokenizer = SDTokenizer(tokenizer, pad_with_end=False, has_start_token=False, has_end_token=False,
                                     pad_to_max_length=False, max_length=INF, min_length=1, pad_token=PAD)

    @property
    def emphasis(self) -> "emphasis.Emphasis":
        return emphasis.EmphasisNone()

    def token_ids(self, chat: str) -> list[int]:
        return self.tokenizer.tokenizer.encode(chat, add_special_tokens=False)

    def tokenize(self, texts: str | list[str]) -> int | list[int]:
        # surrounding whitespace is dropped (as Forge's Krea 2 engine does): "", " " and a prompt left with only LoRA
        # tags are all the null caption
        if isinstance(texts, str):
            return len(self.token_ids(caption.build(texts.strip())[0]))
        return [self.tokenize(t) for t in texts]

    def __call__(self, texts: list[str], images: list[torch.Tensor] = ()) -> list[torch.Tensor]:
        if any(emphasis.uses_emphasis(text) for text in texts):
            dynamic_args.last_extra_generation_params["Emphasis"] = "None"

        zs = []
        cache = {}
        for line in texts:
            if line not in cache:
                cache[line] = self.encode(line, images)
            zs.append(cache[line])
        return zs

    def encode(self, line: str, images: list[torch.Tensor] = ()) -> torch.Tensor:
        """(T, 46080 + 1): the conditioning and its phase marker. images are the pictures to edit, (1, H, W, 3) in
        [0, 1] at their final size."""
        prompt = line.strip()
        states = self.states(prompt, images, text_images=True)
        if len(images) == 0 and prompt and caption.build(prompt)[1]:
            # a generation starts from the caption without the text pictures, as upstream does
            return phases.pack(states, self.states(prompt, images, text_images=False))
        return phases.pack(states)

    def states(self, prompt: str, images, text_images: bool) -> torch.Tensor:
        chat, texts = caption.build(prompt, len(images), text_images)
        pictures = [caption.thumbnail(image) for image in images]
        pictures += [caption.draw_text_image(text, caption.font()) for text in texts]
        pictures = iter([caption.to_qwen_grid(picture) for picture in pictures])

        pairs = []
        for token in self.token_ids(chat):
            if token == IMAGE_PAD and (picture := next(pictures, None)) is not None:
                pairs.append(({"type": "image", "data": picture, "original_type": "image"}, 1.0))
            else:
                pairs.append((token, 1.0))

        # (1, 15, T, 2560) float32 -> unit RMS, bfloat16 like the reference's text encoder output -> (T, 6 x 7680)
        states = self.text_encoder.encode_token_weights([pairs])[0].float()
        states = F.rms_norm(states, (states.shape[-1],), eps=RMS_EPS).to(torch.bfloat16)
        return states[0, BUNDLE_INDEX].transpose(0, 1).flatten(1)
