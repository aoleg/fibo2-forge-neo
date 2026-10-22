# 🖼️ FIBO 2 for Forge Neo

<div align="center">

[![Forge Neo](https://img.shields.io/badge/Forge-Neo-blue)](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)
[![Gradio](https://img.shields.io/badge/Gradio-4.40.0-orange)](https://gradio.app/)
[![Version](https://img.shields.io/badge/Version-0.1.0-brightgreen)](https://github.com/eduardoabreu81/fibo2-forge-neo)
[![License](https://img.shields.io/badge/License-AGPL--3.0-green.svg)](LICENSE)

> **Extension for [Stable Diffusion WebUI Forge - Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)**

</div>

Generate and edit with **[FIBO 2.0](https://huggingface.co/briaai/Fibo-2.0-dev)** by Bria inside Forge Neo, using the checkpoint, preset and Generate button you already know - text-to-image from a structured JSON caption or plain text, and image editing with Edit Turbo in img2img.

> [!Important]
> This extension requires an up-to-date **Forge Neo** (the `neo` branch). On an older version it stays disabled and tells you so in the console - Forge Neo itself keeps working as usual. If your Forge Neo learns FIBO 2.0 by itself, the extension steps aside and says so.

---

## 📋 Table of Contents

- [What's New](#-whats-new)
- [Features](#-features)
- [Installation](#-installation)
- [Recommended Settings](#%EF%B8%8F-recommended-settings)
- [Tips](#-tips)
- [Credits](#-credits)

---

## 🆕 What's New

### v0.1.0 - First Release

- **FIBO 2.0 in Forge Neo** - Bria's Dev and Edit Turbo models, with their Qwen3-VL 4B text encoder
- **Structured captions** - Bria's JSON captions (VGL), cleaned as in training, or plain text
- **Text that renders** - each `text_render` entry is shown to the text encoder as a picture of its words, as upstream does
- **Editing in img2img** - Edit Turbo changes the img2img picture following your instruction; ImageStitch adds more pictures
- **fibo2** and **fibo2_turbo** UI presets - upstream's settings, ready to generate

---

## 🎯 Features

### 🖼️ Native Generation

- Runs inside txt2img and img2img - no separate tab, no extra environment
- Works with the Forge Neo model folders and the **VAE / Text Encoder** selector you already use
- Uses the same autoencoder as FLUX.2 - if you have `flux2-vae`, you already have it
- On an RTX 2070 (8 GB) with the INT8 files: about 3 minutes per 1024x1024 picture with Dev at 30 steps, about 40 seconds with Edit Turbo at 4 steps

### 📝 Structured Captions

- FIBO 2.0 was trained on long JSON captions: `short_description`, `objects` (each with its description, location, size, pose...), `background_setting`, `lighting`, `aesthetics`, `photographic_characteristics`, `style_medium`, `text_render` and more - see [Bria's VGL docs](https://docs.bria.ai/vgl)
- Paste a JSON caption into the prompt; empty fields and scores are dropped, as in training
- Plain sentences work too
- An empty negative prompt is the "null caption" the model was trained with - leave it empty

### 🔤 Text Rendering

- Put the words in `text_render`: `"text_render": [{"text": "OPEN 24H", "location": "center", ...}]`
- The text encoder sees a picture of each text from the middle of the generation on, as Bria's pipeline does

### ✏️ Editing with Edit Turbo

- Put the picture in img2img, write what to change (`Add a red knitted scarf around the neck of the fox`) and set **Denoising strength** to 1 - everything you did not ask to change stays as it was
- Adding, removing and swapping things works best; changing the light of the whole scene (day to night) barely changes it
- More pictures through ImageStitch: the img2img picture is `<image_1>`, the ones you add are `<image_2>`, `<image_3>`... - for example `Put the sleeping fox from <image_1> on the sidewalk in front of the bakery in <image_2>`
- Each extra picture is used at about 1 megapixel, as upstream does, so it costs time: on an RTX 2070 (8 GB) an edit with two pictures took about 10 minutes, against 40 seconds with one
- A JSON prompt can carry the instruction too: `{"edit_instruction": "...", "short_description": "...", ...}`
- Turn editing off under **Settings → Stable Diffusion → [FIBO 2] Enable Reference** for plain image-to-image
- Edit Turbo also makes pictures from text alone, in 4 steps without CFG

### ⚡ Turbo LoRA

- Comfy-Org's turbo LoRA turns Dev into a 4-step model: download `fibo_2.0_turbo_lora_avg_rank_270_bf16` from the `loras` folder of [Comfy-Org/Fibo-2.0](https://huggingface.co/Comfy-Org/Fibo-2.0), put it in `models/Lora` and add it to the prompt, for example `<lora:fibo_2.0_turbo_lora_avg_rank_270_bf16:1>`
- Set **Steps** to 4 and **CFG** to 1 - the **fibo2_turbo** preset values
- With the same seed, Dev with the LoRA gives nearly the same picture as Edit Turbo

### 🛡️ Safe by Design

- Changes no Forge Neo file - remove the extension and everything is as it was
- Checks your Forge Neo at startup and stays disabled if something it needs is missing
- Tells you which file is missing under **VAE / Text Encoder** instead of failing with a generic error
- Other models are not affected

---

## 📦 Installation

1. Open Forge Neo WebUI
2. Go to **Extensions** → **Install from URL**
3. Paste: `https://github.com/eduardoabreu81/fibo2-forge-neo`
4. Click **Install** and restart the WebUI
5. Download the files from [Comfy-Org/Fibo-2.0](https://huggingface.co/Comfy-Org/Fibo-2.0) and place them in the usual folders:

| Part | File | Folder |
| :--- | :--- | :--- |
| FIBO 2.0 checkpoint | `fibo2_dev_int8_convrot` - or `fibo2_edit_turbo_int8_convrot` for Edit Turbo | `models/Stable-diffusion` |
| Qwen3-VL 4B text encoder | `qwen3vl_4b_int8_convrot` | `models/text_encoder` |
| FLUX.2 autoencoder | `flux2-vae` | `models/VAE` |

6. Restart Forge Neo, so the new text encoder shows up in the list
7. Pick the **fibo2** (Dev) or **fibo2_turbo** (Edit Turbo) UI preset, the checkpoint, and the text encoder and autoencoder under **VAE / Text Encoder**

> [!Note]
> Tested with the INT8 files of Dev and Edit Turbo on an RTX 2070 (8 GB), Forge Neo 2.31: text-to-image, text rendering, edits of one picture and of two (ImageStitch), and the turbo LoRA with Dev. The BF16 files have not been tested yet.

---

## ⚙️ Recommended Settings

| Model | Sampler | Schedule type | Steps | CFG |
| :--- | :--- | :--- | :--- | :--- |
| Dev | Euler | Simple | 30 | 5 |
| Edit Turbo | Euler | Simple | 4 | 1 |

The presets set these values; change them on the **FIBO2** and **FIBO2_TURBO** pages of **Settings** and the presets keep yours. FIBO 2.0 was trained at about 1 megapixel: 1024x1024, 1424x800, 800x1424... Use sides that are a multiple of 16.

---

## 💡 Tips

- Long, detailed captions are what the model knows best; a language model can turn a sentence into a VGL caption
- For edits, use **Crop and resize** in img2img so the picture keeps its proportions
- New text encoder files appear in the list after a restart of Forge Neo
- Before uninstalling, switch the UI preset to another one

---

## 📄 Credits

- **[Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)** by Haoming02
- **[FIBO 2.0](https://huggingface.co/briaai/Fibo-2.0-dev)** by Bria - the model and its reference pipeline
- **[ComfyUI](https://github.com/Comfy-Org/ComfyUI)** by comfyanonymous and Comfy-Org - the FIBO 2.0 implementation this port follows ([PR #17240](https://github.com/Comfy-Org/ComfyUI/pull/17240)), and the INT8 files
- **[Qwen3-VL](https://huggingface.co/Qwen)** by the Qwen team - the text encoder

The FIBO 2.0 weights are © BRIA AI under the [Creative Commons Attribution-NonCommercial 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license: non-commercial use only. This extension does not include them.

---

## 📜 License

AGPL-3.0 - see [LICENSE](LICENSE)

---

<div align="center">

Made with ❤️ for the Stable Diffusion community

**[Report Bug](https://github.com/eduardoabreu81/fibo2-forge-neo/issues)** • **[Request Feature](https://github.com/eduardoabreu81/fibo2-forge-neo/issues)** • **[☕ Ko-fi](https://ko-fi.com/eduardoabreu81)**

</div>
