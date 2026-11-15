# 🌊 Iris-Z for Forge Neo

<div align="center">

[![Forge Neo](https://img.shields.io/badge/Forge-Neo-blue)](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)
[![Gradio](https://img.shields.io/badge/Gradio-4.40.0-orange)](https://gradio.app/)
[![Version](https://img.shields.io/badge/Version-0.1.0-brightgreen)](https://github.com/eduardoabreu81/iris-z-forge-neo)
[![License](https://img.shields.io/badge/License-AGPL--3.0-green.svg)](LICENSE)

> **Extension for [Stable Diffusion WebUI Forge - Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)**

</div>

Generate with **[Iris-Z](https://huggingface.co/speridlabs/iris-z)** inside Forge Neo, using the checkpoint, preset and Generate button you already know - one model that paints images and reads their depth, built on FLUX.2 [klein] 9B.

> [!Important]
> This extension requires an up-to-date **Forge Neo** (the `neo` branch). On an older version it stays disabled and tells you so in the console - Forge Neo itself keeps working as usual.

> [!Warning]
> **Preview release:** 0.1.0 has not been tested on a GPU yet. Expect rough edges until the first tested version.

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

- **Iris-Z-small in Forge Neo** - text-to-image through the usual txt2img tab
- **Image to Depth** - the depth map of any picture, in img2img
- **iris_z UI preset** - sampler, schedule, steps and CFG ready to generate

---

## 🎯 Features

### 🖼️ Text to Image

- Runs inside txt2img - no separate tab, no extra environment
- Uses the same text encoder and VAE as FLUX.2 [klein] 9B - if you have Klein 9B, you already have them
- Live preview, batches and LoRAs typed in the prompt work as with any other model

### 🌊 Image to Depth

- **Iris-Z** accordion in img2img → **Image to Depth**
- A greyscale depth map of any picture, near is bright - ready for depth ControlNets
- 4 steps, returned at the picture's own size

### 🛡️ Safe by Design

- Changes no Forge Neo file - remove the extension and everything is as it was
- Checks your Forge Neo at startup and stays disabled if something it needs is missing
- Other models, plain FLUX.2 [klein] included, are not affected

---

## 📦 Installation

1. Open Forge Neo WebUI
2. Go to **Extensions** → **Install from URL**
3. Paste: `https://github.com/eduardoabreu81/iris-z-forge-neo`
4. Click **Install** and restart the WebUI
5. Place the files in the usual folders:

| Part | File | Folder |
| :--- | :--- | :--- |
| Iris-Z checkpoint | `iris-z-small_bf16` (see below) | `models/Stable-diffusion` |
| Qwen3 8B text encoder | [`qwen_3_8b`](https://huggingface.co/Comfy-Org/Flux-2-Klein-9B/tree/main/split_files/text_encoders) or `qwen_3_8b_fp8mixed` | `models/text_encoder` |
| FLUX.2 VAE | [`flux2-vae`](https://huggingface.co/Comfy-Org/Flux-2-Klein-9B/tree/main/split_files/vae) | `models/VAE` |

6. Pick the **iris_z** UI preset, the checkpoint, and the text encoder and VAE under **VAE / Text Encoder**

> [!Note]
> Iris-Z is published in four parts. Download the **Iris-Z-small** folder from [speridlabs/iris-z](https://huggingface.co/speridlabs/iris-z/tree/main/Iris-Z-small) and join it into one checkpoint with the script that comes with this extension:
>
> ```
> python extensions/iris-z-forge-neo/tools/join_shards.py <the Iris-Z-small folder> models/Stable-diffusion/iris-z-small_bf16.safetensors
> ```
>
> The checkpoint is about 18 GB. Iris-Z-big (32B) is not supported: Forge Neo has no loader for its text encoder.

---

## ⚙️ Recommended Settings

| Use | Sampler | Schedule type | Steps | CFG |
| :--- | :--- | :--- | :--- | :--- |
| **Images** | Euler | Flux2 | 50 | 4 |
| **Image to Depth** | set by the extension | - | - | - |

Leave the **negative prompt** empty. Iris-Z was trained at 512 and 1024: use 1024x1024 or about one megapixel with sides a multiple of 16, or 512x512 for quick drafts.

---

## 💡 Tips

- Iris-Z generates each image together with a depth layer, so it takes about twice as long as FLUX.2 [klein] 9B at the same size
- For **Image to Depth**, write nothing in the prompt, or only a LoRA tag - the text is ignored
- **Image to Depth** works at 512 on pictures under half a megapixel and at 1024 above, then returns the map at the picture's size
- Before uninstalling, switch the UI preset to another one

---

## 📄 Credits

- **[Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)** by Haoming02
- **[Iris-Z](https://huggingface.co/speridlabs/iris-z)** by SperidLabs - the model and its [reference code](https://github.com/speridlabs/iris-z-sdk)
- **[FLUX.2](https://github.com/black-forest-labs/flux2)** by Black Forest Labs - the model Iris-Z is built on

---

## 📜 License

AGPL-3.0 - see [LICENSE](LICENSE). The Iris-Z weights are under the [FLUX Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.2-dev/blob/main/LICENSE.txt).

---

<div align="center">

Made with ❤️ for the Stable Diffusion community

**[Report Bug](https://github.com/eduardoabreu81/iris-z-forge-neo/issues)** • **[Request Feature](https://github.com/eduardoabreu81/iris-z-forge-neo/issues)** • **[☕ Ko-fi](https://ko-fi.com/eduardoabreu81)**

</div>
