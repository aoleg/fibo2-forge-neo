# FIBO 2 support for Forge Neo
# Two options in Forge Neo's "Stable Diffusion" settings section, where its per-model options live.
# "[FIBO 2] Enable Reference", next to Forge Neo's own "[Klein] / [Qwen 2.1] / [Krea2] Enable Reference": on, the
# img2img image (and the ones ImageStitch adds) are images to edit, and the prompt is the edit instruction; off,
# img2img is plain image-to-image.
# "[FIBO 2] Fused kernels": on, the DiT runs on comfy-kitchen's fused kernels where it can (kernels.py); off, the
# eager path, the same code as before the fused one existed, for A/B pictures and timings and for bug reports.

OPTION = "fibo2_do_reference"
DEFAULT = True
LABEL = "[FIBO 2] Enable Reference"
FUSED_OPTION = "fibo2_fused_kernels"
FUSED_DEFAULT = True
FUSED_LABEL = "[FIBO 2] Fused kernels"
SECTION = ("sd", "Stable Diffusion")
CATEGORY = "sd"


def _option(name, default) -> bool:
    try:
        from modules.shared import opts

        return bool(getattr(opts, name, default))
    except Exception:  # outside Forge Neo (CPU tests), or before the options exist
        return default


def references_enabled() -> bool:
    return _option(OPTION, DEFAULT)


def fused_enabled() -> bool:
    return _option(FUSED_OPTION, FUSED_DEFAULT)


def register() -> None:
    from modules import shared
    from modules.options import OptionInfo

    if OPTION not in shared.opts.data_labels:
        info = OptionInfo(DEFAULT, LABEL, section=SECTION, category_id=CATEGORY)
        info = info.info("enable Edit ; disable img2img").info("pin to <b>Quicksettings</b> is recommended if changed often")
        shared.opts.add_option(OPTION, info)
    if FUSED_OPTION not in shared.opts.data_labels:
        info = OptionInfo(FUSED_DEFAULT, FUSED_LABEL, section=SECTION, category_id=CATEGORY)
        info = info.info("comfy-kitchen fused norms, RoPE and SwiGLU in the DiT (bf16 and INT8 ConvRot files on CUDA) ; "
                         "off runs the plain path, for comparisons")
        shared.opts.add_option(FUSED_OPTION, info)
