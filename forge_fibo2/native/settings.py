# FIBO 2 support for Forge Neo
# "[FIBO 2] Enable Reference", next to Forge Neo's own "[Klein] / [Qwen 2.1] / [Krea2] Enable Reference": on, the
# img2img image (and the ones ImageStitch adds) are images to edit, and the prompt is the edit instruction; off,
# img2img is plain image-to-image.

OPTION = "fibo2_do_reference"
DEFAULT = True
LABEL = "[FIBO 2] Enable Reference"
# Forge Neo's "Stable Diffusion" settings section and category, where its per-model options live
SECTION = ("sd", "Stable Diffusion")
CATEGORY = "sd"


def references_enabled() -> bool:
    try:
        from modules.shared import opts

        return bool(getattr(opts, OPTION, DEFAULT))
    except Exception:  # outside Forge Neo (CPU tests), or before the options exist
        return DEFAULT


def register() -> None:
    from modules import shared
    from modules.options import OptionInfo

    if OPTION not in shared.opts.data_labels:
        info = OptionInfo(DEFAULT, LABEL, section=SECTION, category_id=CATEGORY)
        info = info.info("enable Edit ; disable img2img").info("pin to <b>Quicksettings</b> is recommended if changed often")
        shared.opts.add_option(OPTION, info)
