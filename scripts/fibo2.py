# FIBO 2 support for Forge Neo
# Entry point: Forge Neo imports every scripts/*.py at startup, before the UI and before any model loads, which is
# when the model gets registered in the checkpoint list. Everything runs in the module scope; any failure leaves
# Forge Neo unchanged and says so in the console.

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ENABLED = False

try:
    from forge_fibo2.native import compat

    _problems = compat.check()
    if _problems:
        compat.report(_problems)
    elif (_native := compat.native_support()) is not None:
        compat.report_native(_native)
    else:
        from forge_fibo2.native import patches

        patches.apply()
        ENABLED = True
except Exception:
    print("[FIBO 2] failed to enable, Forge Neo is unchanged:")
    traceback.print_exc()

if ENABLED:
    from modules import scripts

    from forge_fibo2.native import caption, masking, settings

    class Fibo2MaskedEdit(scripts.Script):
        """No UI: hands the result's size and img2img's inpaint mask, as Forge prepared it for the picture, to the
        FIBO 2 engine, which crops the picture to edit to that size and greys the masked region out (masking.py);
        says in the console when the size does not suit an edit."""

        def title(self):
            return "FIBO 2 masked edit"

        def show(self, is_img2img):
            return scripts.AlwaysVisible

        def process(self, p, *args):
            # every generation starts without a mask; img2img's inpaint sets one below, during p.init
            masking.set_mask(None)
            if getattr(p.sd_model, "is_fibo2", False):
                p.sd_model.fibo2_output_size = (p.width, p.height)

        def before_process_init_images(self, p, pp, *args, **kwargs):
            if not getattr(p.sd_model, "is_fibo2", False):
                return
            masking.set_mask(pp.get("image_mask"))
            if settings.references_enabled() and p.init_images:
                region = pp.get("crop_region")
                picture = (region[2] - region[0], region[3] - region[1]) if region else p.init_images[0].size
                self.advise(picture, (p.width, p.height))

        @staticmethod
        def advise(picture, size):
            from modules.shared import opts

            better = caption.size_advice(picture, size, int(getattr(opts, "res_step", 64)))
            if better is not None:
                print(f"[FIBO 2] img2img is set to {size[0]}x{size[1]} for a {picture[0]}x{picture[1]} picture; "
                      f"edits work best at about 1 megapixel in the picture's proportions, such as "
                      f"{better[0]}x{better[1]} with Crop and resize - other sizes can zoom in or lose detail")
