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

    from forge_fibo2.native import masking

    class Fibo2MaskedEdit(scripts.Script):
        """No UI: hands img2img's inpaint mask, as Forge prepared it for the picture, to the FIBO 2 engine, which
        greys that region out of the picture to edit (masking.py)."""

        def title(self):
            return "FIBO 2 masked edit"

        def show(self, is_img2img):
            return scripts.AlwaysVisible

        def process(self, p, *args):
            # every generation starts without a mask; img2img's inpaint sets one below, during p.init
            masking.set_mask(None)

        def before_process_init_images(self, p, pp, *args, **kwargs):
            if getattr(p.sd_model, "is_fibo2", False):
                masking.set_mask(pp.get("image_mask"))
