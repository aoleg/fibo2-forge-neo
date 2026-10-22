# FIBO 2 support for Forge Neo
# The "fibo2" (Dev) and "fibo2_turbo" (Edit Turbo) UI presets, added at runtime: PresetArch gains a member each and
# their options are registered through Forge Neo's own presets.register, so they match every other preset. Same
# mechanism as the FLUX 3 extension; each preset keeps its own checkpoint, as Forge's presets do.

from enum import Enum

from modules_forge import presets

# upstream's settings (ComfyUI's PR and workflows): Euler with the Simple schedule; Dev 30 steps at CFG 5, the
# guidance-free Edit Turbo 4 steps
PRESETS = {
    "fibo2": ("Euler", "Simple", 30, 5.0),
    "fibo2_turbo": ("Euler", "Simple", 4, 1.0),
}


def _add_enum_member(enum_cls: type[Enum], name: str) -> Enum:
    if name in enum_cls.__members__:
        return enum_cls[name]

    value = max(m.value for m in enum_cls) + 1
    member = object.__new__(enum_cls)
    member._name_ = name
    member._value_ = value
    member.__objclass__ = enum_cls
    member._sort_order_ = len(enum_cls._member_names_)

    # EnumType.__setattr__ refuses new members, so go through type
    type.__setattr__(enum_cls, name, member)
    enum_cls._member_map_[name] = member
    enum_cls._member_names_.append(name)
    enum_cls._value2member_map_[value] = member
    if hasattr(enum_cls, "_hashable_values_"):
        enum_cls._hashable_values_.append(value)
    return member


def _is_preset_option(key: str) -> bool:
    return any(key.startswith(f"{name}_") or key.endswith(f"_{name}") for name in PRESETS)


def register() -> None:
    for name, (sampler, scheduler, steps, cfg) in PRESETS.items():
        arch = _add_enum_member(presets.PresetArch, name)
        presets.SAMPLERS[arch] = sampler
        presets.SCHEDULERS[arch] = scheduler
        presets.STEPS[arch] = steps
        presets.CFG[arch] = cfg

    from modules import shared

    templates: dict = {}
    presets.register(templates)
    for key, info in templates.items():
        if _is_preset_option(key) and key not in shared.opts.data_labels:
            shared.opts.add_option(key, info)
