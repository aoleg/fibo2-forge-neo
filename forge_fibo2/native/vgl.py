# FIBO 2 support for Forge Neo
# A structured caption (Bria's VGL) whose braces sd-dynamic-prompts stripped, put back together.
#
# sd-dynamic-prompts reads {x} as a one-option variant and strips every brace, which leaves
# "short_description":"...","objects":["description":"...","location":"...","description":...],"lighting":
# "conditions":"...","direction":"...","style_medium":"..." for a caption. The braces go back where the caption's
# shape says they go: around the whole prompt; around the value of a key that holds an object (lighting, aesthetics,
# structured_caption...), which ends where a key of the enclosing object starts; around each element of a list of
# key-value pairs (objects, text_render), a new element starting when a key repeats; and {} where a value is missing.
# Only braces are added, the rest of the text is kept as it is, and the result is used only when it parses as a
# caption. Same mechanism as the MOSS-Image extension's layout.py.
#
# No Forge imports: the CPU tests run it as is.

import json
import re

CAPTION_KEYS = frozenset(("short_description", "objects", "background_setting", "lighting", "aesthetics",
                          "photographic_characteristics", "style_medium", "context", "artistic_style", "text_render",
                          "aesthetic_score", "preference_score"))
_IMAGE_KEY = re.compile(r"image(_\d+)?")
_STRIPPED_START = re.compile(r'\s*"(' + "|".join(sorted(CAPTION_KEYS)) + r'|edit_instruction|structured_caption|'
                             r'image(_\d+)?)"\s*:')
_SCALAR_END = re.compile(r'[\s:,\[\]"{}]')


def is_edit_key(key: str) -> bool:
    return key in ("edit_instruction", "structured_caption") or bool(_IMAGE_KEY.fullmatch(key))


def is_top_key(key: str) -> bool:
    return key in CAPTION_KEYS or is_edit_key(key)


def _never(key: str) -> bool:
    return False


def _tokens(text: str) -> list[tuple[str, int, int]]:
    """(kind, start, end) for each JSON token: "s" a string, ":" "," "[" "]" punctuation, "v" any other scalar."""
    tokens, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
        elif c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            if j >= n:
                raise ValueError("unterminated string")
            tokens.append(("s", i, j + 1))
            i = j + 1
        elif c in ":,[]":
            tokens.append((c, i, i + 1))
            i += 1
        elif c in "{}":
            raise ValueError("braces left: not a stripped caption")
        else:
            m = _SCALAR_END.search(text, i)
            j = m.start() if m else n
            tokens.append(("v", i, j))
            i = j
    return tokens


class _Rebuilder:
    def __init__(self, text: str):
        self.text = text
        self.tokens = _tokens(text)
        self.inserts: list[tuple[int, str]] = []

    def kind(self, i: int) -> str | None:
        return self.tokens[i][0] if i < len(self.tokens) else None

    def is_key(self, i: int) -> bool:
        return self.kind(i) == "s" and self.kind(i + 1) == ":"

    def key(self, i: int) -> str:
        _, a, b = self.tokens[i]
        return json.loads(self.text[a:b])

    def value(self, i: int, stop) -> int:
        """Parses the value at token i, adding the braces it lost; returns the index of the token after it."""
        if self.is_key(i):
            # a stripped object as a value: it ends where a key of the enclosing object starts
            return self.members(i, stop=stop, split_on_repeat=False)
        kind = self.kind(i)
        if kind in ("s", "v"):
            return i + 1
        if kind == "[":
            return self.array(i)
        if kind in (",", None):
            # an empty object, {}, stripped to nothing
            self.inserts.append((self.tokens[i - 1][2], "{}"))
            return i
        raise ValueError(f"unexpected token at {i}")

    def array(self, i: int) -> int:
        i += 1
        while self.kind(i) != "]":
            # a list of key-value pairs is a list of stripped objects, one per repeated key
            i = self.members(i, stop=_never, split_on_repeat=True) if self.is_key(i) else self.value(i, _never)
            if self.kind(i) == ",":
                i += 1
            elif self.kind(i) != "]":
                raise ValueError("unterminated list")
        return i + 1

    def members(self, i: int, stop, split_on_repeat: bool) -> int:
        """The key-value pairs of one stripped object from token i, wrapped in braces; returns the index after it."""
        self.inserts.append((self.tokens[i][1], "{"))
        seen = set()
        while True:
            key = self.key(i)
            seen.add(key)
            # an edit's structured_caption holds caption keys and ends at the edit's own keys; any other object
            # value ends at a key of the caption or of the edit
            i = self.value(i + 2, is_edit_key if key == "structured_caption" else is_top_key)
            if self.kind(i) != "," or not self.is_key(i + 1):
                break
            nxt = self.key(i + 1)
            if stop(nxt) or (split_on_repeat and nxt in seen):
                break
            i += 1
        self.inserts.append((self.tokens[i - 1][2], "}"))
        return i

    def rebuild(self) -> str:
        end = self.members(0, stop=_never, split_on_repeat=False)
        if end != len(self.tokens):
            raise ValueError("text after the caption")
        out, last = [], 0
        # stable by position: at one spot an inner brace was recorded before the outer one
        for pos, brace in sorted(self.inserts, key=lambda item: item[0]):
            out.append(self.text[last:pos])
            out.append(brace)
            last = pos
        out.append(self.text[last:])
        return "".join(out)


def is_caption(value) -> bool:
    return isinstance(value, dict) and any(is_top_key(k) for k in value)


def restore_caption_json(text: str) -> str:
    """The prompt with a brace-stripped caption rebuilt; unchanged when it is not one or the result does not parse."""
    if not _STRIPPED_START.match(text):
        return text
    try:
        rebuilt = _Rebuilder(text).rebuild()
        if is_caption(json.loads(rebuilt)):
            return rebuilt
    except (ValueError, IndexError):
        pass
    return text
