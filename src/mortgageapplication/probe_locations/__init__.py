"""
Stealing a lot of the API from spacy, though things are slighly different
because we don't wrap all tokens or spans, only those we're interested in. It
could be worth wrapping all tokens and spans eventually, and having a separate
flow of the 'special' tokens and spans. 
"""

from typing import TYPE_CHECKING, Any
from dataclasses import dataclass
import re

if TYPE_CHECKING:
    from sae_lens import HookedSAETransformer


POINT = "<:>"
START, END = "<:", ":>"


@dataclass(frozen=True)
class Doc:
    text: str
    clean_text: str
    tokens: Any


@dataclass(frozen=True)
class ProbePoint:
    start: int # index in doc
    end: int # index in doc
    start_char_index: int
    end_char_index: int
    doc: Doc
    
    @property
    def string(self):
        return "".join(
            self.doc.clean_text[self.start_char_index:self.end_char_index]
        )


MARKER = re.compile(f"{POINT}|{START}|{END}")  # <:> must come before <: in the alternation

def find_probe_indices(prompt):
    clean, probes, start, last = "", [], None, 0
    for m in MARKER.finditer(prompt):
        clean += prompt[last:m.start()]
        last = m.end()
        tok = m.group()
        if tok == POINT:
            probes.append(len(clean))
        elif tok == START:
            start = len(clean)
        else:
            probes.append((start, len(clean)))
    return clean + prompt[last:], probes


def _token_ending_at(offsets, char_index):
    """Index of the token whose span ends exactly at `char_index`."""
    for i, (_, end) in enumerate(offsets):
        if end == char_index:
            return i
        if end > char_index:
            break
    raise ValueError(
        f"Probe marker at character {char_index} is not at the end of a token"
    )


def _token_starting_at(clean, offsets, char_index):
    """
    Index of the token that begins at `char_index`. Tokenizers fold a leading
    space into the token, so a token starting just before `char_index` counts
    when only whitespace sits between its start and `char_index`.
    """
    for i, (start, end) in enumerate(offsets):
        if end > char_index:
            if start < char_index and clean[start:char_index].strip():
                raise ValueError(
                    f"Probe marker at character {char_index} is inside a token"
                )
            return i
    raise ValueError(f"Probe marker at character {char_index} is past the last token")


def create_probes(prompt, model: "HookedSAETransformer") -> list[ProbePoint]:
    """
    Build a ProbePoint for each marker in `prompt`, in order of appearance.

    `start`/`end` are token indices into `doc.tokens` and are half-open like
    spacy spans (a `<:>` point is a one-token span). The char indices come from
    the tokens, not the markers, so a token's leading space is included. No BOS
    token is added, so indices line up with the tokenizer's offsets; shift by
    one if the model prepends BOS.
    """
    clean, indices = find_probe_indices(prompt)
    offsets = model.tokenizer(clean, return_offsets_mapping=True)["offset_mapping"]
    strs = model.to_str_tokens(clean, prepend_bos=False)
    doc = Doc(text=prompt, clean_text=clean, tokens=strs)

    probes = []
    for idx in indices:
        if isinstance(idx, tuple):
            first = _token_starting_at(clean, offsets, idx[0])
            last = _token_ending_at(offsets, idx[1])
        else:
            first = last = _token_ending_at(offsets, idx)
        probes.append(
            ProbePoint(
                start=first,
                end=last + 1,
                start_char_index=offsets[first][0],
                end_char_index=offsets[last][1],
                doc=doc,
            )
        )
    return probes

