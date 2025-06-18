import os
from tqdm import tqdm
import json
import gzip

# import pandas as pd

import re
from pathlib import Path
from markdown_it import MarkdownIt                 # parses & re-renders Markdown :contentReference[oaicite:1]{index=1}
from symspellpy import SymSpell, Verbosity         # spell/segment fixes     :contentReference[oaicite:2]{index=2}
from deepmultilingualpunctuation import PunctuationModel  # punctuation restore :contentReference[oaicite:3]{index=3}
from mdformat.renderer import MDRenderer   # NEW ✔
import mdformat
import pickle

def preclean(md: str) -> str:
    # de-hyphenate split words
    md = re.sub(r'(\w)-\n(\w)', r'\1\2', md)

    # build a translation table that deletes control chars **except** \t, \n, \r
    ctrl_to_delete = {
        c: None
        for c in range(32)
        if c not in (9, 10, 13)          # 9 = \t, 10 = \n, 13 = \r
    }
    md = md.translate(ctrl_to_delete)
    return md

# 2. SymSpell setup ------------------------------------------------------------------

SYM = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
SYM.load_dictionary("/home/dhonchar/llm_project/frequency_dictionary_en_82_765.txt", term_index=0, count_index=1)

SENT_DELIM = "\uFFF9"

# compile once, at module scope
SHORT_DUP_RE = re.compile(
    rf"\b([A-Za-z]{{1,3}})(?:\s*{SENT_DELIM}\s*|\s+)+\1\b",  # ← sentinel counts as space
    flags=re.I,
)

def fix_words(text: str) -> str:
    # join wrongly split words
    # text = drop_short_duplicates(text)
    text = SYM.lookup_compound(text, max_edit_distance = 2)[0].term
    # split run-on chunks
    text = SYM.word_segmentation(text).corrected_string
    return text

# # ------------------------------------------------------------------
# # 3. punctuation model -------------------------------------------------------------
# # PUNCT_MODEL = PunctuationModel()   # ~50 MB, fast on CPU

PUNCT_MODEL = PunctuationModel()


def restore_punct(text: str) -> str:
    return PUNCT_MODEL.restore_punctuation(text)


SKIP_TYPES = {"code_inline", "code_block", "fence"}  # never touch code


def safe_normalize_link(url: str) -> str:
    """Return md-it’s normalised URL, or the original string if it blows up."""
    try:
        return normalize_url.normalizeLink(url)
    except Exception:
        return url            # keep it verbatim – better than crashing


SENT_DELIM = "\uFFF9"
_DUP_RE = re.compile(
    rf"""
    \b([A-Za-z]{{1,3}})            # short word (capture = \1)
    (?:\s*|{SENT_DELIM})+          # any mix of spaces or sentinel(s)
    \1                             # the SAME short word again …
    (?=\W|{SENT_DELIM})?           # … if it is a stand-alone token
    """,
    re.IGNORECASE | re.VERBOSE,
)


def process_inline_block(inline):
    """
    Fixes duplicate short words, runs SymSpell and punctuation once for the
    entire inline container, then writes the text back into the original
    child-tokens (preserving markup positions).
    """
    # gather pure-text children
    txt_nodes = [t for t in inline.children if t.type == "text"]
    if not txt_nodes:
        return

    joined = SENT_DELIM.join(t.content for t in txt_nodes)

    # ① kill short duplicates ("Th Th e"→"Th e")
    joined = _DUP_RE.sub(r"\1 ", joined)

    # ② SymSpell (case-sensitive)  -----------------------------
    try:
        joined = SYM.lookup_compound(joined, max_edit_distance=2,
                                 transfer_casing=True)[0].term
        joined = SYM.word_segmentation(joined).corrected_string
    except:
        pass
    # ③ punctuation model (skip if you only need casing)
    try:
        joined = PUNCT_MODEL.restore_punctuation(joined)
    except IndexError:
        pass
    # ④ split and write back
    parts = joined.split(SENT_DELIM)
    for node, seg in zip(txt_nodes, parts):
        node.content = seg



def clean_markdown(raw_md: str) -> str:
    raw_md = preclean(raw_md)
    md = MarkdownIt()
    tokens = md.parse(raw_md)

    # process **each inline container once**
    for tok in tokens:
        if tok.type == "inline" and tok.type not in SKIP_TYPES:
            process_inline_block(tok)

    env = {"md": md, "lines": raw_md.splitlines(True)}
    pretty = mdformat.text(MDRenderer().render(tokens, {}, env))
    return pretty

import sys

assert len(sys.argv)==3

proc_num = int(sys.argv[1])
total_proc = int(sys.argv[2])

inputs = []
for file in os.listdir("/home/dhonchar/llm_project/data/"):
    with open("/home/dhonchar/llm_project/data/{}".format(file)) as f:
        inputs.append(f.read())
        #print(file)
del inputs[6326] # broken file

step = len(inputs)//total_proc
print(step)
print(len(inputs))
#print(total_proc)
inputs = inputs[int(step*proc_num):int(step*proc_num) + step]
print(len(inputs))

outputs = {i:clean_markdown(x) for i,x in tqdm(enumerate(inputs), total = len(inputs))}

with gzip.open("cleaned_data/data_cleaned_{}.json.gz".format(proc_num), 'wt', encoding='utf-8') as file:
    json.dump(outputs, file, indent=4)
