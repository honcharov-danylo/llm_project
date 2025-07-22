import os
from tqdm import tqdm
import json
import gzip
from langdetect import detect
import spacy, re
from typing import List, Tuple
# import pandas as pd

from pathlib import Path
from markdown_it import MarkdownIt                 # parses & re-renders Markdown :contentReference[oaicite:1]{index=1}
from symspellpy import SymSpell, Verbosity         # spell/segment fixes     :contentReference[oaicite:2]{index=2}
from deepmultilingualpunctuation import PunctuationModel  # punctuation restore :contentReference[oaicite:3]{index=3}
from mdformat.renderer import MDRenderer   # NEW ✔
import mdformat
import string
import pickle

# load spaCy once
NLP = spacy.load("en_core_web_sm")      # or your preferred model

# NLP = spacy.load("en_core_web_trf")

BRACKET_RE = re.compile(
    r"""(
        \[[^\[\]]*]               |   # [ … ]
        \([^)]+\)                 |   # ( … )
        \{[^}]+\}                 |   # { … }
        "[^"\n]*"                 |   # " … "
        '[^'\n]*'                 |   # ' … '
        “[^”\n]*”                 |   # “ … ”
        ‘[^’\n]*’                     # ‘ … ’
    )""",
    re.VERBOSE,
)

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


IGNORE_RE = re.compile(rf"(?:[^\w\s{SENT_DELIM}]+|\d+)")   # for word_segmentation


def _sentence_case(text: str) -> str:
    text = re.sub(r"^\s*([a-z])", lambda m: m.group(1).upper(), text)
    def _repl(m):
        return m.group(1) + m.group(2).upper()
    text = re.sub(r"([.!?]['”’\")\]]*\s+)([a-z])", _repl, text)
    return text

def _split_on_ner(text: str) -> List[Tuple[str, str]]:
    doc = NLP(text)
    parts = []
    last = 0
    for ent in doc.ents:
        if ent.start_char > last:
            parts.append(("clean", text[last:ent.start_char]))
        parts.append(("keep", ent.text))
        last = ent.end_char
    if last < len(text):
        parts.append(("clean", text[last:]))
    return parts


def _clean_piece(piece: str) -> str:
    """Run the usual text-cleaning pipeline on *one* piece."""
    try:
        piece = SYM.lookup_compound(
            piece, max_edit_distance=1,
            transfer_casing=True, ignore_non_words=True
        )[0].term
    except:
        pass
    if not piece.strip():
        return piece
    try:
        piece = SYM.word_segmentation(piece, ignore_token=IGNORE_RE).corrected_string
    except:
        pass
    return piece



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

SYM = SymSpell(max_dictionary_edit_distance=1, prefix_length=7)
SYM.load_dictionary("frequency_dictionary_en_82_765.txt", term_index=0, count_index=1)

# compile once, at module scope
SHORT_DUP_RE = re.compile(
    rf"\b([A-Za-z]{{1,3}})(?:\s*{SENT_DELIM}\s*|\s+)+\1\b",  # ← sentinel counts as space
    flags=re.I,
)

PUNCT_MODEL = PunctuationModel("oliverguhr/fullstop-punctuation-multilang-large")

def restore_punct(text: str) -> str:
    return PUNCT_MODEL.restore_punctuation(text)

SKIP_TYPES = {"code_inline", "code_block", "fence"}  # never touch code


def process_inline_block(inline):
    txt_nodes = [t for t in inline.children if t.type == "text"]
    if not txt_nodes:
        return

    for node in txt_nodes:
        deduped = _DUP_RE.sub(r"\1 ", node.content)
        outer_parts = re.split(BRACKET_RE, deduped)

        final_parts = []          # will accumulate cleaned + kept chunks

        for idx, outer in enumerate(outer_parts):
            if idx % 2 == 1:      # this is a bracketed reference
                # print("KEEP BRACKETS :", outer)
                final_parts.append(outer)
                continue
            for kind, chunk in _split_on_ner(outer):
                if kind == "keep":
                    # print("KEEP  NER     :", chunk)
                    final_parts.append(chunk)
                else:
                    cleaned = _clean_piece(chunk)
                    # cleaned =  PUNCT_MODEL.restore_punctuation(cleaned)
                    final_parts.append(cleaned)
        joined = " ".join(final_parts)
        try:
            joined = PUNCT_MODEL.restore_punctuation(joined)
        except:
            pass
        joined = _sentence_case(joined)
        node.content = joined



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
for file in os.listdir("data/"):
    with open("data/{}".format(file)) as f:
        inputs.append(f.read())
del inputs[6326] # broken file

non_eng_docs = []
non_eng_indices = []
for i, doc in enumerate(inputs):
    try:
        if detect(doc)!='en':
            non_eng_docs.append(doc)
            non_eng_indices.append(i)
    except Exception:
        non_eng_docs.append(doc)
        non_eng_indices.append(i)


inputs = [doc for i, doc in enumerate(inputs) if i not in set(non_eng_indices)]

step = len(inputs)//total_proc
print(step)
print(len(inputs))
#print(total_proc)
inputs = inputs[int(step*proc_num):int(step*proc_num) + step]
print(len(inputs))

outputs = {i:clean_markdown(x) for i,x in tqdm(enumerate(inputs), total = len(inputs))}

with gzip.open("cleaned_data/data_cleaned_{}.json.gz".format(proc_num), 'wt', encoding='utf-8') as file:
    json.dump(outputs, file, indent=4)
