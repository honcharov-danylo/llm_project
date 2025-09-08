import os
from tqdm import tqdm
import json
import gzip
from langdetect import detect
import spacy
import re
import sys
from typing import List, Tuple
from markdown_it import MarkdownIt                 # parses & re-renders Markdown :contentReference[oaicite:1]{index=1}
from symspellpy import SymSpell         # spell/segment fixes 
from deepmultilingualpunctuation import PunctuationModel  # punctuation restore :contentReference[oaicite:3]{index=3}
from mdformat.renderer import MDRenderer   
import mdformat


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
    """
    Convert text to proper sentence case by capitalizing the first letter of each sentence.
    
    Args:
        text (str): Input text that may have improper capitalization
        
    Returns:
        str: Text with proper sentence case formatting
        
    Examples:
        >>> _sentence_case("hello world. how are you?")
        "Hello world. How are you?"
    """
    text = re.sub(r"^\s*([a-z])", lambda m: m.group(1).upper(), text)
    def _repl(m):
        return m.group(1) + m.group(2).upper()
    text = re.sub(r"([.!?]['”’\")\]]*\s+)([a-z])", _repl, text)
    return text

def _split_on_ner(text: str) -> List[Tuple[str, str]]:
    """
    Split text into parts based on named entity recognition (NER).
    
    Uses spaCy to identify named entities and splits the text into segments,
    marking each segment as either "clean" (regular text) or "keep" (named entities).
    
    Args:
        text (str): Input text to be processed
        
    Returns:
        List[Tuple[str, str]]: List of tuples where each tuple contains:
            - First element: "clean" or "keep" indicating the type of segment
            - Second element: The actual text segment
            
    Examples:
        >>> _split_on_ner("Apple Inc. is headquartered in Cupertino.")
        [("clean", "Apple Inc. is headquartered in "), ("keep", "Cupertino"), ("clean", ".")]
    """
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
    """
    Apply text cleaning pipeline to a single text piece.
    
    Performs spell checking and word segmentation on the input piece using SymSpell.
    Handles compound word lookup and word segmentation with error handling.
    
    Args:
        piece (str): Text piece to be cleaned
        
    Returns:
        str: Cleaned text piece with corrected spelling and segmentation
        
    Examples:
        >>> _clean_piece("helloworld")
        "hello world"
    """
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
    """
    Perform preliminary cleaning on markdown text.
    
    Removes hyphenation at line breaks and deletes unwanted control characters
    while preserving tabs, newlines, and carriage returns.
    
    Args:
        md (str): Raw markdown text to be pre-cleaned
        
    Returns:
        str: Pre-cleaned markdown text with hyphenation removed and control chars cleaned
        
    Examples:
        >>> preclean("hello-\nworld")
        "helloworld"
    """
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
    """
    Restore punctuation to text using a multilingual punctuation model.
    
    Uses a pre-trained model to add appropriate punctuation marks to text
    that may be missing punctuation.
    
    Args:
        text (str): Text without or with incomplete punctuation
        
    Returns:
        str: Text with restored punctuation
        
    Examples:
        >>> restore_punct("hello world how are you")
        "Hello world, how are you?"
    """
    return PUNCT_MODEL.restore_punctuation(text)

SKIP_TYPES = {"code_inline", "code_block", "fence"}  # never touch code


def process_inline_block(inline):
    """
    Process an inline markdown block for text cleaning.
    
    Processes text nodes within an inline block, applying deduplication,
    bracket preservation, NER-based cleaning, and punctuation restoration.
    Modifies the inline block in-place.
    
    Args:
        inline: Markdown inline block object to be processed
        
    Returns:
        None: Modifies the inline block in-place
        
    Note:
        This function preserves bracketed references and named entities while
        cleaning the rest of the text content.
    """
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
    """
    Clean and format markdown text using a comprehensive pipeline.
    
    Applies a series of cleaning operations to markdown text including:
    - Preliminary cleaning (hyphenation removal, control char cleanup)
    - Text deduplication
    - Spell checking and word segmentation
    - Named entity preservation
    - Bracket reference preservation
    - Punctuation restoration
    - Sentence case formatting
    - Final markdown formatting
    
    Args:
        raw_md (str): Raw markdown text to be cleaned and formatted
        
    Returns:
        str: Cleaned and properly formatted markdown text
        
    Examples:
        >>> clean_markdown("# hello world\nThis is a test.")
        "# Hello World\n\nThis is a test."
        
    Note:
        This function preserves code blocks, inline code, and fence blocks
        while cleaning all other text content.
    """
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

assert len(sys.argv)==3

# variables of current condor process and total processes, passed to script from shell
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
