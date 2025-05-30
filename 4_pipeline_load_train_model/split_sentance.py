import re

# Action keywords (only split if phrase starts with these)
ACTION_KEYWORDS = ['open', 'close', 'play']

# WH-question starters
WH_WORDS = [
    "what", "who", "when", "where", "why", "how", "which", "whom", "whose","can", "could", "will", "would", "shall", "should", "may", "might", "is", "are", "was", "were", "do", "does", "did", "has", "have", "had"
]

IMPERATIVE_VERBS = ['tell', 'show', 'remind', 'give', 'say', 'ask', 'play', 'open', 'close']


PREPOSITIONS = {
    'in', 'on', 'at', 'for', 'with', 'via', 'through', 'using',
    'to', 'from', 'about', 'into', 'onto', 'over', 'under', 'by'
}

# Smarter WH/Auxiliary match to catch typical variations and typos
WH_LIKE_PATTERN = re.compile(
    r'\b(wh[a-z]{1,10}|how)\b',
    re.IGNORECASE
)

# Conjunction words for splitting
CONJUNCTIONS = [
    "and", "or", "also", "another", "moreover", "furthermore", "in addition", "additionally",
    "besides", "as well as", "plus", "but", "however", "yet", "although", "though", "even though",
    "whereas", "while", "on the other hand", "in contrast", "conversely", "nevertheless",
    "nonetheless", "still", "despite", "in spite of", "instead", "because", "since", "as",
    "due to", "owing to", "for this reason", "so", "therefore", "thus", "consequently",
    "as a result", "hence", "accordingly", "after", "before", "as soon as", "once", "until",
    "till", "then", "next", "finally", "firstly", "secondly", "subsequently", "afterwards",
    "meanwhile", "by the time", "whenever", "unless", "provided that", "as long as",
    "in case", "otherwise", "for example", "for instance", "namely", "such as", "above all",
    "overall", "in conclusion", "to conclude", "in summary", "to summarize", "briefly","btw","by the way", 
]

# Compile regex patterns
PUNCTUATION_PATTERN = re.compile(r'[?.,;]')
CONJUNCTION_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(c) for c in CONJUNCTIONS) + r')\b', flags=re.IGNORECASE)
# match action anywhere
ACTION_PREFIX_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(a) for a in ACTION_KEYWORDS) + r')\b',
    flags=re.IGNORECASE
)

def split_on_punctuation(text):
    return [p.strip() for p in PUNCTUATION_PATTERN.split(text) if p.strip()]

def split_on_multiple_actions(text):
    """
    Split when multiple action verbs (open/close/play...) appear in a single chunk.
    """
    matches = list(ACTION_PREFIX_PATTERN.finditer(text))
    if not matches or len(matches) <= 1:
        return [text.strip()]

    parts = []
    for i in range(len(matches)):
        start = matches[i].start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        if chunk:
            parts.append(chunk)

    return parts



def smart_split_imperative(text):
    """
    Only splits if multiple imperative verbs exist with clear conjunctions.
    Otherwise, returns the original text as one chunk.
    """
    verb_pattern = r'\b(' + '|'.join(re.escape(v) for v in IMPERATIVE_VERBS) + r')\b'
    matches = list(re.finditer(verb_pattern, text, re.IGNORECASE))

    if len(matches) <= 1:
        return [text.strip()]  # Only one verb – don't split

    # Now check for conjunctions between imperative verbs
    splits = []
    last_index = 0

    for i in range(1, len(matches)):
        prev_end = matches[i - 1].end()
        current_start = matches[i].start()
        between = text[prev_end:current_start]

        if CONJUNCTION_PATTERN.search(between):  # If conjunction between verbs
            chunk = text[last_index:current_start].strip()
            if chunk:
                splits.append(chunk)
            last_index = current_start

    # Append the last chunk
    final_chunk = text[last_index:].strip()
    if final_chunk:
        splits.append(final_chunk)

    return splits if splits else [text.strip()]


def split_on_wh(text):
    matches = list(WH_LIKE_PATTERN.finditer(text))
    if not matches:
        return [text.strip()]

    chunks = []
    last_index = 0
    for i, match in enumerate(matches):
        start = match.start()
        if start > last_index:
            prefix = text[last_index:start].strip()
            if prefix:
                chunks.append(prefix)

        next_start = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        full_chunk = text[match.start():next_start].strip()
        chunks.append(full_chunk)
        last_index = next_start

    return chunks


def split_on_conjunctions(text):
    return [p.strip() for p in CONJUNCTION_PATTERN.split(text) if p.strip() and p.lower() not in [c.lower() for c in CONJUNCTIONS]]


# def extract_entities(chunk, parent_text):
#     # Determine action: from chunk or fallback to parent start
#     m = ACTION_PREFIX_PATTERN.match(chunk)
#     action = m.group(1).lower() if m else None
#     if not action:
#         # fallback: if parent starts with action, use that
#         mp = ACTION_PREFIX_PATTERN.match(parent_text)
#         action = mp.group(1).lower() if mp else None

#     if action:
#         # if chunk starts with action, remove it
#         content = ACTION_PREFIX_PATTERN.sub('', chunk).strip()
#         words = content.split()
#         # multi-entity: split each word into its own action entity
#         if len(words) > 1:
#             return [f"{action} {w.lower()}" for w in words]
#         # otherwise keep full phrase
#         return [f"{action} {content.lower()}".strip()]
#     # no action: return chunk as-is
#     return [chunk.lower().strip()]


def extract_entities(chunk, parent_text):
    """
    - Finds the action in `chunk`.
    - Strips it out to get `content`.
    - Splits content into tokens.
    - Treats every token before the first PREPOSITION as an entity.
    - Joins all tokens after that as the trailing context for the last entity.
    """
    # 1️⃣ find action
    m = ACTION_PREFIX_PATTERN.search(chunk)
    action = m.group(1).lower() if m else None

    # fallback to parent if missing
    if not action:
        mp = ACTION_PREFIX_PATTERN.search(parent_text)
        action = mp.group(1).lower() if mp else None

    if not action:
        return [chunk.lower().strip()]

    # 2️⃣ remove only first occurrence of the action word
    content = re.sub(
        r'\b' + re.escape(action) + r'\b', '',
        chunk, count=1, flags=re.IGNORECASE
    ).strip()

    if not content:
        return [action]

    # 3️⃣ tokenize
    tokens = content.split()
    entities = []
    context_start = len(tokens)
    
    # 4️⃣ detect where trailing context begins
    for idx, tok in enumerate(tokens):
        if tok.lower() in PREPOSITIONS:
            context_start = idx
            break

    # 5️⃣ first, every token before context_start is its own entity
    for tok in tokens[:context_start]:
        entities.append(f"{action} {tok.lower()}")

    # 6️⃣ if there's context, append it to the last entity
    if context_start < len(tokens):
        last_tok = tokens[context_start - 1] if entities else tokens[0]
        ctx = ' '.join(tokens[context_start:])
        # replace the last entity to include context
        if entities:
            entities[-1] = f"{action} {last_tok.lower()} {ctx.lower()}"
        else:
            # edge case: no standalone tokens before prep
            entities.append(f"{action} {ctx.lower()}")
    elif not entities:
        # edge case: only one token, no preposition
        entities.append(f"{action} {tokens[0].lower()}")

    return entities



# Main function to predict intents from user query
def predict_intents(user_query):
    intents = []
    for punct_part in split_on_punctuation(user_query):                             # 1️⃣ Sentence-level split
        for wh_part in split_on_wh(punct_part):                                     # 2️⃣ WH or auxiliary phrase detection
            for conj_part in split_on_conjunctions(wh_part):                        # 3️⃣ Conjunction-based phrase isolation
                for multi_action_chunk in split_on_multiple_actions(conj_part):     # 4️⃣ Split on multiple action verbs
                    for smart_chunk in smart_split_imperative(multi_action_chunk):  # 5️⃣ Smart verb-based logic kicks in here
                        ents = extract_entities(smart_chunk, punct_part)            # 6️⃣ Final intent construction
                        intents.extend(ents)
    return intents



queries = [
    "open yt gta5 insta",
    "close yt gta5 insta",
    "open yt gta5 close gta5 yt",
    "play a-song b-song c-song",
    "play a-song b-song c-song d-song e-song in youtube",
    "play a-song b-song c-song d-song e-song on youtube",
    "play a-song b-song c-song d-song e-song plateform is stotify",
    "play a-song b-song c-song d-song e-song play this things in my_music"
]

## Uncomment to test
for q in queries:
    print(f"User query: {q}")
    print(f"splits: {predict_intents(q)}\n")


