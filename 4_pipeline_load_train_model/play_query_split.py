import re


SUFFIX_KEYWORDS = [ "in", "on", "at", "over", "into", "inside", "within", "through", "from", "to", "using", "via", "with", "plateform is", "platform is", "platform", "plateform","platform be", "this things in", "this in", "through this", "play through", "play using", "sent to", "put in", "queue in", "my music", "in my music", "inside my playlist", "in my collection", "to my playlist", "in favorite app", "spotify", "youtube", "apple music", "amazon music", "wynk", "gaana", "jio saavn", "soundcloud", "youtube music", "your music app", "the music app", "through my device", "in the background", "on speaker", "on bluetooth", "on headphones", "via bluetooth", "on my phone", "on car stereo", "on smart tv", "in home system", "to speakers", "on loop", "on repeat", "in current playlist", "through connected app", "to stream", "through casting", "in queue", "for streaming", "using external app"
]
SUFFIX_KEYWORDS += [ "through music system", "in favorite playlist", "on home speakers", "on bluetooth speaker", "in daily mix", "on personal account", "on the audio app", "on linked app", "via mobile player", "through headphones", "to my sound system", "on the music tab", "on media dashboard", "via home assistant", "through music plugin", "in this session", "in next queue", "to music center", "with audio setup", "via app shortcut", "using smart music", "on my audio gear", "to stream engine", "through home media", "in connected device", "on your dashboard", "on my playlist", "on my stream list", "in the device queue", "in music automation"
]

SUFFIX_KEYWORDS += [ "on this app", "on my system", "in the playlist", "on your favorite app", "on the web", "in the cloud", "to music device", "in the app", "on the interface", "to the player", "on that platform", "in audio mode", "through app integration", "to the default player", "in the default app", "using media player", "on entertainment system", "on audio system", "through cast", "using smart assistant", "via casting device", "via chrome cast", "to smart hub", "on personal player", "through automation", "with my system", "on playback device", "via remote play", "through linked device", "in entertainment zone", "to background music", "on high volume", "in music session", "on audio channel", "through selected source", "on connected screen", "via music extension", "to wireless speaker", "on synced device", "on current device"
]
SUFFIX_KEYWORDS += [
    "on loudspeaker","to the default service","in party mode","via main output","on selected app","to car audio","on linked profile","in your stream list","on soundbar","through echo dot","on google home","in chill playlist","on playback queue","in custom list","from the library","in private mode","on kids profile","on TV output","through my account","to connected audio","to this environment","on this setup","on the cast group","in family playlist","in travel mix","through music controller","on portable speaker","in focus mode","on classic mode","using voice control"
]


SEP_KEYWORDS = [ "and", "or", "as well as", "along with", "also", "with", "plus", "besides", "together with", "in addition to", "and also", "and then", "or even", "next", "then", "followed by", "after that", "&", ",", "not to forget", "including", "even", "additionally", "even also", "maybe", "too", "one by one", "separately", "individually", "back to back", "afterwards", "after each", "then after", "next up", "step by step", "and the next", "continue with", "proceed to", "then play", "not only", "later", "onwards", "right after", "successively"
]
SEP_KEYWORDS += [
    "right after that", "also try", "the next one", "mix in", "as extra", "alongside", "throw in", "pair with", "tie in", "link it with", "and follow with", "switch to", "immediately then", "directly after", "in continuation", "consecutively", "rolling with", "hooked with", "added with", "line it up with", "then fire", "don’t forget", "in a row", "also consider", "don’t miss", "the one after", "loop with", "chain it", "the upcoming", "ride on with"
]
SEP_KEYWORDS += [
    "plus", "even", "then hit me with", "and throw in", "to top it off", "besides that", "oh and", "then bring", "followed by", "queue next", "on the go with", "right then", "next we go with", "and finally", "we move to", "step ahead with", "throw the next", "one more", "and move ahead", "rolling to", "don’t skip", "you can add", "and after that", "one after another", "line up with", "all together", "keep stacking", "and attach", "with the flow", "continue chain"
]

SEP_KEYWORDS += [ "then again", "after that one", "back again", "subsequently", "play later", "eventually", "stepwise", "and next one", "next in line", "sequentially", "on the list", "continue next", "queued with", "combined with", "furthermore", "onward to", "immediately after", "over time", "go on with", "chained to", "next track", "follow up with", "ride along", "attach also", "do include", "let's add", "again with", "jump to", "resume with", "move to", "then hit", "bring next", "trail with", "then load", "replay with", "keep going with", "end with"
]

# Patterns
# Compile patterns
SUFFIX_PATTERN = re.compile(
    r'\b(?:' + '|'.join(re.escape(k) for k in SUFFIX_KEYWORDS) + r')\b.*$',
    flags=re.IGNORECASE
)

SEP_PATTERN = re.compile(
    r'\s*(?:' + '|'.join(re.escape(k) for k in SEP_KEYWORDS) + r')\s*',
    flags=re.IGNORECASE
)


def extract_titles(query: str) -> list[str]:
    """
    Returns raw song titles (with suffix still attached to each).
    """
    # 1) quoted?
    quoted = re.findall(r"""['"]([^'"]+)['"]""", query)
    if quoted:
        last_q = max(query.rfind("'"), query.rfind('"'))
        suffix = query[last_q+1:].strip()
        suffix = re.sub(r'^(play\s*|song\s*)', '', suffix, flags=re.IGNORECASE).strip(' ,')
        return [f"{title} {suffix}".strip() for title in quoted]

    # 2) pull off suffix if present
    m = SUFFIX_PATTERN.search(query)
    if m:
        suffix    = query[m.start():].strip()
        song_part = query[:m.start()].strip()
    else:
        suffix    = ""
        song_part = query.strip()

    # 3) drop leading 'play' or 'play song'
    song_part = re.sub(r'^(play\s+song|play)\b\s*', '', song_part, flags=re.IGNORECASE).strip()

    # 4) split on separators
    raw = [s.strip() for s in SEP_PATTERN.split(song_part) if s.strip()]

    # 5) remove trailing word "song" if it’s part of the raw title
    cleaned = []
    for s in raw:
        s = re.sub(r'\b(song)\b$', '', s, flags=re.IGNORECASE).strip()
        cleaned.append(f"{s} {suffix}".strip() if suffix else s)
    return cleaned

def format_for_play(raw_titles: list[str]) -> list[str]:
    """
    Wraps each title in "play song: '<...>'".
    """
    return [f"play song: '{t}'" for t in raw_titles]


if __name__ == "__main__":

    queries = [
    "play soniya muje pyaar hai",
    "play soniya muje pyaar hai and shape of you in youtube",
    "play soniya muje pyaar hai, lovely day and shape of you in my music",
    # ["play soniya muje pyaar hai in my music", "lovely day in my music", "shape of you in my music"]
    "play soniya muje pyaar hai and sunshine or lovely day as well as three little birds also shape of you in spotify",
    # [soniya muje pyaar hai in spotify, shape of you in spotify, sunshine in spotify, lovely day in spotify, three little birds in spotify]
    "play shape of you in apple music",
    "play a b-song c-song",
    "play a-song b-song c-song d-song e-song in youtube",
    "play a-song b-song c-song d-song e-song on youtube",
    "play a-song b-song c-song d-song e-song plateform is stotify",
    "play 'shape of you', 'blue', 'i love you' play in my_music"
]

    queriess = [
        "play soniya muje pyaar hai and sunshine or lovely day as well as three little birds also shape of you song in spotify",
        "play song soniya muje pyaar hai and sunshine or lovely day as well as three little birds also shape of you in spotify",
        "play song shape of you in apple music",
        "play song honey homko pyaar hai",
        "play song soniya muje pyaar hai",
    ]

    for q in queries:
        titles = extract_titles(q)
        formatted = format_for_play(titles)
        # print as a Python list literal
        print(f"\n# Query:\n{q!r}")
        print(f"{formatted}\n")


# queries = [
#     "play soniya muje pyaar hai",
#     "play soniya muje pyaar hai and shape of you in youtube",
#     "play soniya muje pyaar hai, lovely day and shape of you in my music",
#     # ["play soniya muje pyaar hai in my music", "lovely day in my music", "shape of you in my music"]
#     "play soniya muje pyaar hai and sunshine or lovely day as well as three little birds also shape of you in spotify",
#     # [soniya muje pyaar hai in spotify, shape of you in spotify, sunshine in spotify, lovely day in spotify, three little birds in spotify]
#     "play shape of you in apple music",
#     "play a b-song c-song",
#     "play a-song b-song c-song d-song e-song in youtube",
#     "play a-song b-song c-song d-song e-song on youtube",
#     "play a-song b-song c-song d-song e-song plateform is stotify",
#     "play 'shape of you', 'blue', 'i love you' play in my_music"
# ]


