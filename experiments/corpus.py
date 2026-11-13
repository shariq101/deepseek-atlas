"""
corpus.py — the prompts, chosen as controlled contrasts.

Every prompt ends immediately before the token whose probability we measure.
`target` is the continuation; only its first token is scored. The prompts are
short on purpose: attention is captured densely, so the artefacts stay small
enough to inspect every head at every position.

Groups exist so that routing and depth can be compared *within* a computation
rather than across unrelated text.
"""

PROMPTS = [
    # ---- factual recall -----------------------------------------------------
    dict(id="fact-capital", group="factual", text="The capital of Australia is", target=" Canberra"),
    dict(id="fact-planet", group="factual", text="The largest planet in the solar system is", target=" Jupiter"),
    dict(id="fact-author", group="factual", text="The novel Pride and Prejudice was written by Jane", target=" Austen"),
    dict(id="fact-element", group="factual", text="The chemical symbol for gold is", target=" Au"),
    dict(id="fact-river", group="factual", text="The longest river in Africa is the", target=" Nile"),
    dict(id="fact-currency", group="factual", text="The currency of Japan is the", target=" yen"),

    # ---- syntax: agreement under attraction --------------------------------
    dict(id="syn-singular", group="syntax", text="The collection of ancient coins in the museum's basement", target=" was"),
    dict(id="syn-plural", group="syntax", text="The paintings in the gallery that the curator restored last summer", target=" were"),
    dict(id="syn-attract", group="syntax", text="The key to the cabinets in the hallway", target=" is"),
    dict(id="syn-quote", group="syntax", text="The report, despite the objections from several departments,", target=" was"),

    # ---- induction / copying ----------------------------------------------
    dict(id="ind-1", group="induction", text="A7 Q2 M9 B4 A7 Q2 M9", target=" B"),
    dict(id="ind-2", group="induction", text="zebra compass violin — zebra compass", target=" violin"),
    dict(id="ind-3", group="induction", text="t42 k8 t42 k8 t42", target=" k"),

    # ---- arithmetic --------------------------------------------------------
    dict(id="arith-add", group="arithmetic", text="17 + 26 = ", target="43"),
    dict(id="arith-mul", group="arithmetic", text="9 × 8 = ", target="72"),
    dict(id="arith-chain", group="arithmetic", text="3 × 4 + 5 = ", target="17"),

    # ---- code --------------------------------------------------------------
    dict(id="code-add", group="code", text="def add(a, b):\n    return a + b\n\nadd(3, 4)  # returns ", target="7"),
    dict(id="code-loop", group="code", text="for i in range(3):\n    print(i)\n\n# output starts with ", target="0"),
    dict(id="code-list", group="code", text="nums = [1, 2, 3]\nnums.append(4)\nlen(nums)  # ", target="4"),

    # ---- multilingual lookup ----------------------------------------------
    dict(id="multi-fr", group="multilingual", text="The French word for water is", target=" eau"),
    dict(id="multi-de", group="multilingual", text="The German word for book is", target=" Buch"),
    dict(id="multi-es", group="multilingual", text="The Spanish word for house is", target=" casa"),

    # ---- negation and idiom ------------------------------------------------
    dict(id="neg-always", group="negation", text="The opposite of 'always' is", target=" never"),
    dict(id="neg-whale", group="negation", text="All mammals are warm-blooded, but a whale is not a fish. A whale is a", target=" mammal"),
    dict(id="idiom-bucket", group="idiom", text="When he said he would kick the bucket, he meant he would", target=" die"),
    dict(id="idiom-sword", group="idiom", text="Her argument was a double-edged", target=" sword"),

    # ---- retrieval ---------------------------------------------------------
    dict(id="ret-code", group="retrieval", text="The expedition's access code was 7492, written in the captain's log. When the radio operator asked for the code, she read ", target="7492"),
]

# Deep-dive prompts get full T×T attention matrices and full router vectors.
# Every prompt still gets the final-query per-head attention row used by the atlas.
DEEP = ["syn-attract", "ind-1", "fact-capital", "code-add"]

# --------------------------------------------------------------------------- #
# Probe corpus: 6 properties, 16 prompts each. Probes read the last four
# token positions of each prompt; folds are grouped by prompt id so a probe
# never sees a prompt it was trained on.
# --------------------------------------------------------------------------- #
PROBES = []

_PLURAL = [
    ("The students in the lecture hall", 1), ("The student in the lecture hall", 0),
    ("The keys on the wooden table", 1), ("The key on the wooden table", 0),
    ("The reports from the committee", 1), ("The report from the committee", 0),
    ("The dogs in the park", 1), ("The dog in the park", 0),
    ("The windows of the old house", 1), ("The window of the old house", 0),
    ("The proposals on the agenda", 1), ("The proposal on the agenda", 0),
    ("The machines in the factory", 1), ("The machine in the factory", 0),
    ("The letters from her sister", 1), ("The letter from her sister", 0),
]
for i, (t, y) in enumerate(_PLURAL):
    PROBES.append(dict(id=f"plural-{i}", prop="plural", label=y, text=t + " ", group=f"plural-{i}"))

_LANG = [
    ("The quick brown fox jumps over the lazy dog.", 0),
    ("She sold seashells by the seashore every morning.", 0),
    ("Le renard brun rapide saute par-dessus le chien paresseux.", 1),
    ("Elle vend des coquillages au bord de la mer.", 1),
    ("Der schnelle braune Fuchs springt über den faulen Hund.", 2),
    ("Sie verkauft jeden Morgen Muscheln am Meeresufer.", 2),
    ("El rápido zorro marrón salta sobre el perro perezoso.", 3),
    ("Ella vende conchas en la orilla del mar.", 3),
    ("The weather in the valley was unusually warm today.", 0),
    ("Nous avons marché jusqu'au village avant la pluie.", 1),
    ("Die Bibliothek öffnet am Montag um neun Uhr.", 2),
    ("La biblioteca abre el lunes a las nueve.", 3),
    ("A small boat drifted slowly toward the harbor.", 0),
    ("Les enfants jouent dans le jardin après l'école.", 1),
    ("Der Schüler liest jeden Abend ein Buch.", 2),
    ("Los niños juegan en el jardín después de la escuela.", 3),
]
for i, (t, y) in enumerate(_LANG):
    PROBES.append(dict(id=f"lang-{i}", prop="language", label=y, text=t + " ", group=f"lang-{i}"))

_DOMAIN = [
    ("def train(model, loader):\n    for x, y in loader:\n        loss = model(x, y)\n        loss.backward()\n", 1),
    ("class Node:\n    def __init__(self, value):\n        self.value = value\n", 1),
    ("import numpy as np\narr = np.zeros((3, 4))\n", 1),
    ("for i in range(10):\n    print(i * i)\n", 1),
    ("SELECT name, age FROM users WHERE age > 30;\n", 1),
    ("try:\n    result = compute()\nexcept ValueError:\n    result = None\n", 1),
    ("const total = items.reduce((a, b) => a + b, 0);\n", 1),
    ("fn main() {\n    println!(\"hello\");\n}\n", 1),
    ("The committee met on Tuesday to discuss the annual budget.", 0),
    ("A gentle rain fell over the village throughout the night.", 0),
    ("She opened the letter and read it twice before answering.", 0),
    ("The mountain path narrowed as it climbed above the treeline.", 0),
    ("Negotiations continued for three weeks without a clear result.", 0),
    ("The old librarian remembered every patron by name.", 0),
    ("A warm wind carried the smell of bread through the street.", 0),
    ("The history of the region is recorded in stone and song.", 0),
]
for i, (t, y) in enumerate(_DOMAIN):
    PROBES.append(dict(id=f"domain-{i}", prop="domain", label=y, text=t, group=f"domain-{i}"))

_CATEGORY = [
    ("France", 0), ("Japan", 0), ("Brazil", 0), ("Kenya", 0),
    ("Paris", 1), ("Tokyo", 1), ("Lisbon", 1), ("Nairobi", 1),
    ("eagle", 2), ("salmon", 2), ("tiger", 2), ("beetle", 2),
    ("granite", 3), ("quartz", 3), ("copper", 3), ("marble", 3),
]
for i, (w, y) in enumerate(_CATEGORY):
    PROBES.append(dict(id=f"cat-{i}", prop="category", label=y, text=f"The word '{w}' most naturally names a", group=f"cat-{i}"))

_SENT = [
    ("The film was a joy from beginning to end.", 1),
    ("What a wonderful surprise to see you here.", 1),
    ("The results exceeded every expectation we had.", 1),
    ("Her performance was genuinely moving.", 1),
    ("The meal was delicious and the service impeccable.", 1),
    ("I am delighted with how the project turned out.", 1),
    ("The view from the summit was breathtaking.", 1),
    ("He spoke with warmth and generosity.", 1),
    ("The film was a tedious waste of two hours.", 0),
    ("What a disappointing result after all that work.", 0),
    ("The room was dirty and the staff indifferent.", 0),
    ("Her performance felt hollow and forced.", 0),
    ("The meal was cold and the service painfully slow.", 0),
    ("I regret ever agreeing to this arrangement.", 0),
    ("The view was blocked by an ugly concrete wall.", 0),
    ("He spoke with contempt for everyone present.", 0),
]
for i, (t, y) in enumerate(_SENT):
    PROBES.append(dict(id=f"sent-{i}", prop="sentiment", label=y, text=t, group=f"sent-{i}"))

_PARITY = [
    ("17 + 26 =", 1), ("9 + 8 =", 1), ("14 + 7 =", 1), ("23 + 12 =", 1),
    ("6 + 9 =", 1), ("31 + 4 =", 1), ("8 + 7 =", 1), ("45 + 6 =", 1),
    ("18 + 24 =", 0), ("12 + 8 =", 0), ("27 + 15 =", 0), ("36 + 10 =", 0),
    ("7 + 9 =", 0), ("22 + 6 =", 0), ("41 + 3 =", 0), ("19 + 5 =", 0),
]
for i, (t, y) in enumerate(_PARITY):
    PROBES.append(dict(id=f"parity-{i}", prop="parity", label=y, text=t, group=f"parity-{i}"))

PROBE_PROPS = [
    dict(key="plural", label="subject number", chance=0.5, classes=["singular", "plural"]),
    dict(key="language", label="language", chance=0.25, classes=["EN", "FR", "DE", "ES"]),
    dict(key="domain", label="code vs prose", chance=0.5, classes=["prose", "code"]),
    dict(key="category", label="semantic category", chance=0.25, classes=["country", "city", "animal", "mineral"]),
    dict(key="sentiment", label="sentiment", chance=0.5, classes=["negative", "positive"]),
    dict(key="parity", label="result parity", chance=0.5, classes=["even", "odd"]),
]

# --------------------------------------------------------------------------- #
# Needle retrieval. Filler is neutral prose so the only lexical anchor is the
# code sentence. Depth is where the sentence sits in the context.
# --------------------------------------------------------------------------- #
FILLER = (
    "The archive room was quiet except for the hum of the ventilation. "
    "Shelves of unmarked boxes ran from floor to ceiling, and a narrow aisle "
    "led toward the far wall. Visitors were asked to sign the register before "
    "entering, though few did. The curator had spent thirty years cataloguing "
    "the collection, and still found boxes she did not remember. "
)
NEEDLE = "The expedition's access code was 7492."
NEEDLE_Q = "\nQuestion: What was the expedition's access code? Answer: "
NEEDLE_TARGET = "7492"
NEEDLE_LENGTHS = [256, 512, 1024, 2048, 4096]
NEEDLE_DEPTHS = [0.1, 0.5, 0.9]
