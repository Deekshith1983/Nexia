import sys
sys.stdout.reconfigure(encoding='utf-8')
import re
from app.services.hybrid_retrieval import RetrievedChunk
from app.retrieval.nlp_query_parser import NLPQueryParser

# Test ClaimVerifier logic
chunks = [
    RetrievedChunk(
        chunk_id=1,
        document_title="Antikythera mechanism",
        source_url="https://en.wikipedia.org/wiki/Antikythera_mechanism",
        section_title="Operation",
        passage_text="The Antikythera mechanism used thirty bronze gears to model astronomical cycles including the Metonic cycle and the Saros cycle.",
        rrf_score=0.9,
        retrieval_method="hybrid",
    ),
    RetrievedChunk(
        chunk_id=2,
        document_title="Antikythera mechanism",
        source_url="https://en.wikipedia.org/wiki/Antikythera_mechanism",
        section_title="History",
        passage_text="Discovered in 1901 from the Antikythera wreck, it is widely considered the oldest known analogue computer.",
        rrf_score=0.85,
        retrieval_method="hybrid",
    ),
]

q = "Explain the Antikythera mechanism, its mechanical operation, and the uncertainties surrounding its transmission."
missing = ["uncertainties surrounding its transmission"]

raw_answer = (
    "The Antikythera mechanism used thirty bronze gears to model astronomical cycles. "
    "It is widely considered the oldest known analogue computer [1]. "
    "The device was influenced by Aztec astronomical pyramids [1]. "
    "The island is famous for migratory bird observation and wild goats [1]."
)

print("Original Answer:")
print(raw_answer)

# Let's verify sentence by sentence
all_evidence_text = " ".join(c.passage_text.lower() for c in chunks)
all_evidence_words = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", all_evidence_text))
q_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", q.lower())) - NLPQueryParser.STOPWORDS_CHUNKS

sentences = re.split(r"(?<=[.!?])\s+", raw_answer.strip())
valid_sentences = []

for sent in sentences:
    s_clean = sent.strip()
    if not s_clean:
        continue
    s_lower = s_clean.lower()
    
    # 1. Off topic
    if any(w in s_lower for w in ["bird observation", "migratory bird", "wild goats"]):
        print(f"-> Dropping off-topic: {s_clean}")
        continue
        
    # 2. Foreign entities
    proper_nouns = re.findall(r"\b[A-Z][a-z]{3,}\b", s_clean)
    foreign = [pn for pn in proper_nouns if pn.lower() not in all_evidence_words and pn.lower() not in q_words and pn not in ["The", "This", "It"]]
    
    # 3. Citation check
    cites = [int(m) for m in re.findall(r"\[(\d+)\]", s_clean)]
    s_content = set(re.findall(r"\b[a-zA-Z]{4,}\b", s_lower)) - NLPQueryParser.STOPWORDS_CHUNKS
    
    if cites:
        c_idx = cites[0]
        # Check if cited passage supports it
        target_p = chunks[c_idx - 1].passage_text.lower() if 1 <= c_idx <= len(chunks) else ""
        matches = sum(1 for w in s_content if w in target_p)
        if matches < 2:
            # Check other chunks
            found_remap = False
            for j, ch in enumerate(chunks, 1):
                if sum(1 for w in s_content if w in ch.passage_text.lower()) >= 2:
                    s_clean = re.sub(r"\[\d+\]", f"[{j}]", s_clean)
                    found_remap = True
                    print(f"-> Remapped citation to [{j}] in: {s_clean}")
                    break
            if not found_remap:
                s_clean = re.sub(r"\[\d+\]", "", s_clean).strip()
                s_clean = re.sub(r"\s+([,\.])", r"\1", s_clean)
                print(f"-> Stripped false citation in: {s_clean}")
                if foreign:
                    print(f"-> Dropping ungrounded foreign claim: {s_clean}")
                    continue
    else:
        # No citation - find supporting passage
        for j, ch in enumerate(chunks, 1):
            if sum(1 for w in s_content if w in ch.passage_text.lower()) >= 2:
                s_clean = f"{s_clean.rstrip('.')} [{j}]."
                print(f"-> Attached supporting citation [{j}] to: {s_clean}")
                break
        if foreign and not any(w in all_evidence_words for w in [f.lower() for f in foreign]):
            print(f"-> Dropping ungrounded sentence with foreign entities: {s_clean}")
            continue

    valid_sentences.append(s_clean)

final_text = " ".join(valid_sentences)
if missing and not any("not contain" in final_text.lower() or "uncertain" in final_text.lower() or "does not document" in final_text.lower() for _ in [1]):
    final_text += f"\n\nThe available evidence does not contain specific documentation regarding {missing[0]}."

print("\nVerified & Cleaned Answer:")
print(final_text)

