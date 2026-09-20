"""
BlissBot - Step 1: audit + clean the psychoeducational dataset.

Input : PyschoEducationalDataset.json  (raw, 3214 rows)
Output: bliss_corpus_clean.json  (every kept row, with tier/flags/in_index)
        dropped_rows.json        (every removed row + reason)
        audit_stats.json         (numbers used in the audit report)

Run:  python 01_audit_and_clean.py raw.json out_dir/
"""
import json, re, sys, collections, statistics as st
from pathlib import Path
import ftfy

RAW = Path("/workspaces/BlissBot/PyschoEducationalDataset.json")
OUT = Path("/workspaces/BlissBot/out_dir")
OUT.mkdir(parents=True, exist_ok=True)

raw = json.load(open(RAW, encoding="utf-8"))

# ---------- helpers ----------------------------------------------------------
def dataset_of(x):
    u = x["url"]
    if "counsel-chat" in u: return "counsel-chat"
    if "heliosbrahma" in u: return "heliosbrahma"
    if "tolu07" in u: return "tolu07"
    return "nimh"

DATASET_LABEL = {
    "counsel-chat": "nbertagnolli/counsel-chat (therapist replies to forum posts)",
    "heliosbrahma": "heliosbrahma/mental_health_chatbot_dataset",
    "tolu07": "tolu07/Mental_Health_FAQ",
    "nimh": "NIMH fact sheets (LLM-generated Q&A)",
}
TIER = {"nimh": "A", "tolu07": "A", "heliosbrahma": "A", "counsel-chat": "B"}

norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
wc = lambda s: len(s.split())

def clean_text(s, is_question=False):
    s = ftfy.fix_text(s)                       # mojibake: personâ€™s -> person's
    s = s.replace("\xa0", " ").replace("\r", "")
    s = re.sub(r"(?<=[a-z]{2})([.!?])(?=[A-Z][a-z])", r"\1 ", s)   # "end.Sometimes" -> "end. Sometimes"
    if is_question:
        s = re.sub(r"\s+", " ", s)
    else:
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r"\n\s*\n\s*\n+", "\n\n", s)
        s = re.sub(r" *\n *", "\n", s)
    return s.strip()

NAN_LIKE = {"nan", "none", "null", "n/a", ""}
PHONE = re.compile(r"\b\(?\d{3}\)?[-. ]\d{3}[-. ]\d{4}\b")
PROMO = re.compile(r"((?<!in )\bmy (latest )?(blog|website|profile|book|video)\b|\bcontact me\b|\bemail me\b|"
                   r"\b(reach|call|text) me (at|on)\b|\bfeel free (to )?(contact|reach|email|call) (me|us)\b|\bfeel free (to )?contact\b(?! them)|"
                   r"\bvisit (my|our)\b|\bour (in-house|practice|clinic)\b|\bcheck (this|my) (blog|website)\b|"
                   r"\bcheck (this )?blog out\b|\bfree consultation with (our|my)\b)", re.I)
CLINICIAN_VOICE = re.compile(r"\b(in my experience|(?<!such )as a (therapist|counselor|psychologist|clinician)|my clients|"
                             r"in my (practice|office)|i work with|i often (see|tell)|when i work with)\b", re.I)
REGION_BC = re.compile(r"\bBC\b|British Columbia|Canada|604-|\bCOVID\b", re.I)
US_HOTLINE = re.compile(r"\b988\b|741741|Lifeline|Crisis Text Line|NAMI|\b911\b|SAMHSA|1-800|800-\d{3}-\d{4}", re.I)
AI_PERSONA = re.compile(r"i am not a (medical|mental health)|i'm not a (medical|mental health)|as an ai|language model", re.I)
HAS_URL = re.compile(r"https?://|www\.|\b\w+\.(com|org|net)\b", re.I)

CHITCHAT_Q = re.compile(r"^(hello|hi|hey|bye|goodbye|thanks?)\b|who are you|what are you doing|how are you|"
                        r"^please help me|i am feeling lost|(who|which) is the best|best (mental health )?(hospital|psychiatrist|hypnotherapist)|"
                        r"helpline number", re.I)
BC_NAV_Q = re.compile(r"\b(where can|where do|find (a|an|other|local|free|help|support|child|older|information|self-help)|"
                      r"see a|see an|apply|pay for|help paying|referral|support group|local|inpatient|MSP|"
                      r"income assistance|low-cost|counsell?or)\b|cannabis|CBD|vaping", re.I)
FIRST_PERSON_CRISIS_Q = re.compile(r"feeling suicidal|stop suicidal|i have thoughts of suicide|having thoughts of suicide", re.I)
SAFETY_TERMS = re.compile(r"suicid|self[- ]harm|kill myself|end my life|want to die|hurt myself", re.I)

# ---------- topic normalisation ---------------------------------------------
def slug(t):
    t = t.lower().replace("&", "and")
    t = re.sub(r"[^a-z0-9]+", "_", t).strip("_")
    return t

GROUPS = [
    ("suicide_selfharm_crisis", ["suicide", "self_harm", "crisis"]),
    ("eating_disorders", ["eating"]),
    ("substance_addiction", ["addiction", "substance"]),
    ("trauma_ptsd", ["trauma", "ptsd"]),
    ("psychosis_personality", ["schizo", "psychosis", "borderline"]),
    ("anxiety_stress", ["anxiety", "panic", "phobia", "stress", "ocd", "obsessive"]),
    ("depression_mood", ["depress", "bipolar", "mood_dysreg", "seasonal_affective"]),
    ("neurodevelopmental", ["adhd", "attention_deficit", "autism", "pans_pandas"]),
    ("grief", ["grief"]),
    ("treatment_and_help", ["treatment", "therapy", "medication", "finding_help", "professionals",
                            "health_care_provider", "cost_and_insurance", "counseling", "professional_ethics", "do_i_need_help"]),
    ("coping_wellbeing", ["coping", "self_care", "self_management", "lifestyle", "sleep", "self_esteem",
                          "anger", "behavioral_change"]),
    ("relationships_family", ["relationship", "marriage", "intimacy", "family", "parenting", "social_",
                              "domestic", "lgbtq", "sexuality", "workplace_rel"]),
    ("general_literacy", ["general", "stigma", "symptoms", "diagnosis", "mental_health_", "risk_factors",
                          "genetics", "teen_brain", "children", "physical_health", "workplace_mental"]),
]
def topic_group(s):
    for g, keys in GROUPS:
        if any(k in s for k in keys): return g
    return "other"

# ---------- pass 1: clean + drop ---------------------------------------------
kept, dropped = [], []
seen_pair, seen_answer = {}, {}
def drop(x, reason, extra=None):
    dropped.append({"orig_id": x["id"], "dataset": x["_ds"], "reason": reason,
                    "question": x["question"][:200], "answer": x["answer"][:200], **(extra or {})})

for i, x in enumerate(raw):
    x = dict(x); x["_ds"] = dataset_of(x); x["_row"] = i
    q_raw, a_raw = x["question"], x["answer"]
    if q_raw.strip().lower() in NAN_LIKE:  drop(x, "question_missing"); continue
    if a_raw.strip().lower() in NAN_LIKE:  drop(x, "answer_missing");   continue
    q, a = clean_text(q_raw, True), clean_text(a_raw)
    x["question"], x["answer"] = q, a
    key = (norm(q), norm(a))
    if key in seen_pair:      drop(x, "exact_duplicate_pair", {"duplicate_of_row": seen_pair[key]}); continue
    if x["_ds"] == "counsel-chat":
        m = PROMO.search(a) or (PHONE.search(a) if not re.search(r"988|741741|hotline|lifeline|crisis|1-800|800-|911|211|2-1-1", a, re.I) else None)
        if m:
            i0 = max(0, m.start() - 40)
            drop(x, "promo_or_solicitation", {"matched_context": a[i0:m.end() + 40].replace("\n", " ")}); continue
        if wc(a) < 20:        drop(x, "answer_too_short_lt20w"); continue
    if norm(a) in seen_answer and norm(seen_answer[norm(a)][0]) != norm(q):
        drop(x, "same_answer_reused_for_other_question", {"first_seen_question": seen_answer[norm(a)][0][:120]}); continue
    seen_pair[key] = x["_row"]; seen_answer.setdefault(norm(a), (q, x["_row"]))
    kept.append(x)

# cross-source duplicate questions inside tier A (tolu07 vs heliosbrahma): keep tolu07 copy if answers ~identical
def jacc(a, b):
    A, B = set(norm(a).split()), set(norm(b).split()); return len(A & B) / max(1, len(A | B))
byq = collections.defaultdict(list)
for x in kept:
    if TIER[x["_ds"]] == "A": byq[norm(x["question"])].append(x)
remove_rows = set()
for q, rows in byq.items():
    if len(rows) > 1:
        rows.sort(key=lambda r: 0 if r["_ds"] == "tolu07" else 1)
        for r in rows[1:]:
            if jacc(rows[0]["answer"], r["answer"]) >= 0.80:
                remove_rows.add(r["_row"]); drop(r, "cross_source_duplicate_question_near_identical_answer",
                                               {"kept_dataset": rows[0]["_ds"]})
kept = [x for x in kept if x["_row"] not in remove_rows]

# ---------- pass 2: flags, ids, tiers, index policy --------------------------
prefix_groups = collections.defaultdict(list)
for x in kept:
    if TIER[x["_ds"]] == "A": prefix_groups[(x["_ds"], norm(x["answer"])[:120])].append(x)
shared_prefix_rows = {r["_row"] for v in prefix_groups.values() if len(v) > 1 for r in v}

# confirmed by manual read: tolu07 "What causes mental illness?" carries the answer for "Who does mental illness affect?"
CONFIRMED_MISMATCH = lambda x: x["_ds"] == "tolu07" and norm(x["question"]) == "what causes mental illness"

cc_counter = collections.Counter()
docs = []
for x in kept:
    ds = x["_ds"]; a, q = x["answer"], x["question"]
    flags = []
    if HAS_URL.search(a): flags.append("url_in_answer")
    if US_HOTLINE.search(a): flags.append("hotline_reference")
    if ds == "tolu07" and REGION_BC.search(a): flags.append("region_specific_BC_Canada")
    if CLINICIAN_VOICE.search(a): flags.append("clinician_voice")
    if AI_PERSONA.search(a): flags.append("ai_persona_text")
    if x["_row"] in shared_prefix_rows: flags.append("shared_answer_preamble")
    if ds == "counsel-chat" and wc(q) > 60: flags.append("long_personal_post_question")
    if wc(a) > 400: flags.append("very_long_answer")
    if CONFIRMED_MISMATCH(x): flags.append("answer_mismatch_confirmed")

    if ds == "counsel-chat":
        qg = f"cc_q{x['id']}"; cc_counter[qg] += 1
        doc_id = f"{qg}_a{cc_counter[qg]:03d}"
    elif ds == "nimh":
        qg = doc_id = str(x["id"])
    else:
        prefix = "tolu" if ds == "tolu07" else "helios"
        qg = doc_id = f"{prefix}_{x['id']}"
    if SAFETY_TERMS.search(q + " " + a): flags.append("safety_sensitive")
    if FIRST_PERSON_CRISIS_Q.search(q): flags.append("first_person_crisis_question")
    region = "US" if ds == "nimh" else ("BC-CA" if "region_specific_BC_Canada" in flags else None)
    slugt = slug(x["topic"])
    docs.append({
        "doc_id": doc_id, "qgroup": qg, "orig_id": x["id"], "tier": TIER[ds], "dataset": ds,
        "source_label": DATASET_LABEL[ds], "orig_source_field": x["source"],
        "topic": x["topic"], "topic_norm": slugt, "topic_group": topic_group(slugt),
        "question": q, "answer": a, "is_synthetic": x["is_synthetic"], "url": x["url"],
        "region": region, "q_words": wc(q), "a_words": wc(a), "flags": flags, "in_index": True, "index_exclusion_reason": None,
    })

# index policy: exclude confirmed-bad; cap counsel-chat at 5 answers per question
for d in docs:
    if "answer_mismatch_confirmed" in d["flags"]:
        d["in_index"] = False; d["index_exclusion_reason"] = "answer_mismatch_confirmed"
for d in docs:
    if not d["in_index"]: continue
    if d["dataset"] == "heliosbrahma" and CHITCHAT_Q.search(d["question"]):
        d["in_index"] = False; d["index_exclusion_reason"] = "not_knowledge_chitchat_or_refusal_template"
    elif d["dataset"] == "tolu07" and "region_specific_BC_Canada" in d["flags"] and BC_NAV_Q.search(d["question"]):
        d["in_index"] = False; d["index_exclusion_reason"] = "region_specific_service_navigation_BC"
    elif "first_person_crisis_question" in d["flags"]:
        d["in_index"] = False; d["index_exclusion_reason"] = "route_to_safety_layer_not_llm_retrieval"
    elif d["dataset"] == "counsel-chat" and "safety_sensitive" in d["flags"]:
        d["in_index"] = False; d["index_exclusion_reason"] = "safety_sensitive_tierB"
CAP = 5
grp = collections.defaultdict(list)
for d in docs:
    if d["dataset"] == "counsel-chat" and d["in_index"]: grp[d["qgroup"]].append(d)
for qg, rows in grp.items():
    def rank(d):
        penalty = ("clinician_voice" in d["flags"]) * 2 + ("url_in_answer" in d["flags"]) + ("very_long_answer" in d["flags"])
        return (penalty, abs(d["a_words"] - 150))
    rows.sort(key=rank)
    for d in rows[CAP:]:
        d["in_index"] = False; d["index_exclusion_reason"] = f"over_group_cap_{CAP}"

# ---------- stats -------------------------------------------------------------
def pct(v, n): return round(100 * v / n, 1)
raw_ds = collections.Counter(dataset_of(x) for x in raw)
stats = {
    "raw_rows": len(raw), "raw_by_dataset": raw_ds,
    "raw_id_type_int": sum(isinstance(x["id"], int) for x in raw),
    "raw_rows_with_shared_id": sum(v for v in collections.Counter(str(x["id"]) for x in raw).values() if v > 1),
    "raw_unique_ids": len({str(x["id"]) for x in raw}),
    "dropped_total": len(dropped), "dropped_by_reason": collections.Counter(d["reason"] for d in dropped),
    "kept_rows": len(docs), "kept_by_dataset": collections.Counter(d["dataset"] for d in docs),
    "kept_by_tier": collections.Counter(d["tier"] for d in docs),
    "excluded_from_index_by_reason": collections.Counter(d["index_exclusion_reason"] for d in docs if not d["in_index"]),
    "in_index_rows": sum(d["in_index"] for d in docs),
    "in_index_by_tier": collections.Counter(d["tier"] for d in docs if d["in_index"]),
    "in_index_by_dataset": collections.Counter(d["dataset"] for d in docs if d["in_index"]),
    "flag_counts": collections.Counter(f for d in docs for f in d["flags"]),
    "raw_distinct_topics": len({x["topic"] for x in raw}),
    "distinct_topic_norm": len({d["topic_norm"] for d in docs}),
    "topic_groups": collections.Counter(d["topic_group"] for d in docs if d["in_index"]),
    "cc_question_groups": len(grp),
    "cc_groups_over_cap": sum(len(v) > CAP for v in grp.values()),
    "cc_rows_in_groups_ge10_raw": None,
    "answer_words_by_dataset": {ds: {"median": st.median([d["a_words"] for d in docs if d["dataset"] == ds]),
                                       "p90": sorted([d["a_words"] for d in docs if d["dataset"] == ds])[int(.9 * sum(d["dataset"] == ds for d in docs))],
                                       "max": max(d["a_words"] for d in docs if d["dataset"] == ds)} for ds in DATASET_LABEL},
}
# raw cluster concentration
c = collections.Counter(str(x["id"]) for x in raw if dataset_of(x) == "counsel-chat")
stats["cc_rows_in_groups_ge10_raw"] = sum(v for v in c.values() if v >= 10)
stats["cc_groups_ge10_raw"] = sum(v >= 10 for v in c.values())

json.dump(docs, open(OUT / "bliss_corpus_clean.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
json.dump(dropped, open(OUT / "dropped_rows.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
json.dump(stats, open(OUT / "audit_stats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=dict)
print(json.dumps(stats, indent=1, default=dict))
