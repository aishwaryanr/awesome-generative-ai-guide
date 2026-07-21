"""The model layer, provider-agnostic, plus the two clinical operations built on it.

A clinical scribe needs the model to do two jobs:

- EXTRACT a structured SOAP note (Subjective, Objective, Assessment, Plan) from an
  encounter transcript, and
- CHECK FAITHFULNESS: decide whether every statement in that note is supported by the
  transcript, so a fabricated symptom, diagnosis, or dose is caught before a clinician sees it.

This file gives you two implementations behind each function:

- a deterministic OFFLINE policy so the whole pipeline runs with no API key, which is what
  makes this example runnable and testable in CI, and
- a real path that works with ANY provider through LangChain's init_chat_model. Set a model
  and a key and it uses OpenAI, Anthropic, Gemini, or any supported provider. You do not
  change the graph to change the model.

ALL DATA HERE IS SYNTHETIC. There is no real protected health information anywhere in this
example, and there must never be.

Selecting a model (any one of these):
    export CLINICAL_SCRIBE_MODEL="gpt-4o-mini"        + OPENAI_API_KEY
    export CLINICAL_SCRIBE_MODEL="claude-sonnet-5"    + ANTHROPIC_API_KEY
    export CLINICAL_SCRIBE_MODEL="gemini-2.0-flash"   + GOOGLE_API_KEY
If CLINICAL_SCRIBE_MODEL is unset, the provider is auto-detected from whichever key is present.
Install the matching integration: langchain-openai, langchain-anthropic, or
langchain-google-genai.
"""
import json
import os
import re

SECTIONS = ("S", "O", "A", "P")  # Subjective, Objective, Assessment, Plan

# Keywords that route a transcript line into a SOAP section. Order is checked P, A, O, then
# Subjective as the default, because a plan or assessment line is the more specific signal.
_PLAN_CUES = ("recommend", "prescrib", "order ", "let us order", "follow up", "refer",
              "ice", "rest", "take ", " mg", "x-ray", "return in", "start ")
_ASSESS_CUES = ("consistent with", "looks like", "this looks", "diagnosis", "assessment",
                "likely", "impression", "suggests")
_OBJECTIVE_CUES = ("exam", "temperature", "blood pressure", "swelling", "tender",
                   "range of motion", "instability", "degrees", "vital", "on examination")

# Auto-detect: (env key that must be present) -> (default model, provider)
_AUTODETECT = [
    ("OPENAI_API_KEY", "gpt-4o-mini", "openai"),
    ("ANTHROPIC_API_KEY", "claude-sonnet-5", "anthropic"),
    ("GOOGLE_API_KEY", "gemini-2.0-flash", "google_genai"),
    ("GEMINI_API_KEY", "gemini-2.0-flash", "google_genai"),
]


def _get_model():
    """Return a provider-agnostic chat model, or None to run the offline policy."""
    model = os.environ.get("CLINICAL_SCRIBE_MODEL")
    provider = os.environ.get("CLINICAL_SCRIBE_PROVIDER")
    if not model:
        for key, m, p in _AUTODETECT:
            if os.environ.get(key):
                model, provider = m, p
                break
    if not model:
        return None
    try:
        from langchain.chat_models import init_chat_model
        return init_chat_model(model, model_provider=provider) if provider else init_chat_model(model)
    except Exception:
        return None


# --- shared tokenizer -----------------------------------------------------------------

_STOP = {"the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "is", "are", "was",
         "it", "you", "your", "my", "i", "me", "we", "us", "there", "with", "as", "at",
         "that", "this", "not", "no", "any", "had", "has", "have", "been", "about", "over",
         "most", "when", "but", "let", "us", "per", "am", "pm", "he", "she", "they"}


def _tokens(text):
    """Normalized token set: lowercase words and numbers, punctuation stripped, dots kept."""
    text = text.lower()
    raw = re.split(r"[^a-z0-9.]+", text)
    return {t.strip(".") for t in raw if t.strip(".")}


def _salient(statement):
    """The tokens in a statement that carry clinical meaning and must be grounded."""
    out = []
    for t in _tokens(statement):
        if t in _STOP:
            continue
        if len(t) > 2 or any(c.isdigit() for c in t):
            out.append(t)
    return out


# --- EXTRACT: transcript -> SOAP note -------------------------------------------------

def extract_soap(transcript):
    """Return a SOAP note as {"S":[...], "O":[...], "A":[...], "P":[...]}. Provider-agnostic."""
    model = _get_model()
    if model is not None:
        try:
            return _extract_with_model(model, transcript)
        except Exception:
            pass  # fall back to the offline policy if the provider call fails
    return _extract_offline(transcript)


_SPEAKERS = ("Patient", "Clinician", "Doctor", "Nurse", "Provider")


def _lines(transcript):
    """Split a 'Speaker: text' transcript into (speaker, text) utterances.

    Handles both one-utterance-per-line transcripts and a single line with inline speaker
    labels (e.g. 'Patient: ... Clinician: ...'), by breaking before each known speaker label.
    """
    for name in _SPEAKERS:
        transcript = re.sub(rf"\s+({name}:)", r"\n\1", transcript)
    out = []
    for raw in transcript.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        if ":" in raw:
            speaker, text = raw.split(":", 1)
            out.append((speaker.strip(), text.strip()))
        else:
            out.append(("", raw))
    return out


def _classify(text):
    t = " " + text.lower() + " "
    if any(cue in t for cue in _PLAN_CUES):
        return "P"
    if any(cue in t for cue in _ASSESS_CUES):
        return "A"
    if any(cue in t for cue in _OBJECTIVE_CUES):
        return "O"
    return "S"


def split_utterances(transcript):
    """Public helper: the (speaker, text) utterances the pipeline will read."""
    return _lines(transcript)


def _extract_offline(transcript):
    """Grounded extraction: each SOAP statement is a verbatim utterance from the transcript.

    Because every statement is copied from the transcript, the faithfulness check below finds
    all of them supported. A real model can paraphrase and summarize, which is exactly why the
    faithfulness check exists: to catch the statements a model invents.
    """
    soap = {s: [] for s in SECTIONS}
    for _speaker, text in _lines(transcript):
        soap[_classify(text)].append(text)
    return soap


# --- CHECK FAITHFULNESS: is every statement supported by the transcript? --------------

def check_faithfulness(transcript, soap):
    """Return the list of statements NOT supported by the transcript (candidate fabrications).

    Each item is {"section", "statement", "unsupported"} where unsupported lists the salient
    tokens that never appear in the transcript. An empty list means every statement is grounded.
    """
    model = _get_model()
    if model is not None:
        try:
            return _faithfulness_with_model(model, transcript, soap)
        except Exception:
            pass
    return _faithfulness_offline(transcript, soap)


def _faithfulness_offline(transcript, soap):
    source = _tokens(transcript)
    flags = []
    for section in SECTIONS:
        for statement in soap.get(section, []):
            missing = [tok for tok in _salient(statement) if tok not in source]
            if missing:
                flags.append({"section": section, "statement": statement, "unsupported": missing})
    return flags


# --- real path: any provider via init_chat_model --------------------------------------

_EXTRACT_SYSTEM = (
    "You are a clinical scribe. Read the encounter transcript and write a SOAP note. "
    "Use ONLY information stated in the transcript. Never add a symptom, finding, diagnosis, "
    "medication, or dose that is not in the transcript. Return STRICT JSON with keys "
    "S, O, A, P, each a list of short statement strings. No prose outside the JSON.\n"
    "S = Subjective (what the patient reports). O = Objective (exam findings, vitals). "
    "A = Assessment (the clinician's impression or diagnosis). P = Plan (next steps)."
)

_JUDGE_SYSTEM = (
    "You are a strict faithfulness checker for clinical notes. Given a transcript and one "
    "statement from a draft note, answer with EXACTLY 'SUPPORTED' if the transcript directly "
    "supports the statement, or 'UNSUPPORTED' if it adds anything not present in the transcript. "
    "When in doubt, answer UNSUPPORTED. Reply with the single word only."
)


def _extract_with_model(model, transcript):
    resp = model.invoke([("system", _EXTRACT_SYSTEM), ("human", f"TRANSCRIPT:\n{transcript}")])
    content = (getattr(resp, "content", "") or "").strip()
    content = re.sub(r"^```(json)?|```$", "", content, flags=re.MULTILINE).strip()
    data = json.loads(content)
    return {s: [str(x) for x in data.get(s, [])] for s in SECTIONS}


def _faithfulness_with_model(model, transcript, soap):
    flags = []
    for section in SECTIONS:
        for statement in soap.get(section, []):
            user = f"TRANSCRIPT:\n{transcript}\n\nSTATEMENT: {statement}"
            resp = model.invoke([("system", _JUDGE_SYSTEM), ("human", user)])
            verdict = (getattr(resp, "content", "") or "").strip().upper()
            if verdict.startswith("UNSUPPORTED"):
                flags.append({"section": section, "statement": statement, "unsupported": ["judged unsupported"]})
    return flags
