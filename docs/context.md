# Context and terminology

## The short version

A dropdown that says “sermon” cannot make the current 77M OPUS-MT model understand a
sermon. The model has no instruction or prompt channel. What a context profile *can* do
today is apply conservative terminology preferences, and what it *should* do next is
select a domain-specific fine-tune.

Danglish therefore supports two layers:

1. **Context profiles** (implemented): terminology post-processing, gated on the Danish
   source, plus a placeholder for a future domain model.
2. **Domain fine-tuning** (recommended next): train a sermon-specific model, convert it to
   CTranslate2, and point a profile at it. The model stays the same size at inference.

## Current profiles

```bash
danglish contexts
danglish translate --context sermon "Menigheden samledes søndag."
danglish talk --context sermon
```

- `general` — no terminology overrides.
- `sermon` — prefers congregational terminology:
  - *menighed* → **congregation**, not “church” when the source is about the people;
  - *prædiken* → **sermon**;
  - *præst* → **pastor**;
  - *gudstjeneste* → **worship service**;
  - *nadver* → **Holy Communion**;
  - *helligånd* → **the Holy Spirit**;
  - *frelser* → **Savior**.

The rules are deliberately conservative:

- they only run when the Danish source contains the corresponding term;
- a rule is skipped if the English word occurs more than once (replacing both could
  corrupt the sentence);
- the *menighed* rule is skipped when the source also contains *kirke*, because “church”
  may then refer to the building;
- capitalization is preserved for common nouns, and proper names use their fixed
  capitalization.

This is not discourse context. It will not resolve pronoun ambiguity, understand the
theme of a paragraph, or change the model's style globally. It also cannot fix a term the
model renders as a synonym that is not listed in the profile.

## Why not just prepend “This is a church sermon” to the source?

OPUS-MT is a Marian encoder-decoder trained for sentence translation. It was not trained
with instructions, chat templates, language tags, or context tokens. Prefixing arbitrary
text would be translated as part of the sentence and usually make the output worse.

A model can be *trained* to accept a context token, for example:

```text
<sermon> Menigheden samledes søndag.
```

The same sentence with `<medical>` could learn a different vocabulary. That requires
fine-tuning on domain-tagged parallel data; the token alone does nothing zero-shot.

## The practical path to better domain translation

### 1. Build an evaluation set first

Collect 200–500 held-out Danish sermon sentences with good English references. Track:

- chrF/BLEU against the references;
- terminology accuracy for a list such as congregation, grace, sin, salvation, communion;
- a human review of pronouns and quoted Scripture;
- latency and peak RSS, because quality changes that break the 2 GB budget are not
  acceptable.

Without this, it is impossible to know whether a glossary or a fine-tune helped.

### 2. Fine-tune OPUS-MT for the domain

The 77M model is cheap to fine-tune compared with an LLM:

1. Prepare parallel sermon data (existing translated sermons, licensed subtitles, or a
   curated set you have rights to use).
2. Fine-tune `Helsinki-NLP/opus-mt-da-en` with Hugging Face Transformers, optionally
   prepending a `<sermon>` token to every source sentence.
3. Convert the result with `danglish fetch --model-id <your-model>` or the CTranslate2
   converter.
4. Add a profile with `model_path` pointing at the converted model. The desktop/CLI can
   then select that profile, and the rest of the pipeline is unchanged.
5. Re-run the evaluation and the memory benchmark.

A domain fine-tune is the best quality-per-megabyte option here: inference still needs
roughly the same memory as the base model, and the UI already has a profile selector.

### 3. Add limited discourse context

If terminology is not enough, fine-tune with the previous sentence as context, using an
explicit separator and a target that contains only the current sentence. This can help with
pronouns and continuity, but it needs careful data preparation and evaluation. Simply
concatenating sentences at inference time and hoping the model translates only the last
one is unreliable.

### 4. Optional small-LLM post-editor

A 0.5B–1.5B instruction model could revise the MT output with a prompt such as “make this
sound like a sermon; keep the meaning; use congregation/pastor terminology”. This can
improve style but is much slower on CPU and may push the full stack over 2 GB. It belongs
behind an opt-in “enhanced quality” mode on a larger machine, not in the default path.

## Design notes for contributors

- Profiles live in `src/danglish/context.py`.
- The pipeline applies a profile after the backend returns each sentence, including
  partial previews and document translation.
- A future profile may set `model_path`; the selection plumbing is intentionally present
  even though no domain model ships yet.
- Glossary rules should be conservative. Add an evaluation example for every rule.
- Never claim a profile gives the model “context” in the model-conditioning sense. Be
  precise: it is terminology preference plus optional domain weights.
