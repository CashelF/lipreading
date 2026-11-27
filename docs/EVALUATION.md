# Word accuracy evaluation

Still had no measured word accuracy on webcam video. The FLARE paper's 13.2% LRS3 word error rate (WER) describes broadcast TED clips, not this app. This guide covers the held-out webcam set used to measure it. Any smaller model that replaces FLARE must be compared on the same set ([SMALL_MODEL_PLAN.md](SMALL_MODEL_PLAN.md)).

## Record

Open **Help → Troubleshooting → Record sentences**. Enter a speaker code, which is any short code, one per person, such as initials. Then read each prompt once:

1. Press **Record**. After a 3-second countdown, start with your mouth closed.
2. Read the sentence naturally. Don't exaggerate it.
3. Close your mouth, then press **Done** or Space. Clips can run from 1.5 to 15 seconds.

The panel moves to the next sentence that speaker hasn't recorded yet. If you stumble, use **Delete last recording** and record the sentence again. The 100 prompts in `backend/eval_sentences.json` are original everyday sentences with numbers spelled out. Their ids never change, so a clip always keeps its reference text. Recording all 100 takes about 20 minutes per speaker.

Clips are saved as the original JPEG frames in the ignored `data/eval-sentences/` folder on the inference machine. No audio is captured.

**Keep this set held out.** Never tune thresholds, prompts, decoding or training on these clips. Use the speaking examples or other data for development. More speakers, rooms, cameras and lighting conditions make the number more meaningful than more sentences from one person.

## Score

```bash
# What a person sees: real-time replay through a ready, idle server (gate and checks on).
.venv-model/bin/python scripts/evaluate_sentences.py --mode live

# The model alone: one whole-clip decode per recording, gate and checks off.
.venv-model/bin/python scripts/evaluate_sentences.py --mode model
```

- **Live mode** replays each clip's timestamped frames through `/api/stream` at the original pace, exactly as a browser would, then scores the final transcript after Stop. Speech the gate withholds counts as deletions. It stops if someone starts a session.
- **Model mode** loads FLARE in the script's own process, so it needs about 28 GiB of free GPU memory. Use it to compare recognizers.

Reports go to `models/eval/` by default (owner-only permissions). Each holds corpus WER (total errors ÷ total reference words), a 95% bootstrap interval resampled over recordings, and counts of substitutions, deletions and insertions. It also has exact-sentence counts, failure reasons (`no_text`, `quality`, `error`) and per-speaker WER. Rows contain prompt ids and numbers only: **no recognized text**.

Scoring lowercases text, drops punctuation and treats curly and straight apostrophes the same; hyphens split words. Contractions must match as written ("I'm" ≠ "I am").

## Reading the number

- With a few hundred words, the interval is wide. Report it with the WER.
- Compare models only on the same recordings, in the same mode.
- Read prompts are cleaner than spontaneous speech, and the speakers know which sentences to expect. Treat the result as an optimistic estimate for conversation.
