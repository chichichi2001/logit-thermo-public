# logit-thermo

Code, registration documents, predictions and summary results for the paper
"Predicting how repetition penalties act in language models from unpenalized samples" (Chisa Honda).

- `paper/`: manuscript (`main.tex`, built from `review_copy_v05.md` by `scripts/make_revtex.py`), figures and references.
- `docs/A2_*_GATE.md`: dated registration documents ("gates"), written and frozen before each test or analysis; `docs/A2_*_RESULT.md`: result notes. Most are in Japanese.
- `predictions/`: predictions recorded before each test.
- `results/`: summary results (JSON) produced by the analysis scripts. No generated text or article text is included.
- `scripts/`: analysis and data-collection code; `notebooks/`: the Colab notebooks used for generation.
- `data/articles_meta/`: month, SHA-256 of link and text, and length of the 120 replication articles (no article text).

In this public copy, proper nouns not needed for the research (other private projects, apps, reviewer nicknames)
are replaced by neutral words or PROJECT_A to PROJECT_F, and internal nicknames that point to places are removed or reworded
(marked 〔公開版…〕 where this happens). The owner's verbatim remarks in the registration documents are replaced by
"（オーナーの承認・発言。原文は非公開の記録にある）", keeping their date and time; the notification topic is a placeholder
(`YOUR_NTFY_TOPIC`), and Google Drive file IDs are removed. Because remarks were also removed from the docstrings of some
exploratory scripts and notebooks, their SHA-256 values differ from those recorded in the registration documents; the
unmodified files are kept in the private record. The BBC article texts and the generated continuations are not
redistributed; continuations are available on request.

The code was written by an AI assistant (Claude, Anthropic) under the author's direction; see Appendix E of the paper.

License: code in `scripts/` and `notebooks/` under the MIT License (`LICENSE`); documents, predictions, results, data
and the manuscript under CC BY 4.0 (`LICENSE-docs.md`).
