# RST-Based Discourse Graph Framework for Educational Document Organization

Code accompanying the manuscript *"Discourse Graph-Based Educational Document
Organization Using Transformer-Enhanced RST Relation Classification"*
(under major revision, SN Computer Science, manuscript SNCS-D-26-06127).

This repository provides the scripts used for corpus construction, classifier
training, discourse graph construction, beam-search document assembly, and
the additional validation experiments introduced during revision (hybrid
cascade significance testing, classification significance testing, learner-
profile comparison, and LLM-based ordering baselines).

## Repository structure

```
.
├── data_preparation/
│   ├── fusion_gum_scidtb_v3.py       # Corpus fusion (GUM + SciDTB → raw EDU pairs)
│   └── clean_corpus_properly_v3.py   # Relation consolidation (49 → 22 classes)
├── training/
│   └── fine_tune_deberta_rst_v2.py   # DeBERTa-v3-base classifier fine-tuning
├── inference/
│   └── run_pilot_beam_patched.py     # Hybrid prediction cascade, discourse graph
│                                      # construction, and beam-search traversal
├── validation/
│   ├── validate_hybrid_cascade.py                # McNemar validation of the cascade
│   ├── validate_classification_significance.py   # DeBERTa vs. BERT-base significance
│   ├── compare_learner_profiles.py               # Pairwise learner-profile comparison
│   ├── llm_ordering_stage1_generate.py            # LLM-based ordering baseline (S6), stage 1
│   └── llm_ordering_stage2_score.py               # LLM-based ordering baseline (S6), stage 2
├── data/
│   └── README.md                     # Links to the source corpora (not redistributed here)
├── requirements.txt
└── LICENSE
```

## Reproducing the pipeline

Each stage reads the output of the previous one; run them in order.

1. **Corpus fusion**
   ```bash
   python data_preparation/fusion_gum_scidtb_v3.py
   ```
   Produces the raw fused EDU-pair corpus from GUM and SciDTB (see
   `data/README.md` for how to obtain these source corpora).

2. **Relation consolidation**
   ```bash
   python data_preparation/clean_corpus_properly_v3.py
   ```
   Consolidates the 49 raw relation labels into the 22 classes reported in
   the manuscript (Table 2), filtering classes below the 100-instance
   threshold. The full 49→22 mapping and per-merge justification are given
   in Appendix B of the manuscript.

3. **Classifier fine-tuning**
   ```bash
   python training/fine_tune_deberta_rst_v2.py
   ```
   Fine-tunes `microsoft/deberta-v3-base` on the consolidated corpus.
   Key hyperparameters (Section 3.4): learning rate 2e-5, effective batch
   size 32 (batch 4 x grad-accum 8), up to 10 epochs with early stopping
   (patience 3), seed 42.

4. **Discourse graph construction and document assembly**
   ```bash
   python inference/run_pilot_beam_patched.py
   ```
   Applies the hybrid prediction cascade (Section 3.5), constructs the
   discourse graph $G=(V,E)$, and runs learner-profile-conditioned
   beam-search traversal (Section 3.7, Algorithm 2) to produce an
   organized document ordering.

5. **Validation experiments** (each script is self-contained and documents
   its own required inputs at the top of the file):
   - `validate_hybrid_cascade.py` — paired McNemar test comparing the
     three-strategy cascade, a full-context-only baseline, and a
     marker-free cascade on the held-out test set.
   - `validate_classification_significance.py` — paired McNemar test
     comparing DeBERTa-v3-base and BERT-base on the same test pairs.
   - `compare_learner_profiles.py` — pairwise comparison (exact-order
     agreement, Kendall's tau) of beam-search orderings under the
     novice / intermediate / advanced learner profiles.
   - `llm_ordering_stage1_generate.py` / `llm_ordering_stage2_score.py` —
     zero-shot LLM document-ordering baseline (S6, Appendix A).

## Requirements

```bash
pip install -r requirements.txt
```

Experiments were run with Python 3.x on a CUDA-enabled GPU. See
`requirements.txt` for the exact package versions used.

## Data availability

This repository contains code only. The source corpora are third-party
resources and are not redistributed here; see `data/README.md` for
links and access instructions for GUM, SciDTB, and the OpenStax textbooks
used for evaluation.

## Citation

If you use this code, please cite the manuscript (citation details will be
added upon publication; in the meantime, please cite the manuscript number
SNCS-D-26-06127, SN Computer Science).

## License

This project is licensed under the terms of the license in `LICENSE`.

## Contact

Lamia Hamouche — CERIST (DTISI Laboratory, Algiers) / Université A. Mira de
Béjaïa (Laboratoire LIMED). For questions about this code, please open an
issue on this repository.
