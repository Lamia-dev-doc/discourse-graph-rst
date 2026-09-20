#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llm_ordering_stage2_score.py

"""

import json

import run_pilot_beam_patched as rpb

CONFIG = {
    'INPUT_FILE': 'llm_orderings_raw.json',
    'OUTPUT_FILE': 'llm_ordering_results.json',
}


def main():
    print("=" * 74)
    print("  S6 - ETAPE 2/2 : NOTATION DES ORDRES (DeBERTa + SBERT + GPT-2)")
    print("=" * 74)

    with open(CONFIG['INPUT_FILE'], 'r', encoding='utf-8') as f:
        data = json.load(f)
    sections = data['sections']
    print(f"\n{len(sections)} sections a noter")

    print("\nChargement DeBERTa + SBERT (le LLM n'est pas charge ici)...")
    deberta_model, deberta_tokenizer, label_encoder, device = rpb.load_model(
        rpb.CONFIG['MODEL_PATH'])
    sbert_model = rpb.load_sbert_model(rpb.CONFIG['SBERT_MODEL_NAME'])

    results = []
    n_fallback = 0

    for i, sec in enumerate(sections):
        topic = sec['topic']
        paras = sec['paragraphs']
        order = sec['order']
        fallback = sec['fallback']
        if fallback:
            n_fallback += 1

        ordered_paragraphs = [paras[idx] for idx in order]

        rst_score, _ = rpb.compute_coherence_score(
            ordered_paragraphs, deberta_model, deberta_tokenizer,
            label_encoder, device)
        sbert_score = rpb.sbert_coherence_score(ordered_paragraphs, sbert_model)
        gpt2_ppl = rpb.gpt2_perplexity(ordered_paragraphs)

        print(f"[{i+1}/{len(sections)}] {topic[:40]:<40} "
              f"RST={rst_score:.3f} SBERT={sbert_score:.3f} GPT2={gpt2_ppl:.1f}")

        results.append({
            'topic': topic,
            'order': order,
            'fallback': fallback,
            'scores': {
                'rst_coherence': rst_score,
                'sbert_coherence': sbert_score,
                'gpt2_perplexity': gpt2_ppl,
            },
        })

    n = len(results)
    mean_rst = sum(r['scores']['rst_coherence'] for r in results) / n
    mean_sbert = sum(r['scores']['sbert_coherence'] for r in results) / n
    mean_gpt2 = sum(r['scores']['gpt2_perplexity'] for r in results) / n

    print("\n" + "=" * 74)
    print(f"  RESUME S6 (LLM-guided) - {n} sections")
    print("=" * 74)
    print(f"  RST coherence (up)     : {mean_rst:.3f}")
    print(f"  SBERT coherence (up)   : {mean_sbert:.3f}")
    print(f"  GPT-2 perplexity (down): {mean_gpt2:.1f}")
    print(f"  Parsing incomplet      : {n_fallback}/{n} sections "
          f"({n_fallback/n*100:.1f}%)")

    with open(CONFIG['OUTPUT_FILE'], 'w', encoding='utf-8') as f:
        json.dump({
            'n_sections': n,
            'mean_rst_coherence': mean_rst,
            'mean_sbert_coherence': mean_sbert,
            'mean_gpt2_perplexity': mean_gpt2,
            'fallback_rate': n_fallback / n if n else 0,
            'per_section': results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nResultats sauvegardes : {CONFIG['OUTPUT_FILE']}")


if __name__ == "__main__":
    main()
