#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llm_ordering_stage1_generate.py
=================================

"""

import json
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CONFIG = {
    'LLM_MODEL_NAME': 'Qwen/Qwen2.5-7B-Instruct',
    'ARTICLES_JSON': 'articles_raw_clean.json',
    'N_PILOT': 100,
    'MAX_NEW_TOKENS': 200,
    'MAX_CHARS_PER_PARA': 600,
    'OUTPUT_FILE': 'llm_orderings_raw.json',
}

ORDERING_PROMPT_TEMPLATE = """You are an expert technical editor.
You will be given {n} paragraphs from an educational document, in a
SCRAMBLED order, each preceded by its current index in brackets.

Your task: determine the most coherent reading order for these
paragraphs, as they would appear in a well-organized textbook section.

{paragraphs_block}

Return ONLY a comma-separated list of the {n} indices in your proposed
reading order (e.g., "2,0,3,1"). No explanation, no other text.

Order:"""


def load_llm():
    model_name = CONFIG['LLM_MODEL_NAME']
    print(f"Chargement du LLM ({model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='auto',
    )
    model.eval()
    print("Modele charge.")
    return model, tokenizer


def build_ordering_prompt(paragraphs):
    n = len(paragraphs)
    blocks = []
    for i, p in enumerate(paragraphs):
        text = p[:CONFIG['MAX_CHARS_PER_PARA']]
        blocks.append(f"[{i}] {text}")
    paragraphs_block = "\n\n".join(blocks)
    return ORDERING_PROMPT_TEMPLATE.format(
        n=n, paragraphs_block=paragraphs_block)


def parse_ordering(raw_output, n):
    numbers = re.findall(r'\d+', raw_output)
    candidate = []
    seen = set()
    for tok in numbers:
        idx = int(tok)
        if 0 <= idx < n and idx not in seen:
            candidate.append(idx)
            seen.add(idx)
    if len(candidate) == n:
        return candidate, False
    missing = [i for i in range(n) if i not in seen]
    return candidate + missing, True


def llm_order_paragraphs(paragraphs, model, tokenizer):
    prompt = build_ordering_prompt(paragraphs)
    messages = [{"role": "user", "content": prompt}]

    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True,
        return_tensors="pt", return_dict=True
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=CONFIG['MAX_NEW_TOKENS'],
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    input_len = inputs['input_ids'].shape[1]
    raw = tokenizer.decode(
        output_ids[0][input_len:], skip_special_tokens=True)
    order, fallback = parse_ordering(raw, len(paragraphs))
    return order, fallback, raw


def main():
    print("=" * 74)
    print("  S6 - ETAPE 1/2 : GENERATION DES ORDRES PAR LE LLM (seul en memoire)")
    print("=" * 74)

    model, tokenizer = load_llm()

    with open(CONFIG['ARTICLES_JSON'], 'r', encoding='utf-8') as f:
        articles = json.load(f)
    n_pilot = CONFIG.get('N_PILOT')
    if n_pilot and n_pilot < len(articles):
        articles = articles[:n_pilot]
    print(f"\n{len(articles)} sections chargees")

    results = []
    n_fallback = 0
    start = time.time()

    for i, article in enumerate(articles):
        topic = article['topic']
        paras = article['paragraphs']
        if len(paras) < 3:
            continue

        order, fallback, raw = llm_order_paragraphs(paras, model, tokenizer)
        if fallback:
            n_fallback += 1

        results.append({
            'topic': topic,
            'paragraphs': paras,   # on garde le texte pour l'etape 2
            'order': order,
            'fallback': fallback,
        })

        # Sauvegarde incrementale : si la session coupe, on garde tout
        # ce qui a deja ete genere.
        with open(CONFIG['OUTPUT_FILE'], 'w', encoding='utf-8') as f:
            json.dump({'n_done': len(results), 'sections': results},
                       f, ensure_ascii=False, indent=2)

        elapsed = time.time() - start
        print(f"[{i+1}/{len(articles)}] {topic[:50]:<50} "
              f"fallback={fallback}  ({elapsed:.0f}s ecoulees)")

    print(f"\nTermine : {len(results)} sections, {n_fallback} avec repli partiel.")
    print(f"Fichier : {CONFIG['OUTPUT_FILE']}")
    print("\nProchaine etape : redemarre la session (liberer la VRAM du LLM),")
    print("puis lance llm_ordering_stage2_score.py sur ce fichier.")


if __name__ == "__main__":
    main()
