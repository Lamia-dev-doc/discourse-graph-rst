#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_classification_significance.py
==========================================

"""

import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score

try:
    from statsmodels.stats.contingency_tables import mcnemar
    HAVE_MCNEMAR = True
except ImportError:
    import os
    os.system("pip install statsmodels --break-system-packages -q")
    from statsmodels.stats.contingency_tables import mcnemar
    HAVE_MCNEMAR = True

import fine_tune_deberta_rst_v2 as ft

CONFIG = {
    'BERT_MODEL_PATH':    'bert_rst_model',        # <-- adapte au vrai chemin
    'DEBERTA_MODEL_PATH': 'deberta_rst_model_v2',   # <-- adapte au vrai chemin
    'MAX_LENGTH': 128,
}


def load_classifier(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    import pickle
    from pathlib import Path
    with open(Path(model_path) / 'label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    return model, tokenizer, label_encoder, device


def predict_batch(model, tokenizer, label_encoder, device, text1, text2, max_length):
    encoding = tokenizer(
        text1, text2, add_special_tokens=True, max_length=max_length,
        padding='max_length', truncation=True, return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        pred_idx = torch.argmax(outputs.logits, dim=1).item()
    return label_encoder.inverse_transform([pred_idx])[0]


def main():
    print("=" * 74)
    print("  VALIDATION STATISTIQUE — BERT-base vs DeBERTa-v3-base (Reviewer #4, C.2)")
    print("=" * 74)

    print("\nReproduction du split de test officiel...")
    pairs, labels = ft.load_corpus(ft.CONFIG['DATA_FILE'])
    from sklearn.model_selection import train_test_split
    train_pairs, temp_pairs, train_labels, temp_labels = train_test_split(
        pairs, labels, test_size=ft.CONFIG['VAL_SIZE'] + ft.CONFIG['TEST_SIZE'],
        random_state=ft.CONFIG['SEED'], stratify=labels)
    val_pairs, test_pairs, val_labels, test_labels = train_test_split(
        temp_pairs, temp_labels,
        test_size=ft.CONFIG['TEST_SIZE'] / (ft.CONFIG['VAL_SIZE'] + ft.CONFIG['TEST_SIZE']),
        random_state=ft.CONFIG['SEED'], stratify=temp_labels)
    print(f"   Split test reproduit : {len(test_pairs)} paires")

    print("\nChargement DeBERTa-v3-base...")
    deberta_model, deberta_tok, deberta_le, deberta_dev = load_classifier(
        CONFIG['DEBERTA_MODEL_PATH'])

    print("Chargement BERT-base...")
    bert_model, bert_tok, bert_le, bert_dev = load_classifier(
        CONFIG['BERT_MODEL_PATH'])

    print(f"\nPrédiction sur {len(test_pairs)} paires (les deux modèles)...")
    deberta_preds, bert_preds = [], []
    for i, (text1, text2) in enumerate(test_pairs):
        if i % 200 == 0:
            print(f"   ... {i}/{len(test_pairs)}")
        deberta_preds.append(predict_batch(
            deberta_model, deberta_tok, deberta_le, deberta_dev,
            text1, text2, CONFIG['MAX_LENGTH']))
        bert_preds.append(predict_batch(
            bert_model, bert_tok, bert_le, bert_dev,
            text1, text2, CONFIG['MAX_LENGTH']))

    gold = list(test_labels)

    acc_d = accuracy_score(gold, deberta_preds)
    acc_b = accuracy_score(gold, bert_preds)
    f1m_d = f1_score(gold, deberta_preds, average='macro', zero_division=0)
    f1m_b = f1_score(gold, bert_preds, average='macro', zero_division=0)
    f1w_d = f1_score(gold, deberta_preds, average='weighted', zero_division=0)
    f1w_b = f1_score(gold, bert_preds, average='weighted', zero_division=0)

    print("\n" + "=" * 74)
    print("  RÉSULTATS — mêmes paires de test, comparaison appariée")
    print("=" * 74)
    print(f"\n{'Métrique':14s} {'DeBERTa-v3':>12s} {'BERT-base':>12s} {'Δ':>10s}")
    print("-" * 52)
    print(f"{'Accuracy':14s} {acc_d:12.4f} {acc_b:12.4f} {acc_d-acc_b:+10.4f}")
    print(f"{'F1-macro':14s} {f1m_d:12.4f} {f1m_b:12.4f} {f1m_d-f1m_b:+10.4f}")
    print(f"{'F1-weighted':14s} {f1w_d:12.4f} {f1w_b:12.4f} {f1w_d-f1w_b:+10.4f}")
    print("\n  (Compare a Table 5 : 71.34%/0.6617/0.7150 vs 70.12%/0.6542/0.7025")
    print("   -- si ces chiffres different sensiblement, verifie CONFIG ci-dessus)")

    EXPECTED_DEBERTA_ACC = 0.7134
    EXPECTED_BERT_ACC = 0.7012
    if abs(acc_d - EXPECTED_DEBERTA_ACC) > 0.002:
        print(f"\n  ATTENTION : l'accuracy DeBERTa ({acc_d:.4f}) ne reproduit pas le "
              f"Tableau 5 ({EXPECTED_DEBERTA_ACC:.4f}) -- verifie CONFIG['DEBERTA_MODEL_PATH'].")
    if abs(acc_b - EXPECTED_BERT_ACC) > 0.002:
        print(f"\n  ATTENTION : l'accuracy BERT-base ({acc_b:.4f}) ne reproduit pas le "
              f"Tableau 5 ({EXPECTED_BERT_ACC:.4f}) -- verifie CONFIG['BERT_MODEL_PATH'].")

    gold_arr = np.array(gold)
    deberta_arr = np.array(deberta_preds)
    bert_arr = np.array(bert_preds)

    if HAVE_MCNEMAR:
        correct_d = (deberta_arr == gold_arr)
        correct_b = (bert_arr == gold_arr)
        both_right = np.sum(correct_d & correct_b)
        d_only_right = np.sum(correct_d & ~correct_b)
        b_only_right = np.sum(~correct_d & correct_b)
        both_wrong = np.sum(~correct_d & ~correct_b)
        table = [[both_right, d_only_right], [b_only_right, both_wrong]]
        result = mcnemar(table, exact=False, correction=True)
        print("\n" + "-" * 74)
        print("  TEST DE MCNEMAR (DeBERTa vs BERT-base, paires appariées)")
        print("  Ce test porte sur l'EXACTITUDE (correct/incorrect par paire),")
        print("  pas directement sur le F1-macro/F1-weighted.")
        print("-" * 74)
        print(f"  Les deux corrects   : {both_right} paires")
        print(f"  DeBERTa correct seul : {d_only_right} paires")
        print(f"  BERT-base correct seul : {b_only_right} paires")
        print(f"  Les deux incorrects : {both_wrong} paires")
        print(f"  statistic={result.statistic:.3f}  p-value={result.pvalue:.6f}")
        print(f"  -> {'Difference statistiquement significative' if result.pvalue < .05 else 'PAS de difference statistiquement significative'} (alpha=.05)")

    with open('classification_significance_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'n_test_pairs': len(gold),
            'accuracy_deberta': acc_d, 'accuracy_bert': acc_b,
            'f1_macro_deberta': f1m_d, 'f1_macro_bert': f1m_b,
            'f1_weighted_deberta': f1w_d, 'f1_weighted_bert': f1w_b,
            'mcnemar_statistic': float(result.statistic) if HAVE_MCNEMAR else None,
            'mcnemar_pvalue': float(result.pvalue) if HAVE_MCNEMAR else None,
        }, f, ensure_ascii=False, indent=2)
    print("\n  Résultats sauvegardés : classification_significance_results.json")


if __name__ == "__main__":
    main()
