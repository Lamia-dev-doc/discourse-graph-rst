#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_hybrid_cascade.py

"""

import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

try:
    from statsmodels.stats.contingency_tables import mcnemar
    HAVE_MCNEMAR = True
except ImportError:
    HAVE_MCNEMAR = False
    print("⚠️  statsmodels indisponible (pip install statsmodels) — "
          "le test de McNemar sera omis.")

# ── Réutilise le split officiel (même corpus, même seed) ──────────────────
import fine_tune_deberta_rst_v2 as ft
# ── Réutilise la cascade réellement utilisée (mêmes seuils, même code) ────
import run_pilot_beam_patched as rpb


def reproduce_test_split():
    """Reproduit exactement le split test de fine_tune_deberta_rst_v2.py
    (mêmes 2045 paires que Table 3 / Table 5)."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    pairs, labels = ft.load_corpus(ft.CONFIG['DATA_FILE'])
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    X_train, X_temp, y_train, y_temp = train_test_split(
        pairs, encoded_labels,
        test_size=ft.CONFIG['VAL_SIZE'] + ft.CONFIG['TEST_SIZE'],
        random_state=ft.CONFIG['SEED'],
        stratify=encoded_labels)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=ft.CONFIG['TEST_SIZE'] / (ft.CONFIG['VAL_SIZE'] + ft.CONFIG['TEST_SIZE']),
        random_state=ft.CONFIG['SEED'],
        stratify=y_temp)

    gold_labels = label_encoder.inverse_transform(y_test)
    print(f"   ✓ Split test reproduit : {len(X_test):,} paires "
          f"(identique à Table 3 / Table 5 si le corpus et le seed n'ont "
          f"pas changé)")
    return X_test, gold_labels, label_encoder


def main():
    print("=" * 74)
    print("  VALIDATION QUANTITATIVE DE LA CASCADE HYBRIDE")
    print("=" * 74)

    print("\n📚 Reproduction du split de test officiel...")
    X_test, gold_labels, label_encoder = reproduce_test_split()

    print("\n🤖 Chargement du modèle...")
    model, tokenizer, model_label_encoder, device = rpb.load_model(
        rpb.CONFIG['MODEL_PATH'])

    hybrid_preds, fullctx_preds, strategies = [], [], []

    print(f"\n🔬 Prédiction sur {len(X_test):,} paires "
          f"(cascade hybride + full-context seul)...")
    for i, (edu1, edu2) in enumerate(X_test):
        if i % 200 == 0:
            print(f"   ... {i}/{len(X_test)}")

        # (a) Cascade hybride complète — pipeline réellement utilisé
        rel_h, strategy, conf_h, _ = rpb.predict_pair_hybrid(
            edu1, edu2, model, tokenizer, model_label_encoder, device)
        hybrid_preds.append(rel_h)
        strategies.append(strategy)

        # (b) Full-context SEUL (Stratégie 3 pour tout le monde)
        rel_f, conf_f = rpb.predict_deberta(
            model, tokenizer, model_label_encoder, device, edu1, edu2)
        fullctx_preds.append(rel_f)

    gold = list(gold_labels)

    # ── Métriques globales ──────────────────────────────────────────────────
    acc_h = accuracy_score(gold, hybrid_preds)
    acc_f = accuracy_score(gold, fullctx_preds)
    f1m_h = f1_score(gold, hybrid_preds, average='macro', zero_division=0)
    f1m_f = f1_score(gold, fullctx_preds, average='macro', zero_division=0)
    f1w_h = f1_score(gold, hybrid_preds, average='weighted', zero_division=0)
    f1w_f = f1_score(gold, fullctx_preds, average='weighted', zero_division=0)

    print("\n" + "=" * 74)
    print("  RÉSULTATS — mêmes 2045 paires de test, comparaison appariée")
    print("=" * 74)
    print(f"\n{'Métrique':20s} {'Cascade hybride':>18s} {'Full-context seul':>20s} {'Δ':>10s}")
    print("-" * 70)
    print(f"{'Accuracy':20s} {acc_h:18.4f} {acc_f:20.4f} {acc_h-acc_f:+10.4f}")
    print(f"{'F1-macro':20s} {f1m_h:18.4f} {f1m_f:20.4f} {f1m_h-f1m_f:+10.4f}")
    print(f"{'F1-weighted':20s} {f1w_h:18.4f} {f1w_f:20.4f} {f1w_h-f1w_f:+10.4f}")

    # ── Ventilation par stratégie ─────────────────────────────────────────
    print("\n" + "-" * 74)
    print("  EXACTITUDE PAR STRATÉGIE (la cascade hybride vs. ce que le")
    print("  full-context SEUL aurait donné sur les MÊMES paires)")
    print("-" * 74)
    strategies_arr = np.array(strategies)
    gold_arr = np.array(gold)
    hybrid_arr = np.array(hybrid_preds)
    fullctx_arr = np.array(fullctx_preds)

    for strat in sorted(set(strategies)):
        mask = strategies_arr == strat
        n = mask.sum()
        if n == 0:
            continue
        acc_h_s = accuracy_score(gold_arr[mask], hybrid_arr[mask])
        acc_f_s = accuracy_score(gold_arr[mask], fullctx_arr[mask])
        print(f"  Stratégie '{strat}' (n={n}, {n/len(gold)*100:.1f}% des paires) : "
              f"hybride={acc_h_s:.4f}  full-context-seul={acc_f_s:.4f}  "
              f"Δ={acc_h_s-acc_f_s:+.4f}")

    # ── Test de McNemar (comparaison appariée, correcte/incorrecte) ───────
    if HAVE_MCNEMAR:
        correct_h = (hybrid_arr == gold_arr)
        correct_f = (fullctx_arr == gold_arr)
        # Table de contingence 2x2 : [both_wrong, f_only_right / h_only_right, both_right]
        both_right   = np.sum(correct_h & correct_f)
        h_only_right = np.sum(correct_h & ~correct_f)
        f_only_right = np.sum(~correct_h & correct_f)
        both_wrong   = np.sum(~correct_h & ~correct_f)
        table = [[both_right, h_only_right], [f_only_right, both_wrong]]
        result = mcnemar(table, exact=False, correction=True)
        print("\n" + "-" * 74)
        print("  TEST DE MCNEMAR (hybride vs full-context-seul, paires appariées)")
        print("-" * 74)
        print(f"  Hybride correct seul : {h_only_right} paires")
        print(f"  Full-context correct seul : {f_only_right} paires")
        print(f"  statistic={result.statistic:.3f}  p-value={result.pvalue:.4f}")
        print(f"  → {'Différence statistiquement significative' if result.pvalue < .05 else 'PAS de différence statistiquement significative'} (α=.05)")

    # ── Sauvegarde ──────────────────────────────────────────────────────
    with open('hybrid_cascade_validation.json', 'w', encoding='utf-8') as f:
        json.dump({
            'n_test_pairs': len(gold),
            'accuracy_hybrid': acc_h, 'accuracy_fullcontext': acc_f,
            'f1_macro_hybrid': f1m_h, 'f1_macro_fullcontext': f1m_f,
            'f1_weighted_hybrid': f1w_h, 'f1_weighted_fullcontext': f1w_f,
            'strategy_distribution': {s: int((strategies_arr == s).sum())
                                       for s in set(strategies)},
        }, f, ensure_ascii=False, indent=2)
    print("  💾 Résultats sauvegardés : hybrid_cascade_validation.json")


if __name__ == "__main__":
    main()
