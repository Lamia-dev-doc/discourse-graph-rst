#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_learner_profiles.py
============================

"""

import json
import itertools
from pathlib import Path

import numpy as np

try:
    from scipy.stats import kendalltau
except ImportError:
    kendalltau = None
    print("⚠️  scipy indisponible (pip install scipy) — le tau de Kendall sera omis.")

# ── Réutilise le pipeline existant : mêmes modèles, même graphe, mêmes profils ──
import run_pilot_beam_patched as rpb

CONFIG = rpb.CONFIG
PROFILES = ['novice', 'intermediate', 'advanced']

# 22 classes finales de l'article (Table 2)
RELATIONS_22 = [
    "elaboration-additional", "elaboration-attribute", "attribution-positive",
    "purpose-goal", "evaluation-comment", "joint", "context-background",
    "mode-means", "comparison-contrast", "context-circumstance",
    "contingency-condition", "organization-phatic", "temporal-sequence",
    "causal-cause", "adversative-concession", "causal-result",
    "explanation-evidence", "organization-preparation", "restatement-partial",
    "mode-manner", "explanation-justify", "adversative-antithesis",
]


def print_preference_matrix():
    """Affiche et retourne la matrice complète W_l(r), r=1..22, l=len(PROFILES) profils.
    Entièrement dynamique sur PROFILES pour ne plus jamais se désynchroniser
    si la liste des profils change (bug corrigé : les colonnes étaient
    codées en dur avec l'ancien trio novice/advanced/review)."""
    print("\n" + "=" * 78)
    print(f"  MATRICE DE PRÉFÉRENCE COMPLÈTE  W_l(r)  —  r = 1..22, l ∈ {{{', '.join(PROFILES)}}}")
    print("  (valeur basse = relation priorisée en premier ; défaut = 5 si absente)")
    print("=" * 78)
    col_width = max(10, max(len(p) for p in PROFILES) + 2)
    header = f"{'Relation':28s} " + " ".join(f"{p:>{col_width}s}" for p in PROFILES)
    print(header)
    print("-" * len(header))
    matrix = {}
    for r in RELATIONS_22:
        row = {}
        for profile in PROFILES:
            w = rpb.PROFILE_WEIGHTS.get(profile, {})
            row[profile] = w.get(r, 5)
        matrix[r] = row
        line = f"{r:28s} " + " ".join(f"{row[p]:{col_width}d}" for p in PROFILES)
        print(line)

    identical = [r for r in RELATIONS_22
                 if len({matrix[r][p] for p in PROFILES}) == 1]
    print(f"\n⚠️  {len(identical)}/22 relations ont le même poids (défaut=5) dans les 3 "
          f"profils, donc ne sont pas différenciées par le profil apprenant :")
    for r in identical:
        print(f"   - {r}")
    return matrix


def kendall_tau_orderings(order_a, order_b):
    """Tau de Kendall entre deux permutations (listes d'indices de paragraphes)."""
    if kendalltau is None or len(order_a) < 2:
        return None
    rank_a = {p: i for i, p in enumerate(order_a)}
    rank_b = {p: i for i, p in enumerate(order_b)}
    common = sorted(rank_a.keys())
    a = [rank_a[p] for p in common]
    b = [rank_b[p] for p in common]
    tau, _ = kendalltau(a, b)
    return tau


def pct_positions_changed(order_a, order_b):
    """% de positions où le paragraphe diffère entre les deux ordres."""
    n = len(order_a)
    if n == 0:
        return 0.0
    diff = sum(1 for a, b in zip(order_a, order_b) if a != b)
    return 100.0 * diff / n


def relation_usage(edges):
    """Histogramme des relations utilisées sur le chemin gagnant."""
    counts = {}
    for e in edges:
        counts[e['relation']] = counts.get(e['relation'], 0) + 1
    return counts


def main():
    matrix = print_preference_matrix()

    model, tokenizer, label_encoder, device = rpb.load_model(CONFIG['MODEL_PATH'])
    sbert_model = rpb.load_sbert_model(CONFIG['SBERT_MODEL_NAME'])

    articles_json = CONFIG.get('ARTICLES_JSON')
    n_pilot = CONFIG.get('N_PILOT')
    with open(articles_json, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    if n_pilot and n_pilot < len(articles):
        articles = articles[:n_pilot]
    print(f"\n📂 {len(articles)} sections chargées depuis {articles_json}")

    per_section_results = []
    pair_stats = {pair: {'identical': 0, 'taus': [], 'pct_changed': []}
                  for pair in itertools.combinations(PROFILES, 2)}
    relation_usage_by_profile = {p: {} for p in PROFILES}

    for article in articles:
        topic = article['topic']
        paras = article['paragraphs']
        if len(paras) < 3:
            continue
        print(f"\n── {topic} ({len(paras)} paragraphes) ──")

        # Graphe construit UNE SEULE FOIS (coût DeBERTa payé une fois)
        graph, adjacency, sbert_matrix = rpb.build_rst_discourse_graph(
            paras, model, tokenizer, label_encoder, device,
            sbert_model=sbert_model)

        orderings = {}
        edges_by_profile = {}
        for profile in PROFILES:
            _, edges, ordering = rpb.traverse_beam_search(
                paras, graph, profile=profile, beam_width=5, alpha=1.0)
            orderings[profile] = ordering
            edges_by_profile[profile] = edges
            for rel, cnt in relation_usage(edges).items():
                relation_usage_by_profile[profile][rel] = \
                    relation_usage_by_profile[profile].get(rel, 0) + cnt
            print(f"   {profile:10s} → {ordering}")

        section_result = {'topic': topic, 'n_paragraphs': len(paras),
                           'orderings': orderings, 'pairwise': {}}

        for p1, p2 in itertools.combinations(PROFILES, 2):
            identical = orderings[p1] == orderings[p2]
            tau = kendall_tau_orderings(orderings[p1], orderings[p2])
            pct = pct_positions_changed(orderings[p1], orderings[p2])
            pair_stats[(p1, p2)]['identical'] += int(identical)
            if tau is not None:
                pair_stats[(p1, p2)]['taus'].append(tau)
            pair_stats[(p1, p2)]['pct_changed'].append(pct)
            section_result['pairwise'][f"{p1}_vs_{p2}"] = {
                'identical': identical, 'kendall_tau': tau,
                'pct_positions_changed': pct}
            print(f"      {p1} vs {p2}: identique={identical} "
                  f"tau={tau if tau is not None else 'n/a'} "
                  f"%chgé={pct:.1f}%")

        per_section_results.append(section_result)

    # ── Résumé agrégé ──────────────────────────────────────────────────────────
    n = len(per_section_results)
    print("\n" + "=" * 78)
    print(f"  RÉSUMÉ — {n} sections, comparaison des 3 profils apprenants")
    print("=" * 78)
    for (p1, p2), stats in pair_stats.items():
        pct_identical = 100.0 * stats['identical'] / n if n else 0.0
        mean_tau = np.mean(stats['taus']) if stats['taus'] else None
        mean_pct_changed = np.mean(stats['pct_changed']) if stats['pct_changed'] else None
        print(f"\n  {p1} vs {p2} :")
        print(f"    Ordres identiques : {stats['identical']}/{n} ({pct_identical:.1f}%)")
        if mean_tau is not None:
            print(f"    Tau de Kendall moyen : {mean_tau:.3f} "
                  f"(1.0 = ordres identiques, 0.0 = non corrélés)")
        print(f"    Positions changées en moyenne : {mean_pct_changed:.1f}%")

    print("\n  Utilisation des relations sur le chemin gagnant, par profil :")
    for profile in PROFILES:
        top = sorted(relation_usage_by_profile[profile].items(),
                     key=lambda x: -x[1])[:5]
        print(f"    {profile:10s} → top 5 relations : {top}")

    print("""
  ─────────────────────────────────────────────────────────────────────
  Lecture pour la réponse à R3 :
  - Si %identique est bas et tau << 1.0 : le profil change bien
    substantiellement l'ordre → revendication de personnalisation soutenue.
  - Si %identique est élevé : les 9/22 relations à poids par défaut (5)
    dans les 3 profils limitent la différenciation → à nuancer dans
    la réponse plutôt qu'à cacher.
  - La distribution "top 5 relations" par profil doit être cohérente
    avec les poids de PROFILE_WEIGHTS (ex. novice → context-background
    et elaboration en tête ; advanced → explanation-evidence/justify).
""")

    # ── Sauvegarde ──────────────────────────────────────────────────────────
    out_dir = Path(CONFIG['OUTPUT_DIR'])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'learner_profile_comparison.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'preference_matrix': matrix,
            'per_section': per_section_results,
            'relation_usage_by_profile': relation_usage_by_profile,
        }, f, ensure_ascii=False, indent=2)
    print(f"  💾 Détails complets sauvegardés : {out_path}")


if __name__ == "__main__":
    main()
