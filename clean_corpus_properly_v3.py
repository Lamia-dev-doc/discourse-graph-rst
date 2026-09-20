"""
clean_corpus_properly.py
========================
"""

import json
from collections import Counter

print("=" * 70)
print("  NETTOYAGE INTELLIGENT DU CORPUS GUM+SciDTB  (v3b — extraction corrigée)")
print("=" * 70)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # ⚠️ CHANGEMENT : pointe vers le corpus issu de l'extraction corrigée
    'INPUT_FILE':  'data/gum_scidtb_fused_v3.json',
    'OUTPUT_FILE': 'data/gum_scidtb_cleaned_v3.json',

    # Minimum d'exemples par relation pour être conservée
    # Justification empirique : classes < 100 ex. ont F1 instable
    'MIN_EXAMPLES': 100,

    # =========================================================================
    # MAPPING UNIFIÉ : reproduit la taxonomie à 22 classes de l'article
    # (Table 2 / Annexe B), appliqué au corpus issu de l'extraction corrigée.
    # =========================================================================
    'MERGE_RELATIONS': {

        # ── GUM (eRST natif) ─────────────────────────────────────────────────
        'elaboration-additional':    'elaboration-additional',
        'elaboration-attribute':     'elaboration-attribute',

        'adversative-antithesis':    'adversative-antithesis',
        'adversative-concession':    'adversative-concession',
        'adversative-contrast':      'comparison-contrast',     # multinuc → symétrique

        'attribution-positive':      'attribution-positive',
        'attribution-negative':      'attribution-negative',    # candidate, filtrée si <100

        'causal-cause':              'causal-cause',
        'causal-result':             'causal-result',

        'context-background':        'context-background',
        'context-circumstance':      'context-circumstance',

        'contingency-condition':     'contingency-condition',

        'evaluation-comment':        'evaluation-comment',

        'explanation-evidence':      'explanation-evidence',
        'explanation-justify':       'explanation-justify',
        'explanation-motivation':    'explanation-motivation',  # candidate, filtrée si <100

        'joint-list':                'joint',
        'joint-other':               'joint',
        'joint-disjunction':         'joint',
        'joint-sequence':            'temporal-sequence',

        'mode-manner':               'mode-manner',
        'mode-means':                'mode-means',

        'organization-heading':      'organization-preparation',
        'organization-phatic':       'organization-phatic',
        'organization-preparation':  'organization-preparation',

        'purpose-attribute':         'purpose-goal',
        'purpose-goal':              'purpose-goal',

        'restatement-partial':       'restatement-partial',
        'restatement-repetition':    'restatement-partial',     # multinuc GUM

        'topic-question':            'topic-question',          # candidate, filtrée si <100
        'topic-solutionhood':        'purpose-goal',

        # ── SciDTB (fine-grained officiel) ───────────────────────────────────
        'elab-addition':             'elaboration-additional',
        'elab-aspect':               'elaboration-attribute',
        'elab-definition':           'elaboration-attribute',
        'elab-example':              'elaboration-attribute',
        'elab-enum_member':          'elaboration-attribute',
        'elab-process_step':         'elaboration-attribute',

        'attribution':               'attribution-positive',

        'bg-general':                'context-background',
        'bg-goal':                   'context-background',
        'bg-related':                'context-background',
        'bg-compare':                'comparison-contrast',

        'cause':                     'causal-cause',
        'result':                    'causal-result',

        'comparison':                'comparison-contrast',
        'contrast':                  'comparison-contrast',     # cf. Section 3.3 de l'article

        'condition':                 'contingency-condition',

        'enablement':                'purpose-goal',

        'evaluation':                'evaluation-comment',

        'exp-evidence':              'explanation-evidence',
        'exp-reason':                'explanation-justify',

        'joint':                     'joint',

        'manner-means':              'mode-means',

        'progression':               'temporal-sequence',

        'summary':                   'restatement-partial',

        'temporal':                  'temporal-sequence',
    }
}

# ============================================================================
# CHARGEMENT
# ============================================================================

print("\n📚 Chargement du corpus...")
with open(CONFIG['INPUT_FILE'], 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"   ✓ {len(data):,} relations chargées")

original_relations = Counter(item['relation'] for item in data)
print(f"   ✓ {len(original_relations)} relations uniques")

# ============================================================================
# DIAGNOSTIC PRÉALABLE : relations non couvertes par le mapping
# ============================================================================

print("\n🔍 DIAGNOSTIC : vérification de la couverture du mapping...")

all_relations = set(original_relations.keys())
mapped_relations = set(CONFIG['MERGE_RELATIONS'].keys())
unmapped = all_relations - mapped_relations

if unmapped:
    total_lost = sum(original_relations[r] for r in unmapped)
    pct_lost = (total_lost / len(data) * 100)
    print(f"\n   ⚠️  RELATIONS NON MAPPÉES (seront perdues) :")
    for rel in sorted(unmapped):
        print(f"      ❌ {rel:40s}: {original_relations[rel]:5,} ex.")
    print(f"\n   ⚠️  Total perdu : {total_lost:,} / {len(data):,} ({pct_lost:.1f}%)")
    print(f"   → Ajoutez ces relations au MERGE_RELATIONS avant de continuer !")
else:
    print(f"   ✅ Toutes les relations sont couvertes par le mapping !")

# ============================================================================
# ÉTAPE 1 : FUSIONNER RELATIONS SIMILAIRES
# ============================================================================

print("\n🔀 ÉTAPE 1 : Fusion des relations similaires...")

merged_data = []
merge_stats = Counter()
skipped     = 0

for item in data:
    original_rel = item['relation']

    if original_rel in CONFIG['MERGE_RELATIONS']:
        new_rel = CONFIG['MERGE_RELATIONS'][original_rel]

        if original_rel != new_rel:
            merge_stats[f"{original_rel} → {new_rel}"] += 1

        merged_item = item.copy()
        merged_item['relation']          = new_rel
        merged_item['relation_original'] = original_rel
        merged_data.append(merged_item)
    else:
        skipped += 1

print(f"   ✓ {len(merged_data):,} relations conservées")
if skipped:
    print(f"   ⚠️  {skipped:,} relations ignorées (non mappées)")

if merge_stats:
    print(f"\n   📊 Top 10 fusions effectuées :")
    for fusion, count in merge_stats.most_common(10):
        print(f"      • {fusion}: {count:,}")

after_merge_counts = Counter(item['relation'] for item in merged_data)
print(f"\n   ✓ {len(after_merge_counts)} relations uniques"
      f" (avant fusion : {len(original_relations)})")

# ============================================================================
# ÉTAPE 2 : FILTRER RELATIONS RARES
# ============================================================================

print(f"\n🔍 ÉTAPE 2 : Filtrage (< {CONFIG['MIN_EXAMPLES']} exemples)...")

keep_relations    = {rel for rel, count in after_merge_counts.items()
                     if count >= CONFIG['MIN_EXAMPLES']}
removed_relations = set(after_merge_counts.keys()) - keep_relations

if removed_relations:
    print(f"   ⚠️  Relations supprimées (trop rares) :")
    for rel in sorted(removed_relations):
        print(f"      • {rel:40s}: {after_merge_counts[rel]:3d} exemples")
else:
    print(f"   ✅ Aucune relation supprimée")

filtered_data = [item for item in merged_data
                 if item['relation'] in keep_relations]

print(f"   ✓ {len(filtered_data):,} relations conservées")
print(f"   ✓ {len(keep_relations)} relations uniques finales")

if len(keep_relations) != 22:
    print(f"\n   ⚠️  ATTENTION : {len(keep_relations)} classes finales ≠ 22 "
          f"(taxonomie originale de l'article).")
    print(f"      Si des classes candidates (attribution-negative, "
          f"explanation-motivation, topic-question)")
    print(f"      dépassent maintenant 100 exemples grâce au corpus "
          f"agrandi, une décision manuelle est nécessaire :")
    print(f"      soit les conserver (taxonomie étendue), soit les "
          f"retirer explicitement pour rester à 22 classes.")

# ============================================================================
# STATISTIQUES FINALES
# ============================================================================

print("\n" + "=" * 70)
print("  📊 RÉSUMÉ DES TRANSFORMATIONS")
print("=" * 70)

final_counts = Counter(item['relation'] for item in filtered_data)

perte     = len(data) - len(filtered_data)
pct_perte = (perte / len(data) * 100) if data else 0

print(f"\n📈 AVANT → APRÈS :")
print(f"   Relations uniques : {len(original_relations):3d} → {len(final_counts):3d}")
print(f"   Exemples          : {len(data):,} → {len(filtered_data):,}")
print(f"   Perte             : {perte:,} ({pct_perte:.1f}%)")

print(f"\n🏆 DISTRIBUTION FINALE ({len(final_counts)} relations) :")
for i, (rel, count) in enumerate(final_counts.most_common(), 1):
    pct = (count / len(filtered_data) * 100)
    bar = '█' * int(pct / 2)
    print(f"  {i:2d}. {rel:40s}: {count:6,} ({pct:5.1f}%) {bar}")

# ============================================================================
# SAUVEGARDE
# ============================================================================

print(f"\n💾 Sauvegarde...")

with open(CONFIG['OUTPUT_FILE'], 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, indent=2, ensure_ascii=False)

print(f"   ✅ Corpus nettoyé   : {CONFIG['OUTPUT_FILE']}")

mapping_file = CONFIG['OUTPUT_FILE'].replace('.json', '_mapping.json')
mapping_info = {
    'merge_mapping':          CONFIG['MERGE_RELATIONS'],
    'original_relations':     len(original_relations),
    'final_relations':        len(final_counts),
    'original_examples':      len(data),
    'final_examples':         len(filtered_data),
    'relation_distribution':  dict(final_counts),
}

with open(mapping_file, 'w', encoding='utf-8') as f:
    json.dump(mapping_info, f, indent=2, ensure_ascii=False)

print(f"   ✅ Mapping sauvegardé : {mapping_file}")

# ============================================================================
# PROCHAINES ÉTAPES
# ============================================================================

print("\n" + "=" * 70)
print("  🚀 PROCHAINE ÉTAPE : Fine-tuning DeBERTa-v3 (split document-level)")
print("=" * 70)
print(f"""
   Dataset prêt : {CONFIG['OUTPUT_FILE']}
   Exemples     : {len(filtered_data):,}
   Classes      : {len(final_counts)}

   python fine_tune_deberta_rst_v2_docsplit.py
   (n'oubliez pas de mettre à jour son chemin de données vers
    {CONFIG['OUTPUT_FILE']})
""")
print("=" * 70)
