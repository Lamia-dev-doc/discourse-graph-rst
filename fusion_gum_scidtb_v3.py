"""
fusion_gum_scidtb.py

"""

import os
import json
from pathlib import Path
from collections import Counter, defaultdict
import xml.etree.ElementTree as ET
from tqdm import tqdm


# ============================================================================
# RELATIONS À IGNORER (structurelles, pas sémantiques)
# ============================================================================

RELATIONS_IGNORE = {
    'span',       # GUM : nœud intermédiaire RST
    'same-unit',  # GUM + SciDTB : pseudo-relation pour EDU discontinues
    'null',       # SciDTB : nœud ROOT
    'ROOT',       # SciDTB : relation du nœud racine
}


class CorpusFusion:
    """Fusion GUM (eRST .rs4) + SciDTB (.dep JSON)"""

    def __init__(self):
        self.dataset = []
        self.stats = {
            'gum_docs': 0,
            'gum_relations': 0,
            'gum_ignored': 0,
            'scidtb_docs': 0,
            'scidtb_relations': 0,
            'scidtb_ignored': 0,
            'relation_distribution': Counter()
        }

    # -------------------------------------------------------------------------
    # PARSING GUM
    # -------------------------------------------------------------------------

    def parse_gum(self, gum_dir='rst/rstweb'):
        """
        Parse le corpus GUM (fichiers .rs4 eRST).
        Cherche dans plusieurs emplacements possibles.
        """
        print("\n" + "=" * 60)
        print("PARSING GUM CORPUS (.rs4)")
        print("=" * 60)

        possible_dirs = [
            gum_dir,
            'rstweb',
            'rst/rstweb',
            'data/GUM/rst/rstweb',
            '../rstweb',
            '../rst/rstweb',
        ]

        # ✅ FIX 1 : initialiser avant la boucle pour éviter NameError
        rs4_files = []

        for dir_path in possible_dirs:
            p = Path(dir_path)
            if p.exists():
                found = list(p.glob('**/*.rs4'))
                if found:
                    rs4_files = found
                    print(f"✓ Trouvé dans : {dir_path}")
                    break

        print(f"📁 Fichiers .rs4 trouvés : {len(rs4_files)}")

        if not rs4_files:
            print("⚠️  AUCUN FICHIER GUM TROUVÉ !")
            print("   Chemins cherchés :", possible_dirs)
            print("   Spécifiez le bon chemin en argument : parse_gum('votre/chemin')")
            return

        # Compteur diagnostic : paires qui auraient été perdues sous l'ancienne
        # logique (parent est un <group>, jamais présent dans un dict "segments only")
        self.stats['gum_multinuc_pairs_recovered'] = 0
        self.stats['gum_mononuc_pairs_via_group']  = 0

        for rs4_file in tqdm(rs4_files, desc="Parsing GUM"):
            try:
                tree = ET.parse(rs4_file)
                root = tree.getroot()

                # -----------------------------------------------------------
                # 1) Table unifiée des nœuds : <segment> (texte) + <group>
                #    (nœuds internes non lexicalisés : type="span" ou
                #    type="multinuc")
                # -----------------------------------------------------------
                nodes = {}

                for seg in root.findall('.//segment'):
                    seg_id = seg.attrib.get('id')
                    text   = seg.text.strip() if seg.text else ''
                    if seg_id and text:
                        nodes[seg_id] = {
                            'kind':       'segment',
                            'text':       text,
                            'parent':     seg.attrib.get('parent'),
                            'relation':   seg.attrib.get('relname', ''),
                            'group_type': None,
                        }

                for grp in root.findall('.//group'):
                    grp_id = grp.attrib.get('id')
                    if grp_id:
                        nodes[grp_id] = {
                            'kind':       'group',
                            'text':       None,
                            'parent':     grp.attrib.get('parent'),
                            'relation':   grp.attrib.get('relname', ''),
                            'group_type': grp.attrib.get('type', ''),  # 'span' | 'multinuc'
                        }

                # -----------------------------------------------------------
                # 2) Index enfants-par-parent, trié en ordre document (id
                #    numérique croissant = ordre de lecture dans .rs4)
                # -----------------------------------------------------------
                children_by_parent = defaultdict(list)
                for node_id, node_data in nodes.items():
                    if node_data['parent']:
                        children_by_parent[node_data['parent']].append(node_id)

                def _sort_key(nid):
                    return int(nid) if str(nid).isdigit() else str(nid)

                for pid in children_by_parent:
                    children_by_parent[pid].sort(key=_sort_key)

                # -----------------------------------------------------------
                # 3) Résolution récursive : tout nœud (segment ou groupe) →
                #    texte de son UDE tête.
                #    - segment  : lui-même
                #    - groupe   : l'enfant noyau (relname == "span"), sinon
                #                 le premier enfant en ordre document
                # -----------------------------------------------------------
                memo = {}

                def resolve_head(node_id, _depth=0):
                    if node_id in memo:
                        return memo[node_id]
                    if node_id not in nodes or _depth > 50:  # garde-fou anti-cycle
                        return None
                    node_data = nodes[node_id]
                    if node_data['kind'] == 'segment':
                        memo[node_id] = node_data['text']
                        return node_data['text']

                    kids = children_by_parent.get(node_id, [])
                    if not kids:
                        memo[node_id] = None
                        return None
                    nucleus_kids = [k for k in kids if nodes[k]['relation'] == 'span']
                    head_child = nucleus_kids[0] if nucleus_kids else kids[0]
                    result = resolve_head(head_child, _depth + 1)
                    memo[node_id] = result
                    return result

                # Groupes multinucléaires (co-noyaux)
                multinuc_group_ids = {
                    gid for gid, gd in nodes.items()
                    if gd['kind'] == 'group' and gd.get('group_type') == 'multinuc'
                }

                processed_multinuc_pairs = set()

                # -----------------------------------------------------------
                # 4) Construction des paires
                # -----------------------------------------------------------
                for node_id, node_data in nodes.items():
                    relation  = node_data['relation']
                    parent_id = node_data['parent']

                    if relation in RELATIONS_IGNORE:
                        self.stats['gum_ignored'] += 1
                        continue

                    if not parent_id or parent_id not in nodes:
                        continue

                    # --- Cas A : nœud sœur d'un groupe multinucléaire -------
                    # → apparier avec la sœur ADJACENTE suivante (ordre
                    #   document) plutôt qu'avec le groupe partagé non
                    #   lexicalisé. Une seule paire par couple adjacent.
                    if parent_id in multinuc_group_ids:
                        siblings = children_by_parent.get(parent_id, [])
                        if node_id in siblings:
                            idx = siblings.index(node_id)
                            if idx < len(siblings) - 1:
                                next_id  = siblings[idx + 1]
                                pair_key = (node_id, next_id)
                                if pair_key not in processed_multinuc_pairs:
                                    processed_multinuc_pairs.add(pair_key)
                                    head1 = resolve_head(node_id)
                                    head2 = resolve_head(next_id)
                                    if head1 and head2:
                                        self.dataset.append({
                                            'edu1':     head1,
                                            'edu2':     head2,
                                            'relation': relation,
                                            'source':   'GUM',
                                            'doc_id':   rs4_file.stem,
                                        })
                                        self.stats['gum_relations'] += 1
                                        self.stats['gum_multinuc_pairs_recovered'] += 1
                                        self.stats['relation_distribution'][relation] += 1
                        continue  # ne pas créer en plus une paire vers le groupe

                    # --- Cas B : relation satellite → noyau (mononucléaire) -
                    # Le parent peut être un <segment> (comme avant) ou un
                    # <group type="span"> (nouveau : résolu récursivement
                    # vers son UDE tête).
                    parent_head = resolve_head(parent_id)
                    child_head  = (node_data['text'] if node_data['kind'] == 'segment'
                                   else resolve_head(node_id))

                    if parent_head and child_head:
                        self.dataset.append({
                            'edu1':     parent_head,
                            'edu2':     child_head,
                            'relation': relation,
                            'source':   'GUM',
                            'doc_id':   rs4_file.stem,
                        })
                        self.stats['gum_relations'] += 1
                        if nodes[parent_id]['kind'] == 'group':
                            self.stats['gum_mononuc_pairs_via_group'] += 1
                        self.stats['relation_distribution'][relation] += 1

                self.stats['gum_docs'] += 1

            except ET.ParseError as e:
                print(f"  ⚠️  XML invalide {rs4_file.name}: {e}")
            except Exception as e:
                print(f"  ⚠️  Erreur {rs4_file.name}: {e}")

        print(f"✅ GUM : {self.stats['gum_docs']} docs, "
              f"{self.stats['gum_relations']} relations, "
              f"{self.stats['gum_ignored']} ignorées (span/same-unit)")

    # -------------------------------------------------------------------------
    # PARSING SCIDTB
    # -------------------------------------------------------------------------

    def parse_scidtb(self, scidtb_dir='SciDTB/dataset'):
        """
        Parse le corpus SciDTB (fichiers .dep JSON).
        Structure réelle :
          - train : fichiers .dep directement dans train/
          - dev   : fichiers .dep dans dev/gold/
          - test  : fichiers .dep dans test/gold/
        Splits officiels : train (492), dev (154), test (152)
        """
        print("\n" + "=" * 60)
        print("PARSING SCIDTB CORPUS (gold annotations)")
        print("=" * 60)

        for split in ['train', 'dev', 'test']:
            split_dir = Path(scidtb_dir) / split

            if not split_dir.exists():
                print(f"  ⚠️  Split introuvable : {split_dir}")
                continue

            # ✅ FIX : train n'a pas de sous-dossier gold/
            #          dev et test ont leurs fichiers dans gold/
            if split == 'train':
                gold_dir = split_dir
            else:
                gold_dir = split_dir / 'gold'

            if not gold_dir.exists():
                print(f"  ⚠️  Dossier introuvable : {gold_dir}")
                continue

            dep_files = list(gold_dir.glob('*.dep'))
            label = split if split == 'train' else f"{split}/gold"
            print(f"📁 {label} : {len(dep_files)} fichiers")

            errors = 0
            for dep_file in tqdm(dep_files, desc=f"Parsing {split}"):
                try:
                    # ✅ utf-8-sig pour gérer le BOM (byte order mark)
                    with open(dep_file, 'r', encoding='utf-8-sig') as f:
                        data = json.load(f)

                    # ✅ FIX 6 : accès sécurisé avec .get()
                    nodes = data.get('root', [])
                    if not nodes:
                        continue

                    # Extraire les EDUs
                    edus = {}
                    for node in nodes:
                        edu_id   = node.get('id')
                        # ✅ FIX 7 : nettoyage <S> centralisé ici
                        text     = node.get('text', '').replace('<S>', '').strip()
                        parent   = node.get('parent', -1)
                        relation = node.get('relation', '')

                        if edu_id is not None and text:
                            edus[edu_id] = {
                                'text':     text,
                                'parent':   parent,
                                'relation': relation,
                            }

                    # Créer les paires EDU-Relation
                    for edu_id, edu_data in edus.items():
                        relation  = edu_data['relation']
                        parent_id = edu_data['parent']

                        # Ignorer ROOT, null, same-unit
                        if relation in RELATIONS_IGNORE:
                            self.stats['scidtb_ignored'] += 1
                            continue

                        # parent > 0 : exclut ROOT (parent=-1 ou 0)
                        if parent_id > 0 and relation and parent_id in edus:
                            parent_text = edus[parent_id]['text']
                            child_text  = edu_data['text']

                            if parent_text and child_text:
                                self.dataset.append({
                                    'edu1':     parent_text,
                                    'edu2':     child_text,
                                    'relation': relation,
                                    'source':   'SciDTB',
                                    'doc_id':   dep_file.stem,
                                })

                                self.stats['scidtb_relations'] += 1
                                self.stats['relation_distribution'][relation] += 1

                    self.stats['scidtb_docs'] += 1

                except json.JSONDecodeError as e:
                    # ✅ FIX 5 : erreurs visibles (plus de pass silencieux)
                    errors += 1
                    print(f"  ⚠️  JSON invalide {dep_file.name}: {e}")
                except Exception as e:
                    errors += 1
                    print(f"  ⚠️  Erreur {dep_file.name}: {e}")

            if errors:
                print(f"  ⚠️  {errors} fichiers en erreur dans {split}/gold")

        print(f"✅ SciDTB : {self.stats['scidtb_docs']} docs, "
              f"{self.stats['scidtb_relations']} relations, "
              f"{self.stats['scidtb_ignored']} ignorées (ROOT/null/same-unit)")

    # -------------------------------------------------------------------------
    # DIAGNOSTIC : relations non reconnues
    # -------------------------------------------------------------------------

    def print_unmapped_relations(self, merge_mapping: dict):
        """
        Affiche les relations du dataset qui ne sont pas dans le mapping
        de clean_corpus_properly.py → elles seront perdues au nettoyage.
        """
        print("\n" + "=" * 60)
        print("🔍 DIAGNOSTIC : RELATIONS NON MAPPÉES")
        print("   (ces relations seront perdues dans clean_corpus_properly.py)")
        print("=" * 60)

        all_relations = set(item['relation'] for item in self.dataset)
        unmapped = all_relations - set(merge_mapping.keys())

        if not unmapped:
            print("  ✅ Toutes les relations sont couvertes par le mapping !")
            return

        total = len(self.dataset)
        total_lost = 0
        for rel in sorted(unmapped):
            count = sum(1 for item in self.dataset if item['relation'] == rel)
            total_lost += count
            src = set(item['source'] for item in self.dataset
                      if item['relation'] == rel)
            print(f"  ❌ {rel:40s}: {count:5d} ex. [{', '.join(src)}]")

        pct = (total_lost / total * 100) if total > 0 else 0
        print(f"\n  ⚠️  Total perdu si non mappé : "
              f"{total_lost:,} / {total:,} ({pct:.1f}%)")

    # -------------------------------------------------------------------------
    # DÉDUPLICATION
    # -------------------------------------------------------------------------

    def remove_duplicates(self):
        """Supprime les doublons exacts (edu1, edu2, relation)."""
        print("\n🗑️  Suppression des doublons...")

        before = len(self.dataset)
        seen   = set()
        unique = []

        for item in self.dataset:
            key = (item['edu1'], item['edu2'], item['relation'])
            if key not in seen:
                seen.add(key)
                unique.append(item)

        self.dataset = unique
        removed = before - len(self.dataset)

        if removed > 0:
            print(f"  ⚠️  {removed:,} doublons supprimés")
        else:
            print(f"  ✅ Aucun doublon")

    # -------------------------------------------------------------------------
    # SAUVEGARDE
    # -------------------------------------------------------------------------

    def save_dataset(self, output_file='data/gum_scidtb_fused.json'):
        """Sauvegarde le dataset fusionné brut (avant nettoyage)."""
        print(f"\n💾 Sauvegarde...")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.dataset, f, ensure_ascii=False, indent=2)

        print(f"✅ {output_file} — {len(self.dataset):,} paires EDU-Relation")

    # -------------------------------------------------------------------------
    # STATISTIQUES
    # -------------------------------------------------------------------------

    def print_statistics(self):
        """Affiche les statistiques finales."""
        print("\n" + "=" * 60)
        print("📊 STATISTIQUES FINALES")
        print("=" * 60)

        total_docs = self.stats['gum_docs'] + self.stats['scidtb_docs']
        total_rel  = len(self.dataset)

        if total_docs == 0:
            print("\n⚠️  AUCUN DOCUMENT PARSÉ !")
            return

        print(f"\n📚 Documents :")
        print(f"  • GUM    : {self.stats['gum_docs']:4d}")
        print(f"  • SciDTB : {self.stats['scidtb_docs']:4d}")
        print(f"  • TOTAL  : {total_docs:4d}")

        print(f"\n🔗 Relations extraites :")
        print(f"  • GUM    : {self.stats['gum_relations']:6,}")
        print(f"  • SciDTB : {self.stats['scidtb_relations']:6,}")
        print(f"  • TOTAL  : {total_rel:6,}")

        print(f"\n🚫 Relations ignorées (structurelles) :")
        print(f"  • GUM    : {self.stats['gum_ignored']:6,}")
        print(f"  • SciDTB : {self.stats['scidtb_ignored']:6,}")

        print(f"\n🔧 Correction v3 — paires récupérées via <group> :")
        print(f"  • Sœurs multinucléaires (paires adjacentes)      : "
              f"{self.stats.get('gum_multinuc_pairs_recovered', 0):6,}")
        print(f"  • Satellite→noyau via <group type='span'>        : "
              f"{self.stats.get('gum_mononuc_pairs_via_group', 0):6,}")
        print(f"  • Total récupéré (aurait été perdu sous l'ancienne "
              f"logique) : "
              f"{self.stats.get('gum_multinuc_pairs_recovered', 0) + self.stats.get('gum_mononuc_pairs_via_group', 0):6,}")

        print(f"\n📈 Top 15 relations :")
        for rel, count in self.stats['relation_distribution'].most_common(15):
            pct = (count / total_rel * 100) if total_rel > 0 else 0
            bar = '█' * int(pct / 2)
            print(f"  {rel:40s}: {count:5,} ({pct:5.1f}%) {bar}")

        n_unique = len(self.stats['relation_distribution'])
        print(f"\n🎯 Relations uniques dans le dataset brut : {n_unique}")
        print(f"   → Le script clean_corpus_properly.py va les fusionner")
        print(f"     et filtrer pour obtenir vos 23 classes finales.")


# ============================================================================
# MAPPING DE RÉFÉRENCE (pour le diagnostic)
# À synchroniser avec clean_corpus_properly.py
# ============================================================================

MERGE_RELATIONS_REFERENCE = {
    # === GUM (eRST natif) ===
    'adversative-antithesis':    'adversative-antithesis',
    'adversative-concession':    'adversative-concession',
    'adversative-contrast':      'comparison-contrast',
    'attribution-negative':      'attribution-negative',
    'attribution-positive':      'attribution-positive',
    'causal-cause':              'causal-cause',
    'causal-result':             'causal-result',
    'context-background':        'context-background',
    'context-circumstance':      'context-circumstance',
    'contingency-condition':     'contingency-condition',
    'elaboration-additional':    'elaboration-additional',
    'elaboration-attribute':     'elaboration-additional',
    'evaluation-comment':        'evaluation-comment',
    'explanation-evidence':      'explanation-evidence',
    'explanation-justify':       'explanation-justify',
    'explanation-motivation':    'explanation-motivation',
    'joint-disjunction':         'joint',
    'joint-list':                'joint',
    'joint-other':               'joint',
    'joint-sequence':            'temporal-sequence',
    'mode-manner':               'mode-means',
    'mode-means':                'mode-means',
    'organization-heading':      'organization-preparation',
    'organization-phatic':       'organization-phatic',
    'organization-preparation':  'organization-preparation',
    'purpose-attribute':         'purpose-goal',
    'purpose-goal':              'purpose-goal',
    'restatement-partial':       'restatement-partial',
    'restatement-repetition':    'restatement-partial',
    'topic-question':            'topic-question',
    'topic-solutionhood':        'purpose-goal',

    # === SciDTB (fine-grained officiel) ===
    'attribution':               'attribution-positive',
    'bg-related':                'context-background',
    'bg-goal':                   'context-background',
    'bg-general':                'context-background',
    'bg-compare':                'comparison-contrast',   # comparaison avec travaux existants
    'cause':                     'causal-cause',
    'result':                    'causal-result',
    'comparison':                'comparison-contrast',
    'contrast':                  'adversative-antithesis',
    'condition':                 'contingency-condition',
    'elab-addition':             'elaboration-additional',
    'elab-aspect':               'elaboration-additional',
    'elab-process_step':         'elaboration-additional',
    'elab-definition':           'elaboration-additional',
    'elab-enum_member':          'elaboration-additional',
    'elab-example':              'elaboration-additional',
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


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║    FUSION CORPUS GUM + SciDTB POUR BERT RST             ║
║    Version corrigée - eRST .rs4 + SciDTB gold/         ║
╚══════════════════════════════════════════════════════════╝
    """)

    fusion = CorpusFusion()

    # --- Parse GUM ---
    # Adaptez le chemin selon votre installation
    fusion.parse_gum('rstweb')

    # --- Parse SciDTB ---
    # Structure attendue : SciDTB/dataset/train|dev|test/gold/*.dep
    fusion.parse_scidtb('SciDTB/dataset')

    # --- Dédupliquer ---
    fusion.remove_duplicates()

    # --- Sauvegarder ---
    # NOTE : nom de fichier distinct de l'original (gum_scidtb_fused.json)
    # pour permettre une comparaison avant/après la correction v3.
    # Une fois validé, renommer en gum_scidtb_fused.json pour que
    # clean_corpus_properly.py le consomme sans autre changement.
    fusion.save_dataset('data/gum_scidtb_fused_v3.json')

    # --- Statistiques ---
    fusion.print_statistics()

    # --- Diagnostic mapping ---
    fusion.print_unmapped_relations(MERGE_RELATIONS_REFERENCE)

    print("\n" + "=" * 60)
    print("✅ FUSION TERMINÉE")
    print("=" * 60)
    print("\n🎯 PROCHAINE ÉTAPE :")
    print("   python clean_corpus_properly.py")
    print()


if __name__ == "__main__":
    main()