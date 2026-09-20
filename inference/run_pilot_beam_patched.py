"""
generate_educational_corpus_openstax_v2.py
==========================================

"""

import json
import random
import pickle
import re
import os
import html
import shutil
import torch
import numpy as np
from pathlib import Path
from collections import Counter

try:
    import pdfplumber
except ImportError:
    os.system("pip install pdfplumber --break-system-packages")
    import pdfplumber

try:
    import pdfkit
    PDFKIT_AVAILABLE = True
except ImportError:
    PDFKIT_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    os.system("pip install sentence-transformers --break-system-packages")
    try:
        from sentence_transformers import SentenceTransformer
        SBERT_AVAILABLE = True
    except ImportError:
        SBERT_AVAILABLE = False

try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    GPT2_AVAILABLE = True
except ImportError:
    GPT2_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'PDF_DIR': 'openstax_pdfs',   # dossier contenant les PDF OpenStax téléchargés

    # MODE AUTOMATIQUE : si True, le script scanne TOUS les PDF présents
    # dans PDF_DIR, détecte automatiquement les sections (par taille de
    # police des titres) et traite chacune d'elles (S0/S1/S2/S3).
    # Aucune configuration manuelle de CONFIG['SECTIONS'] nécessaire.
    # Si False, seules les sections listées dans CONFIG['SECTIONS'] sont
    # traitées (mode manuel, contrôle fin).
    'AUTO_SCAN': False,

    # Limite de sécurité : nombre max de sections détectées par PDF.
    # None = aucune limite (traite tout le document, peut être long sur
    # un manuel de 1000+ pages : potentiellement 50-150+ sections).
    'MAX_SECTIONS_PER_PDF': None,

    # Sections à extraire MANUELLEMENT (ignoré si AUTO_SCAN=True) :
    # (nom_lisible, fichier_pdf, page_debut, page_fin)
    # Les numéros de page sont ceux affichés DANS le PDF (pas le numéro
    # de page du lecteur, qui inclut souvent la couverture/préface en décalage).
    'SECTIONS': [
        ('Cell Structure',          'biology_2e.pdf',        45,  52),
        ('Photosynthesis',          'biology_2e.pdf',        201, 210),
        ('DNA Replication',         'biology_2e.pdf',        330, 338),
        ('Algorithms Intro',        'intro_cs.pdf',          12,  20),
        ('Newtonian Mechanics',     'university_physics_1.pdf', 88, 96),
        # --- décommente / complète avec tes propres PDF et pages (mode manuel) ---
    ],

    # Documents composites multi-sources, même thème.
    # Chaque thème regroupe plusieurs (fichier_pdf, page_debut, page_fin)
    # traitant du MÊME SUJET dans des manuels/cours différents.
    # Les paragraphes de toutes les sources sont fusionnés en un seul pool,
    # puis mélangés (Version A) ou réordonnés par RST (Version B).
    #
    # Format : { 'Nom du thème': [(fichier_pdf, page_debut, page_fin), ...] }
    #
    # Exemple :
    # 'THEMES': {
    #     'Photosynthesis': [
    #         ('biology_2e.pdf',          201, 206),
    #         ('concepts_of_biology.pdf', 150, 154),
    #         ('ap_biology.pdf',          88,  92),
    #     ],
    # },
    'THEMES': {
        # --- complète avec tes propres PDF et pages ---
    },

    'MIN_WORDS':       40,    # paragraphe minimum
    'MAX_WORDS':       200,   # paragraphe maximum
    'MIN_PARAGRAPHS':  5,     # minimum paragraphes par section
    'MAX_PARAGRAPHS':  10,    # maximum paragraphes à garder

    # Pour les THEMES, on accepte plus de paragraphes par source car ils
    # seront fusionnés (le pool final peut dépasser MAX_PARAGRAPHS).
    'THEME_MAX_PARAGRAPHS_PER_SOURCE': 6,
    'THEME_MIN_TOTAL_PARAGRAPHS':      4,

    'MODEL_PATH':      'deberta_rst_model_v2',

    # ── PILOTE / CORPUS PARTIEL ───────────────────────────────────────────────
    # ARTICLES_JSON : chemin vers articles_raw_clean.json déjà extrait.
    #   → None  = extraire depuis les PDF (premier lancement)
    #   → 'articles_raw_clean.json' = charger directement (skip PDF)
    'ARTICLES_JSON': 'articles_raw_clean.json',

    # N_PILOT : nombre de sections à traiter (sélection équilibrée par manuel).
    #   → 10   = ~2 sections par manuel  (pilote rapide, ~15 min GPU)
    #   → 40   = ~8 sections par manuel  (évaluation intermédiaire)
    #   → None = toutes les sections     (corpus complet, ~3h GPU)
    'N_PILOT': 100,

    'OUTPUT_DIR':      'data/educational_corpus_v2',
    'MAX_LENGTH':      128,
    'SEUIL_MARKER':    0.60,
    'SEUIL_EDU':       0.60,

    # Profil apprenant pour la traversée du graphe RST (S2).
    # Valeurs possibles : 'novice' | 'advanced' | 'review'
    # Détermine la priorité des relations RST lors de la traversée greedy.
    'LEARNER_PROFILE': 'novice',

    # Baseline S1 : ordonnancement par similarité sémantique (Sentence-BERT)
    'SBERT_MODEL_NAME': 'all-MiniLM-L6-v2',

    'THEMES_SUBDIR':   'themes',
    'SEED':            42,
    'WKHTMLTOPDF_PATH': None,
}

random.seed(CONFIG['SEED'])

# ============================================================================
# MARQUEURS DISCURSIFS
# ============================================================================

MARQUEURS = {
    'elaboration-additional': [
        'Furthermore', 'Moreover', 'In addition', 'Additionally', 'Also'
    ],
    'elaboration-attribute': [
        'For example', 'For instance', 'Such as', 'Namely', 'Specifically'
    ],
    'causal-cause': [
        'Because', 'Since', 'Due to', 'As a result of', 'Owing to'
    ],
    'causal-result': [
        'Therefore', 'Thus', 'Hence', 'Consequently', 'As a result'
    ],
    'adversative-antithesis': [
        'However', 'Nevertheless', 'Nonetheless', 'But', 'Yet'
    ],
    'adversative-concession': [
        'Although', 'Even though', 'Despite', 'Admittedly'
    ],
    'comparison-contrast': [
        'Similarly', 'Likewise', 'In contrast', 'On the other hand'
    ],
    'temporal-sequence': [
        'First', 'Then', 'Next', 'Finally', 'Subsequently', 'Afterwards'
    ],
    'purpose-goal': [
        'In order to', 'So that', 'To achieve', 'With the aim of'
    ],
    'context-background': [
        'Historically', 'Traditionally', 'Originally', 'In the past'
    ],
    'restatement-partial': [
        'In other words', 'That is', 'In summary', 'To summarize'
    ],
}


def detect_marker(text):
    text_lower = text.strip().lower()
    for relation, markers in MARQUEURS.items():
        for marker in sorted(markers, key=len, reverse=True):
            pattern = rf'^{re.escape(marker.lower())}[,\s]'
            if re.match(pattern, text_lower):
                conf = min(0.95, 0.70 + len(marker) * 0.02)
                return relation, conf, marker
    return None, 0.0, None


def clean_text(text):
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\{\{.*?\}\}', '', text)
    text = ' '.join(text.split())
    return text.strip()


def count_words(text):
    return len(text.split())


# ============================================================================
# EXTRACTION OPENSTAX (PDF LOCAL)
# ============================================================================

def extract_text_from_pages(pdf_path, page_start, page_end):
    """Extrait le texte brut des pages [page_start, page_end] (1-indexées,
    numéros affichés dans le PDF) avec pdfplumber."""
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        n_pages = len(pdf.pages)
        start = max(1, page_start)
        end = min(n_pages, page_end)
        for i in range(start - 1, end):
            page_text = pdf.pages[i].extract_text() or ''
            text_parts.append(page_text)
    return '\n'.join(text_parts)


def segment_into_paragraphs(raw_text, min_words, max_words, max_para):
    """Découpe un texte brut en paragraphes valides (40-200 mots typiquement).

    Stratégie en deux temps :
    1. Essai par blocs séparés par une ligne vide (\\n\\s*\\n). Fonctionne
       pour les PDF où pdfplumber préserve les sauts de paragraphe.
    2. Si cela produit trop peu de blocs valides (cas fréquent : pdfplumber
       aplatit tout en un flux de lignes sans ligne vide détectable),
       fallback : on nettoie le texte en un flux continu, on découpe en
       phrases, puis on regroupe des phrases consécutives jusqu'à
       atteindre [min_words, max_words] — ce qui reconstitue des
       "paragraphes" de taille pédagogique même sans repère typographique.
    """
    cleaned_full = _clean_pdf_text(raw_text)

    # --- Tentative 1 : blocs séparés par ligne vide ---
    blocks = re.split(r'\n\s*\n', raw_text)
    valid = []
    for block in blocks:
        text = _clean_pdf_text(' '.join(block.split()))
        if not text or _looks_like_noise(text):
            continue
        n = count_words(text)
        if min_words <= n <= max_words:
            valid.append(text)

    if len(valid) >= 2:
        return valid[:max_para]

    # --- Tentative 2 : regroupement de phrases ---
    # Découpe en phrases : point/!/? suivi d'espace + majuscule (ou fin de texte)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', cleaned_full)
    sentences = [s.strip() for s in sentences if s.strip()]

    valid = []
    current_words = []
    for sent in sentences:
        if _looks_like_noise(sent):
            continue
        current_words.extend(sent.split())
        n = len(current_words)

        if n >= min_words:
            text = ' '.join(current_words)
            if n <= max_words:
                valid.append(text)
            else:
                # Trop long : on prend les min_words..max_words premiers mots
                valid.append(' '.join(current_words[:max_words]))
            current_words = []

        if len(valid) >= max_para:
            break

    return valid[:max_para]


def _clean_pdf_text(text):
    """Normalise un texte extrait de PDF en texte brut propre.
    Supprime : tirets de coupure de mots, numéros de page isolés,
    en-têtes/pieds répétés, références numériques [n], espaces multiples.
    """
    # Tirets de coupure de mots en fin de ligne (photo-\nsynth → photosynthesis)
    text = re.sub(r'-\s*\n\s*', '', text)
    # Retours à la ligne simples → espace
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    # Numéros de page isolés sur une ligne (ex: "  42\n" ou "\n42\n")
    text = re.sub(r'(?<!\w)\d{1,4}(?!\w)', '', text)
    # Références bibliographiques [n] ou (n)
    text = re.sub(r'\[\d+\]|\(\d+\)', '', text)
    # Caractères parasites courants dans les extractions PDF
    text = re.sub(r'[•●◆▪▸►]', '', text)
    # Espaces multiples
    text = re.sub(r'\s+', ' ', text)
    return clean_text(text.strip())


def _looks_like_noise(text):
    """Détecte les fragments probablement non pertinents : légendes de
    figure/tableau, numéros de page isolés, titres courts en majuscules."""
    if re.match(r'^(Figure|Table|FIGURE|TABLE)\s*\d', text):
        return True
    if re.match(r'^\d+(\.\d+)*$', text):
        return True
    if text.isupper() and count_words(text) < 8:
        return True
    return False


def detect_sections_by_font_size(pdf_path, max_pages=None):
    """
    Détecte automatiquement les frontières de section dans un PDF en
    repérant les lignes dont la taille de police est significativement
    plus grande que le corps de texte (heuristique typographique).

    Retourne une liste de tuples (titre_detecte, page_debut, page_fin)
    couvrant tout le document (1-indexé, inclusif).
    """
    with pdfplumber.open(pdf_path) as pdf:
        n_pages = len(pdf.pages)
        if max_pages:
            n_pages = min(n_pages, max_pages)

        # Passe 1 : collecter toutes les tailles de police pour estimer
        # la taille du corps de texte (la plus fréquente).
        size_counter = Counter()
        for page in pdf.pages[:n_pages]:
            for ch in page.chars:
                size_counter[round(ch['size'], 1)] += 1

        if not size_counter:
            return [("Document entier", 1, n_pages)]

        body_size = size_counter.most_common(1)[0][0]
        # Seuil : un titre est notablement plus grand que le corps de texte
        title_threshold = body_size * 1.25

        # Passe 2 : pour chaque page, regrouper les caractères en lignes
        # (par position verticale 'top' arrondie) et repérer les lignes
        # dont la taille moyenne dépasse le seuil.
        headings = []  # (page_num, texte_titre)

        for page_idx, page in enumerate(pdf.pages[:n_pages]):
            page_num = page_idx + 1
            lines = {}
            for ch in page.chars:
                key = round(ch['top'], 0)
                lines.setdefault(key, []).append(ch)

            for top, chars in sorted(lines.items()):
                avg_size = sum(c['size'] for c in chars) / len(chars)
                if avg_size >= title_threshold:
                    text = ''.join(c['text'] for c in chars).strip()
                    text = re.sub(r'\s+', ' ', text)
                    # Filtrer le bruit : titres trop courts (1-2 car.)
                    # ou trop longs (probablement un paragraphe en gras)
                    if 3 <= len(text) <= 120:
                        headings.append((page_num, text))

    # Construire les sections à partir des frontières détectées
    sections = []
    if not headings:
        return [("Document entier", 1, n_pages)]

    # Dédupliquer les titres consécutifs sur la même page
    dedup = []
    seen = set()
    for page_num, text in headings:
        key = (page_num, text)
        if key not in seen:
            dedup.append((page_num, text))
            seen.add(key)

    for i, (page_num, title) in enumerate(dedup):
        start = page_num
        end = dedup[i + 1][0] if i + 1 < len(dedup) else n_pages
        # Une section doit couvrir au moins 1 page
        end = max(end, start)
        sections.append((title, start, end))

    return sections


def discover_sections(pdf_dir, max_sections_per_pdf=None):
    """
    Scanne tous les fichiers .pdf de pdf_dir, détecte automatiquement
    les sections de chacun (via detect_sections_by_font_size), et
    retourne une liste au même format que CONFIG['SECTIONS'] :
        (nom_section, fichier_pdf, page_debut, page_fin)

    Le nom de section combine le titre détecté et le nom du fichier
    pour rester lisible et traçable même si plusieurs PDF ont des
    titres similaires.
    """
    pdf_dir_path = Path(pdf_dir)
    if not pdf_dir_path.exists():
        print(f"   ❌ Dossier introuvable : {pdf_dir_path}")
        return []

    pdf_files = sorted(pdf_dir_path.glob('*.pdf'))
    if not pdf_files:
        print(f"   ❌ Aucun fichier .pdf trouvé dans {pdf_dir_path}")
        return []

    discovered = []
    for pdf_path in pdf_files:
        print(f"   🔍 Analyse de {pdf_path.name}...")
        try:
            sections = detect_sections_by_font_size(pdf_path)
        except Exception as e:
            print(f"      ❌ erreur lors de la détection ({e}) — fichier ignoré")
            continue

        if max_sections_per_pdf:
            sections = sections[:max_sections_per_pdf]

        print(f"      → {len(sections)} sections détectées")

        for title, start, end in sections:
            section_name = f"{title} [{pdf_path.stem}]"
            discovered.append((section_name, pdf_path.name, start, end, title))

    return discovered


def extract_section(name, pdf_filename, page_start, page_end,
                    pdf_dir, min_words, max_words, min_para, max_para,
                    heading_text=None):
    """Extrait et segmente une section d'un manuel OpenStax PDF local.

    Si heading_text est fourni (titre de section détecté automatiquement),
    il est retiré du début du texte brut avant segmentation, pour éviter
    qu'il ne soit fusionné avec le premier paragraphe.
    """
    pdf_path = Path(pdf_dir) / pdf_filename

    if not pdf_path.exists():
        print(f"   ❌ {name} : fichier introuvable ({pdf_path})")
        return None

    try:
        raw_text = extract_text_from_pages(pdf_path, page_start, page_end)
    except Exception as e:
        print(f"   ❌ {name} : erreur de lecture PDF ({e})")
        return None

    if heading_text:
        # Retire la première occurrence du titre (avec espaces normalisés)
        pattern = re.escape(' '.join(heading_text.split()))
        raw_text = re.sub(pattern, '', raw_text, count=1)

    valid = segment_into_paragraphs(raw_text, min_words, max_words, max_para)

    if len(valid) < min_para:
        print(f"   ⚠️  {name} : seulement {len(valid)} paragraphes valides "
              f"(pages {page_start}-{page_end})")
        return None

    print(f"   ✅ {name} : {len(valid)} paragraphes (pages {page_start}-{page_end})")
    return {
        'topic':       name,
        'source':      'OpenStax (CC BY)',
        'pdf_file':    pdf_filename,
        'pages':       f"{page_start}-{page_end}",
        'paragraphs':  valid[:max_para],
    }


def extract_theme_source(pdf_filename, page_start, page_end, pdf_dir,
                          min_words, max_words, max_para_per_source):
    """Extrait les paragraphes valides d'une source pour un document
    composite thématique. Pas de seuil min_para strict ici : chaque
    paragraphe valide est conservé (jusqu'à max_para_per_source),
    annoté avec sa source d'origine pour traçabilité."""
    pdf_path = Path(pdf_dir) / pdf_filename

    if not pdf_path.exists():
        print(f"      ❌ fichier introuvable ({pdf_path})")
        return []

    try:
        raw_text = extract_text_from_pages(pdf_path, page_start, page_end)
    except Exception as e:
        print(f"      ❌ erreur de lecture PDF ({e})")
        return []

    valid = segment_into_paragraphs(raw_text, min_words, max_words,
                                     max_para_per_source)

    print(f"      ✅ {pdf_filename} (pages {page_start}-{page_end}) : "
          f"{len(valid)} paragraphes")

    source_label = f"{pdf_filename} (p. {page_start}-{page_end})"
    return [
        {'text': text, 'source': source_label}
        for text in valid
    ]


# ============================================================================
# PRÉDICTION RST HYBRIDE
# ============================================================================

def load_model(model_path):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer     = AutoTokenizer.from_pretrained(model_path)
    device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model         = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    with open(Path(model_path) / 'label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    return model, tokenizer, label_encoder, device


def predict_deberta(model, tokenizer, label_encoder, device, text1, text2):
    encoding = tokenizer(
        text1, text2,
        add_special_tokens=True,
        max_length=CONFIG['MAX_LENGTH'],
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids      = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs   = torch.softmax(outputs.logits, dim=1)

    top_val, top_idx = torch.topk(probs[0], k=3)
    top3 = [
        (label_encoder.inverse_transform([i.item()])[0], v.item())
        for i, v in zip(top_idx, top_val)
    ]
    return top3[0][0], top3[0][1], top3


def get_sentences_filtered(text):
    """Segmente un paragraphe en phrases valides (≥ 5 mots, sans artefacts PDF)."""
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    sents = [s.strip() for s in sents if len(s.split()) >= 5]
    # Filtrer les artefacts PDF courants (crédits, URLs, numéros de page)
    sents = [s for s in sents if not re.match(
        r'^(credit|access for free|figure \d|openstax\.org|\d{1,4}\s)', s, re.I)]
    return sents


def extract_edu_boundary(para1, para2):
    """
    Mode C (max_conf) — deux EDUs séparés combinés par confiance maximale :
      EDU1 : avant-dernière phrase de para1 + première phrase de para2
      EDU2 : dernière phrase de para1       + première phrase de para2

    Chaque paire reste dans la distribution d'entraînement GUM+SciDTB
    (2 phrases courtes). La prédiction la plus confiante est retenue.

    Amélioration vs Mode A (1 EDU) : +4.7pp de confiance moyenne
    sur 7 paires Photosynthesis (Biology 2e), 5/7 accords avec Mode A.
    """
    sents1 = get_sentences_filtered(para1)
    sents2 = get_sentences_filtered(para2)
    t2 = sents2[0] if sents2 else para2[:200]

    # EDU2 = dernière phrase P1 (Mode A baseline)
    t1_edu2 = sents1[-1] if sents1 else para1[:200]

    # EDU1 = avant-dernière phrase P1 (contexte additionnel)
    if len(sents1) >= 2:
        t1_edu1 = sents1[-2]
        return t1_edu1, t1_edu2, t2   # retourner les 3 pour predict_pair_hybrid
    else:
        return None, t1_edu2, t2       # EDU1 indisponible → fallback Mode A


def predict_pair_hybrid(para1, para2, model, tokenizer,
                        label_encoder, device):
    """
    Prédit la relation RST entre deux paragraphes.
    Cascade à trois niveaux :
      1. Marqueur discursif explicite au début de para2 (règle déterministe,
         conf ≥ SEUIL_MARKER) — rapide et très précis quand présent.
      2. Mode C (max_conf) : deux EDUs séparés, retenir la confiance maximale.
           EDU1 = avant-dernière phrase P1 + première phrase P2
           EDU2 = dernière phrase P1       + première phrase P2
         → Chaque paire reste dans la distribution d'entraînement.
         → +4.7pp de confiance vs Mode A sur corpus Photosynthesis.
      3. Fallback Mode A si EDU1 indisponible (paragraphe trop court).
    """
    # Niveau 1 : marqueur discursif
    marker_rel, marker_conf, marker = detect_marker(para2)
    if marker_rel and marker_conf >= CONFIG['SEUIL_MARKER']:
        return marker_rel, 'marker', marker_conf, marker

    # Extraction EDU boundaries
    t1_edu1, t1_edu2, t2 = extract_edu_boundary(para1, para2)

    # Niveau 2 : Mode C — deux EDUs séparés, max confiance
    if t1_edu1 is not None:
        rel1, conf1, _ = predict_deberta(model, tokenizer, label_encoder, device,
                                          t1_edu1, t2)
        rel2, conf2, _ = predict_deberta(model, tokenizer, label_encoder, device,
                                          t1_edu2, t2)
        # max_conf : retenir la prédiction la plus confiante
        if conf1 >= conf2:
            return rel1, 'edu_boundary_C1', conf1, None
        else:
            return rel2, 'edu_boundary_C2', conf2, None

    # Niveau 3 : fallback Mode A (paragraphe court, EDU1 indisponible)
    rel2, conf2, _ = predict_deberta(model, tokenizer, label_encoder, device,
                                      t1_edu2, t2)
    return rel2, 'edu_boundary_A', conf2, None


# ============================================================================
# RST-BASED DISCOURSE GRAPH  G = (V, E)
# Aligné sur §3.5 du manuscrit IP&M :
#   - Nœuds V = paragraphes
#   - Arêtes dirigées (vi, vj, r) = relation RST prédite par DeBERTa
#   - Graphe COMPLET : toutes les N×(N-1) paires dirigées
#   - Traversée conditionnée par profil apprenant
# ============================================================================

# Priorité de chaque relation selon le profil apprenant.
# Valeur basse = relation préférée en premier lors de la traversée greedy.
PROFILE_WEIGHTS = {
    'novice': {
        # Novice : d'abord le contexte et les définitions, puis élaboration
        'context-background':     1,
        'elaboration-additional': 2,
        'elaboration-attribute':  2,
        'purpose-goal':           3,
        'causal-cause':           4,
        'causal-result':          4,
        'explanation-evidence':   5,
        'explanation-justify':    5,
        'adversative-antithesis': 6,
        'adversative-concession': 6,
        'comparison-contrast':    6,
        'evaluation-comment':     7,
        'restatement-partial':    8,
    },
    'advanced': {
        # Avancé : d'abord preuves et justifications, puis causes/contrastes
        'explanation-evidence':   1,
        'explanation-justify':    1,
        'causal-cause':           2,
        'causal-result':          2,
        'adversative-antithesis': 3,
        'comparison-contrast':    3,
        'adversative-concession': 4,
        'elaboration-additional': 5,
        'elaboration-attribute':  5,
        'purpose-goal':           6,
        'context-background':     7,
        'evaluation-comment':     7,
        'restatement-partial':    8,
    },
    'review': {
        # Révision : d'abord résumés et reformulations, puis évaluation
        'restatement-partial':    1,
        'evaluation-comment':     2,
        'explanation-justify':    3,
        'causal-result':          4,
        'causal-cause':           4,
        'elaboration-additional': 5,
        'elaboration-attribute':  5,
        'explanation-evidence':   6,
        'adversative-antithesis': 7,
        'comparison-contrast':    7,
        'adversative-concession': 7,
        'purpose-goal':           8,
        'context-background':     9,
    },
}

# Nœud Nucleus dans chaque relation (celui qui porte le contenu essentiel)
# Utilisé pour identifier le nœud de départ optimal du graphe.
NUCLEUS_ROLE = {
    'elaboration-additional': 'source',   # vi élabore vj → vi est noyau
    'elaboration-attribute':  'source',
    'explanation-evidence':   'source',
    'explanation-justify':    'source',
    'causal-cause':           'target',   # la cause mène à l'effet
    'causal-result':          'source',
    'context-background':     'target',   # le contexte précède le noyau
    'purpose-goal':           'target',
    'adversative-antithesis': 'source',
    'adversative-concession': 'source',
    'comparison-contrast':    'source',
    'evaluation-comment':     'source',
    'restatement-partial':    'source',
    'temporal-sequence':      'source',
}


def build_rst_discourse_graph(paragraphs, model, tokenizer,
                               label_encoder, device, sbert_model=None):
    """
    Construit le graphe de discours RST G = (V, E) complet.

    Prédit toutes les N×(N-1) paires dirigées (vi → vj).
    Si sbert_model est fourni, calcule aussi la matrice de similarité
    SBERT pour le score hybride RST+SBERT lors de la traversée.

    Retourne :
        graph      : dict[(i,j)] = {'relation', 'confidence', 'strategy',
                                    'sbert_sim'} — sbert_sim=None si pas SBERT
        adjacency  : np.array (N,N) — confiances RST
        sbert_sim  : np.array (N,N) — similarités SBERT (None si indisponible)
    """
    n          = len(paragraphs)
    graph      = {}
    adjacency  = np.zeros((n, n), dtype=np.float32)
    sbert_sim  = None

    # ── Matrice de similarité SBERT (une seule passe d'encodage) ─────────────
    if sbert_model is not None:
        embeddings = sbert_model.encode(paragraphs, show_progress_bar=False,
                                        convert_to_numpy=True)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        emb_norm  = embeddings / norms
        sbert_sim = np.dot(emb_norm, emb_norm.T).astype(np.float32)
        np.fill_diagonal(sbert_sim, 0.0)

    # ── Prédictions RST N×(N-1) ───────────────────────────────────────────────
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            rel, strategy, conf, _ = predict_pair_hybrid(
                paragraphs[i], paragraphs[j],
                model, tokenizer, label_encoder, device
            )
            sim = float(sbert_sim[i, j]) if sbert_sim is not None else None
            graph[(i, j)] = {
                'relation':   rel,
                'confidence': round(float(conf), 4),
                'strategy':   strategy,
                'sbert_sim':  round(sim, 4) if sim is not None else None,
            }
            adjacency[i, j] = conf

    return graph, adjacency, sbert_sim


def traverse_rst_graph(paragraphs, graph, adjacency, profile='novice',
                       alpha=1.0, sbert_sim=None):
    """
    Traversée greedy du graphe RST G avec score hybride RST+SBERT.

    Score de sélection à chaque étape :
        score = alpha × conf_RST(i→j)
              + (1-alpha) × sim_SBERT(i,j)
              - 0.05 × profile_weight(relation)

    alpha = 1.0 → RST pur
    alpha = 0.3 → meilleur compromis RST/SBERT (ablation study)
    """
    n       = len(paragraphs)
    weights = PROFILE_WEIGHTS.get(profile, PROFILE_WEIGHTS['novice'])

    if n <= 1:
        return list(paragraphs), [], list(range(n))

    start    = 0
    visited  = [False] * n
    ordering = [start]
    visited[start] = True
    traversal_info = []

    while len(ordering) < n:
        current    = ordering[-1]
        best_idx   = -1
        best_score = -np.inf
        best_edge  = None

        for j in range(n):
            if visited[j]:
                continue
            edge = graph[(current, j)]
            rel  = edge['relation']
            conf = edge['confidence']
            pw   = weights.get(rel, 5)
            sim  = edge.get('sbert_sim') or 0.0
            score = (alpha * conf
                     + (1.0 - alpha) * sim
                     - 0.05 * pw)

            if score > best_score:
                best_score = score
                best_idx   = j
                best_edge  = edge

        if best_idx == -1:
            best_idx  = next(j for j in range(n) if not visited[j])
            best_edge = graph[(current, best_idx)]

        ordering.append(best_idx)
        visited[best_idx] = True
        traversal_info.append({
            'from':       current,
            'to':         best_idx,
            'relation':   best_edge['relation'],
            'confidence': best_edge['confidence'],
            'sbert_sim':  best_edge.get('sbert_sim'),
            'strategy':   best_edge['strategy'],
            'profile':    profile,
            'alpha':      alpha,
        })

    ordered = [paragraphs[i] for i in ordering]
    return ordered, traversal_info, ordering


def traverse_beam_search(paragraphs, graph, profile='novice',
                         beam_width=3, alpha=1.0):
    """
    Beam Search sur le graphe RST — remplace le greedy local par une
    exploration de k chemins en parallèle.

    Principe :
      À chaque étape, on maintient les `beam_width` meilleurs chemins
      partiels. Pour chaque chemin, on étend vers tous les nœuds non
      visités et on garde les beam_width meilleurs selon le score cumulé
      moyen (cohérence globale du chemin, pas seulement l'arête locale).

    Score d'un chemin = moyenne des scores de toutes ses arêtes :
        score_path = mean(alpha × conf_RST(i→j)
                         + (1-alpha) × sim_SBERT(i,j)
                         - 0.05 × profile_weight)

    Avantage vs greedy :
      Le greedy peut choisir P1→P7 (conf=0.95) même si P7 bloque
      l'accès aux nœuds restants. Le beam search détecte ce piège
      en comparant plusieurs chemins simultanément.

    beam_width=1 → équivalent au greedy (pour comparaison).
    beam_width=3 → recommandé (bon équilibre qualité/coût).
    beam_width=5 → meilleur mais plus lent.

    Retourne :
        ordered_paragraphs, traversal_info, ordering
    """
    n       = len(paragraphs)
    weights = PROFILE_WEIGHTS.get(profile, PROFILE_WEIGHTS['novice'])

    if n <= 1:
        return list(paragraphs), [], list(range(n))

    def edge_score(i, j):
        """Score d'une arête selon le critère hybride."""
        edge = graph[(i, j)]
        conf = edge['confidence']
        pw   = weights.get(edge['relation'], 5)
        sim  = edge.get('sbert_sim') or 0.0
        return (alpha * conf + (1.0 - alpha) * sim - 0.05 * pw)

    # État d'un beam : (score_cumulé_moyen, chemin, traversal_edges)
    # score = moyenne des scores des arêtes du chemin (normalisé par longueur)
    # Initialisation depuis P1
    beams = [(0.0, [0], [])]   # (score, ordering, edges)

    for step in range(n - 1):
        candidates = []

        for beam_score, ordering, edges in beams:
            current  = ordering[-1]
            visited  = set(ordering)

            for j in range(n):
                if j in visited:
                    continue
                s  = edge_score(current, j)
                e  = graph[(current, j)]

                # Score cumulé moyen sur tout le chemin
                new_score = (beam_score * len(edges) + s) / (len(edges) + 1)
                new_path  = ordering + [j]
                new_edges = edges + [{
                    'from':       current,
                    'to':         j,
                    'relation':   e['relation'],
                    'confidence': e['confidence'],
                    'sbert_sim':  e.get('sbert_sim'),
                    'strategy':   e['strategy'],
                    'profile':    profile,
                    'alpha':      alpha,
                    'edge_score': round(s, 4),
                }]
                candidates.append((new_score, new_path, new_edges))

        # Garder les beam_width meilleurs candidats
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_width]

    # Meilleur chemin final
    best_score, best_ordering, best_edges = beams[0]
    ordered = [paragraphs[i] for i in best_ordering]
    return ordered, best_edges, best_ordering


def traverse_beam_search_candidates(paragraphs, graph, profile='novice',
                                     beam_width=5, alpha=1.0):
    """
    Variante de traverse_beam_search() qui NE COLLAPSE PAS sur le meilleur
    chemin : retourne les `beam_width` chemins finaux survivants, triés par
    score local décroissant (le même score que traverse_beam_search()
    utilise pour choisir beams[0]).

    Objectif : permettre un RE-CLASSEMENT GLOBAL des k candidats finaux
    (cf. rerank_beam_candidates_global) au lieu de se fier uniquement à la
    moyenne des scores locaux (Eq. 5-6 / Algorithme 2), qui peut favoriser
    une séquence localement confiante mais globalement sous-optimale
    (question du Reviewer #1).

    Retourne : liste de dicts triée par score décroissant, un par candidat :
        [{'score': float, 'ordering': [...], 'edges': [...]}, ...]
    """
    n       = len(paragraphs)
    weights = PROFILE_WEIGHTS.get(profile, PROFILE_WEIGHTS['novice'])

    if n <= 1:
        return [{'score': 0.0, 'ordering': list(range(n)), 'edges': []}]

    def edge_score(i, j):
        edge = graph[(i, j)]
        conf = edge['confidence']
        pw   = weights.get(edge['relation'], 5)
        sim  = edge.get('sbert_sim') or 0.0
        return (alpha * conf + (1.0 - alpha) * sim - 0.05 * pw)

    beams = [(0.0, [0], [])]

    for step in range(n - 1):
        candidates = []
        for beam_score, ordering, edges in beams:
            current = ordering[-1]
            visited = set(ordering)
            for j in range(n):
                if j in visited:
                    continue
                s = edge_score(current, j)
                e = graph[(current, j)]
                new_score = (beam_score * len(edges) + s) / (len(edges) + 1)
                new_path  = ordering + [j]
                new_edges = edges + [{
                    'from':       current,
                    'to':         j,
                    'relation':   e['relation'],
                    'confidence': e['confidence'],
                    'sbert_sim':  e.get('sbert_sim'),
                    'strategy':   e['strategy'],
                    'profile':    profile,
                    'alpha':      alpha,
                    'edge_score': round(s, 4),
                }]
                candidates.append((new_score, new_path, new_edges))
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_width]

    return [
        {'score': score, 'ordering': ordering, 'edges': edges}
        for score, ordering, edges in beams
    ]


def rerank_beam_candidates_global(paragraphs, candidates, graph, sbert_model,
                                   gamma=0.6, use_gpt2=False):
    """
    Re-classe les k candidats finaux du beam search (traverse_beam_search_
    candidates) selon un critère GLOBAL et indépendant de DeBERTa, au lieu
    du seul score local moyen (Eq. 6).

        final_score = gamma       × norm(RST_coherence)
                    + (1-gamma-d) × norm(SBERT_coherence)
                    + d           × norm(-GPT2_perplexity)   [si use_gpt2]

    Les trois métriques sont normalisées min-max SUR LES k CANDIDATS de
    cette section uniquement (elles vivent sur des échelles différentes),
    donc gamma est comparable d'une section à l'autre — même logique que
    l'ablation α déjà présente dans le script (ALPHAS).

    gamma=1.0  → reproduit exactement le choix actuel (beams[0]),
                 sert de test de non-régression.
    gamma=0.0  → classement purement SBERT (ignore complètement RST).

    Retourne : (best_candidate_dict, report_dict)
    """
    rst_vals   = []
    sbert_vals = []
    for c in candidates:
        rst, _ = coherence_from_graph(c['ordering'], graph)
        ordered_paras = [paragraphs[i] for i in c['ordering']]
        sbert = sbert_coherence_score(ordered_paras, sbert_model) or 0.0
        rst_vals.append(rst)
        sbert_vals.append(sbert)

    def _norm(vals, higher_is_better=True):
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-9:
            return [1.0] * len(vals)
        n = [(v - lo) / (hi - lo) for v in vals]
        return n if higher_is_better else [1.0 - v for v in n]

    rst_norm   = _norm(rst_vals, True)
    sbert_norm = _norm(sbert_vals, True)

    delta = 0.0
    gpt2_vals = [None] * len(candidates)
    gpt2_norm = [0.0] * len(candidates)
    if use_gpt2:
        delta = 0.2
        for idx, c in enumerate(candidates):
            ordered_paras = [paragraphs[i] for i in c['ordering']]
            gpt2_vals[idx] = gpt2_perplexity(ordered_paras)
        gpt2_norm = _norm([v if v is not None else 1e9 for v in gpt2_vals], False)

    beta_weight = 1.0 - gamma - delta
    finals = [
        gamma * r + beta_weight * s + delta * g
        for r, s, g in zip(rst_norm, sbert_norm, gpt2_norm)
    ]

    best_idx = int(np.argmax(finals))
    report = {
        'n_candidates':   len(candidates),
        'rst_coherence':  [round(v, 4) for v in rst_vals],
        'sbert_coherence':[round(v, 4) if v is not None else None for v in sbert_vals],
        'gpt2_perplexity':[round(v, 1) if v is not None else None for v in gpt2_vals],
        'final_scores':   [round(v, 4) for v in finals],
        'chosen_index':   best_idx,
        'changed_vs_beam_only': best_idx != 0,  # candidates[0] = current paper's pick
    }
    return candidates[best_idx], report


def _select_start_node(paragraphs, graph, n):
    """
    Choisit le nœud de départ optimal pour la traversée :
    le paragraphe avec le plus fort score de sortie pondéré
    (somme des confiances sur ses arêtes sortantes).
    En pratique correspond souvent au paragraphe introductif.
    """
    out_scores = []
    for i in range(n):
        score = sum(graph[(i, j)]['confidence']
                    for j in range(n) if j != i)
        out_scores.append(score)
    return int(np.argmax(out_scores))


def compute_coherence_score(paragraphs, model, tokenizer,
                             label_encoder, device):
    """
    Score de cohérence RST sur une séquence linéaire de paragraphes.
    Utilisé pour évaluer S0 et S1 (paires adjacentes uniquement).
    Pour S2, les scores sont lus directement depuis le graphe (plus rapide).
    """
    if len(paragraphs) < 2:
        return 0.0, []

    scores    = []
    relations = []

    for i in range(len(paragraphs) - 1):
        rel, strategy, conf, marker = predict_pair_hybrid(
            paragraphs[i], paragraphs[i+1],
            model, tokenizer, label_encoder, device
        )
        scores.append(conf)
        relations.append({
            'para_from':  i,
            'para_to':    i + 1,
            'relation':   rel,
            'strategy':   strategy,
            'confidence': round(conf, 3),
            'marker':     marker,
        })

    return float(np.mean(scores)), relations


def coherence_from_graph(ordering, graph):
    """
    Calcule le score de cohérence RST de S2 directement depuis le graphe
    (sans nouvel appel à DeBERTa) — score sur les paires adjacentes
    dans l'ordre traversé.
    """
    if len(ordering) < 2:
        return 0.0, []
    scores    = []
    relations = []
    for k in range(len(ordering) - 1):
        i, j = ordering[k], ordering[k + 1]
        edge  = graph[(i, j)]
        scores.append(edge['confidence'])
        relations.append({
            'para_from':  i,
            'para_to':    j,
            'relation':   edge['relation'],
            'confidence': edge['confidence'],
            'strategy':   edge['strategy'],
        })
    return float(np.mean(scores)), relations


# ============================================================================
# MÉTRIQUES INDÉPENDANTES (sans appel à DeBERTa)
# ============================================================================

def sbert_coherence_score(paragraphs, sbert_model):
    """
    Cohérence sémantique SBERT : moyenne des similarités cosinus
    entre paragraphes adjacents.
    Métrique indépendante de DeBERTa — élimine la circularité pour S2.
    ↑ = meilleur (paragraphes thématiquement proches et enchaînés).
    """
    if len(paragraphs) < 2 or sbert_model is None:
        return None
    embeddings = sbert_model.encode(paragraphs, show_progress_bar=False,
                                    convert_to_numpy=True)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    embeddings = embeddings / norms
    sims = [float(np.dot(embeddings[i], embeddings[i+1]))
            for i in range(len(embeddings) - 1)]
    return round(float(np.mean(sims)), 4)


_gpt2_model     = None
_gpt2_tokenizer = None
_gpt2_device    = None

def _load_gpt2():
    global _gpt2_model, _gpt2_tokenizer, _gpt2_device
    if _gpt2_model is None and GPT2_AVAILABLE:
        print("   📦 Chargement GPT-2 (perplexité)...")
        _gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        _gpt2_model     = GPT2LMHeadModel.from_pretrained('gpt2')
        _gpt2_device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _gpt2_model.to(_gpt2_device)
        _gpt2_model.eval()
        print(f"   ✓ GPT-2 chargé | {_gpt2_device}")


def gpt2_perplexity(paragraphs, window=512, stride=256):
    """
    Perplexité GPT-2 sur la concaténation des paragraphes.
    Métrique indépendante de DeBERTa — mesure la fluidité textuelle.
    ↓ = meilleur (texte plus naturel et fluide).
    Utilise une fenêtre glissante pour les textes longs.
    """
    if not GPT2_AVAILABLE:
        return None
    _load_gpt2()
    text      = ' '.join(paragraphs)
    encodings = _gpt2_tokenizer(text, return_tensors='pt')
    input_ids = encodings['input_ids'][0]

    if len(input_ids) == 0:
        return None

    if len(input_ids) <= window:
        chunk = input_ids.unsqueeze(0).to(_gpt2_device)
        with torch.no_grad():
            loss = _gpt2_model(chunk, labels=chunk).loss
        return round(float(torch.exp(loss).cpu()), 3)

    losses = []
    for i in range(0, len(input_ids) - window, stride):
        chunk = input_ids[i:i+window].unsqueeze(0).to(_gpt2_device)
        with torch.no_grad():
            loss = _gpt2_model(chunk, labels=chunk).loss
        losses.append(loss.item())
    return round(float(np.exp(np.mean(losses))), 3) if losses else None


# ============================================================================
# BASELINE S1 — ORDONNANCEMENT PAR SIMILARITÉ SÉMANTIQUE (Sentence-BERT)
# ============================================================================

def load_sbert_model(model_name):
    """Charge un modèle Sentence-BERT pour la baseline S2."""
    if not SBERT_AVAILABLE:
        return None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return SentenceTransformer(model_name, device=str(device))


def order_paragraphs_by_similarity(paragraphs, sbert_model):
    """
    Baseline S2 : ordonne les paragraphes par similarité sémantique
    (Sentence-BERT + similarité cosinus), avec une stratégie gloutonne
    identique dans son principe à order_paragraphs_by_rst (S3) — à
    chaque étape, on choisit le paragraphe restant le plus similaire
    au paragraphe courant.

    Cela teste si un simple regroupement par proximité thématique
    suffit à produire un ordre cohérent, sans tenir compte des
    relations rhétoriques (cause, contraste, élaboration, etc.).
    """
    if len(paragraphs) <= 1 or sbert_model is None:
        return list(paragraphs), []

    embeddings = sbert_model.encode(paragraphs, convert_to_numpy=True)

    # Normaliser pour similarité cosinus = produit scalaire
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    embeddings = embeddings / norms

    ordered_idx = [0]
    remaining   = list(range(1, len(paragraphs)))
    sims        = []

    while remaining:
        current_emb = embeddings[ordered_idx[-1]]
        best_idx = None
        best_sim = -2.0

        for idx in remaining:
            sim = float(np.dot(current_emb, embeddings[idx]))
            if sim > best_sim:
                best_sim = sim
                best_idx = idx

        ordered_idx.append(best_idx)
        sims.append(round(best_sim, 3))
        remaining.remove(best_idx)

    ordered = [paragraphs[i] for i in ordered_idx]
    return ordered, sims


def order_theme_items_by_similarity(items, sbert_model):
    """Variante de order_paragraphs_by_similarity pour items
    {'text': ..., 'source': ...} (mode THEMES)."""
    if len(items) <= 1 or sbert_model is None:
        return list(items), []

    texts = [it['text'] for it in items]
    ordered_texts, sims = order_paragraphs_by_similarity(texts, sbert_model)

    # Reconstituer les items dans le nouvel ordre (texte → item)
    # On suppose les textes uniques ; sinon on retombe sur un mapping par index.
    text_to_item = {}
    for it in items:
        text_to_item.setdefault(it['text'], []).append(it)

    ordered_items = []
    for text in ordered_texts:
        ordered_items.append(text_to_item[text].pop(0))

    return ordered_items, sims


def order_theme_items_by_rst(items, model, tokenizer, label_encoder, device,
                              profile='novice'):
    """
    Variante de traverse_rst_graph pour des items au format
    {'text': ..., 'source': ...} (mode THEMES, multi-sources).
    Construit le graphe RST complet sur les textes puis traverse selon profil.
    """
    if len(items) <= 1:
        return items, []

    texts = [it['text'] for it in items]
    graph, _ = build_rst_discourse_graph(
        texts, model, tokenizer, label_encoder, device)
    _, traversal_info, ordering = traverse_rst_graph(
        texts, graph, None, profile=profile)

    ordered_items = [items[i] for i in ordering]
    all_rels      = [t['relation'] for t in traversal_info]
    return ordered_items, all_rels


def compute_theme_coherence(items, model, tokenizer, label_encoder, device):
    """Comme compute_coherence_score mais pour items {'text','source'}."""
    if len(items) < 2:
        return 0.0, []

    scores    = []
    relations = []

    for i in range(len(items) - 1):
        rel, strategy, conf, marker = predict_pair_hybrid(
            items[i]['text'], items[i+1]['text'],
            model, tokenizer, label_encoder, device
        )
        scores.append(conf)
        relations.append({
            'para_from': i,
            'para_to':   i+1,
            'relation':  rel,
            'strategy':  strategy,
            'confidence': round(conf, 3),
            'marker':    marker,
        })

    return float(np.mean(scores)), relations


# ============================================================================
# EXPORT HTML / PDF — DOCUMENTS COMPOSITES PAR THÈME
# ============================================================================

RELATION_LABELS_FR = {
    'elaboration-additional': 'Élaboration (ajout)',
    'elaboration-attribute':  'Élaboration (exemple)',
    'causal-cause':           'Cause',
    'causal-result':          'Conséquence',
    'adversative-antithesis': 'Contraste',
    'adversative-concession': 'Concession',
    'comparison-contrast':    'Comparaison',
    'temporal-sequence':      'Séquence temporelle',
    'purpose-goal':           'Objectif',
    'context-background':     'Contexte',
    'restatement-partial':    'Résumé',
}


def render_theme_html(theme_name, items, relations, score, version_label,
                       version_desc):
    """Génère le HTML d'un document composite (Version A ou B) pour un thème."""
    rows = []
    for i, item in enumerate(items):
        text_esc = html.escape(item['text'])
        source_esc = html.escape(item['source'])
        rows.append(
            f'<div class="paragraph">\n'
            f'  <div class="source-tag">Source : {source_esc}</div>\n'
            f'  <p>{text_esc}</p>\n'
            f'</div>'
        )
        if i < len(relations):
            rel = relations[i]
            rel_key = rel['relation'] if isinstance(rel, dict) else rel
            rel_label = RELATION_LABELS_FR.get(rel_key, rel_key or '—')
            conf = rel.get('confidence') if isinstance(rel, dict) else None
            conf_str = f" (confiance {conf:.2f})" if conf is not None else ""
            rows.append(
                f'<div class="relation-badge">↓ {html.escape(rel_label)}'
                f'{conf_str}</div>'
            )

    body = '\n'.join(rows)
    score_str = f"{score:.3f}" if score else "—"

    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>{html.escape(theme_name)} — {html.escape(version_label)}</title>
<style>
  body {{
    font-family: Georgia, 'Times New Roman', serif;
    max-width: 760px;
    margin: 40px auto;
    padding: 0 20px;
    line-height: 1.6;
    color: #222;
  }}
  h1 {{ font-size: 1.6em; margin-bottom: 0; }}
  .subtitle {{ color: #666; margin-top: 4px; margin-bottom: 24px; }}
  .meta {{
    background: #f4f4f4;
    border-left: 4px solid #888;
    padding: 10px 14px;
    margin-bottom: 30px;
    font-size: 0.9em;
    color: #444;
  }}
  .paragraph {{ margin-bottom: 4px; }}
  .source-tag {{
    font-size: 0.75em;
    color: #999;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
  }}
  .relation-badge {{
    display: inline-block;
    font-size: 0.8em;
    color: #1a5276;
    background: #eaf2f8;
    border-radius: 4px;
    padding: 2px 10px;
    margin: 6px 0 14px 0;
  }}
  p {{ margin: 0 0 14px 0; text-align: justify; }}
</style>
</head>
<body>
<h1>{html.escape(theme_name)}</h1>
<div class="subtitle">{html.escape(version_label)} — {html.escape(version_desc)}</div>
<div class="meta">
  Score de cohérence (auto, modèle DeBERTa) : {score_str}<br>
  Nombre de paragraphes : {len(items)} &middot; Sources distinctes :
  {len(set(it['source'] for it in items))}
</div>
{body}
</body>
</html>"""


def export_html_and_pdf(html_content, out_path_html):
    """Écrit le fichier HTML et tente une conversion PDF avec pdfkit/wkhtmltopdf."""
    with open(out_path_html, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"      💾 {out_path_html}")

    out_path_pdf = out_path_html.with_suffix('.pdf')

    wkhtmltopdf_path = CONFIG.get('WKHTMLTOPDF_PATH')
    wkhtmltopdf_found = wkhtmltopdf_path or shutil.which('wkhtmltopdf')

    if not PDFKIT_AVAILABLE:
        print(f"      ⚠️  pdfkit non installé — PDF non généré pour "
              f"{out_path_html.name}")
        print(f"         (pip install pdfkit, + installer wkhtmltopdf)")
        return

    if not wkhtmltopdf_found:
        print(f"      ⚠️  wkhtmltopdf introuvable — PDF non généré pour "
              f"{out_path_html.name}")
        print(f"         Installe-le depuis https://wkhtmltopdf.org/downloads.html "
              f"ou ouvre le HTML dans un navigateur puis 'Imprimer → PDF'.")
        return

    try:
        config = (pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)
                  if wkhtmltopdf_path else None)
        if config:
            pdfkit.from_string(html_content, str(out_path_pdf), configuration=config)
        else:
            pdfkit.from_string(html_content, str(out_path_pdf))
        print(f"      💾 {out_path_pdf}")
    except Exception as e:
        print(f"      ⚠️  Conversion PDF échouée pour {out_path_html.name} : {e}")
        print(f"         Le HTML reste disponible — ouvre-le dans un "
              f"navigateur puis 'Imprimer → PDF'.")


def generate_theme_documents(model, tokenizer, label_encoder, device):
    """Pour chaque thème de CONFIG['THEMES'], extrait les paragraphes de
    toutes les sources, génère Version A (mélange) et Version B (RST),
    et exporte chacune en HTML + PDF."""
    if not CONFIG['THEMES']:
        return

    themes_dir = Path(CONFIG['OUTPUT_DIR']) / CONFIG['THEMES_SUBDIR']
    themes_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📑 Génération des documents composites par thème...")

    for theme_name, sources in CONFIG['THEMES'].items():
        print(f"\n   [{theme_name}]")

        pool = []
        for pdf_filename, page_start, page_end in sources:
            pool.extend(extract_theme_source(
                pdf_filename, page_start, page_end, CONFIG['PDF_DIR'],
                CONFIG['MIN_WORDS'], CONFIG['MAX_WORDS'],
                CONFIG['THEME_MAX_PARAGRAPHS_PER_SOURCE']
            ))

        if len(pool) < CONFIG['THEME_MIN_TOTAL_PARAGRAPHS']:
            print(f"   ⚠️  {theme_name} : seulement {len(pool)} paragraphes "
                  f"au total (minimum {CONFIG['THEME_MIN_TOTAL_PARAGRAPHS']}) "
                  f"— thème ignoré.")
            continue

        # Version A : mélange aléatoire de tous les paragraphes (toutes sources)
        pool_shuffled = pool.copy()
        random.shuffle(pool_shuffled)
        score_A, rels_A = compute_theme_coherence(
            pool_shuffled, model, tokenizer, label_encoder, device)
        print(f"      Version A (mélange)  : cohérence = {score_A:.3f}")

        # Version B : réordonnancement RST sur le pool fusionné
        pool_rst, rst_seq = order_theme_items_by_rst(
            pool, model, tokenizer, label_encoder, device)
        score_B, rels_B = compute_theme_coherence(
            pool_rst, model, tokenizer, label_encoder, device)
        print(f"      Version B (RST)      : cohérence = {score_B:.3f} "
              f"({'↑' if score_B > score_A else '↓'} vs A "
              f"{abs(score_B-score_A):.3f})")

        safe_name = re.sub(r'[^a-zA-Z0-9_-]+', '_', theme_name)

        html_A = render_theme_html(
            theme_name, pool_shuffled, rels_A, score_A,
            "Version A", "paragraphes mélangés aléatoirement, multi-sources")
        export_html_and_pdf(html_A, themes_dir / f"{safe_name}_version_A.html")

        html_B = render_theme_html(
            theme_name, pool_rst, rels_B, score_B,
            "Version B", "réordonnancement RST, multi-sources")
        export_html_and_pdf(html_B, themes_dir / f"{safe_name}_version_B.html")

        # Sauvegarder aussi les données brutes (traçabilité)
        theme_json = {
            'theme': theme_name,
            'sources': [f"{f} (p. {s}-{e})" for f, s, e in sources],
            'version_A': {
                'order': 'random_multi_source',
                'coherence_score': round(score_A, 3),
                'items': pool_shuffled,
                'rst_relations': rels_A,
            },
            'version_B': {
                'order': 'rst_multi_source',
                'coherence_score': round(score_B, 3),
                'items': pool_rst,
                'rst_relations': rels_B,
            },
            'delta_B_vs_A': round(score_B - score_A, 3),
        }
        json_path = themes_dir / f"{safe_name}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(theme_json, f, ensure_ascii=False, indent=2)
        print(f"      💾 {json_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("""
============================================================
  GÉNÉRATION CORPUS ÉDUCATIF RST — OPENSTAX (PDF locaux, CC BY)
  Mode 1 (SECTIONS) : S0 / S1-SBERT / S2-RST + scores par section
                      Métriques : RST auto + SBERT coh. + GPT-2 perp.
  Mode 2 (THEMES)   : documents composites multi-sources → HTML/PDF
============================================================""")

    Path(CONFIG['OUTPUT_DIR']).mkdir(parents=True, exist_ok=True)

    # ── Déterminer la liste de sections à traiter ─────────────────────────────
    sections_to_process = []

    if CONFIG['AUTO_SCAN']:
        print(f"\n🔎 Mode automatique : scan de '{CONFIG['PDF_DIR']}/'...")
        sections_to_process = discover_sections(
            CONFIG['PDF_DIR'], CONFIG['MAX_SECTIONS_PER_PDF'])
        print(f"   ✅ {len(sections_to_process)} sections détectées au total "
              f"(tous PDF confondus)")
        if not sections_to_process:
            print(f"""
❌ Aucune section détectée.

   Vérifie que :
     - le dossier '{CONFIG['PDF_DIR']}/' existe et contient au moins un .pdf
     - les PDF sont lisibles par pdfplumber (pas des scans image sans texte)
""")
    else:
        # Normaliser au format à 5 éléments (heading_text=None pour le
        # mode manuel, puisque le titre n'a pas été détecté automatiquement)
        sections_to_process = [
            (name, pdf_filename, page_start, page_end, None)
            for name, pdf_filename, page_start, page_end in CONFIG['SECTIONS']
        ]
        if not sections_to_process:
            print(f"""
❌ CONFIG['AUTO_SCAN'] est False et CONFIG['SECTIONS'] est vide — rien à faire.

   Soit :
     - mets CONFIG['AUTO_SCAN'] = True pour traiter automatiquement tous
       les PDF de '{CONFIG['PDF_DIR']}/', ou
     - remplis CONFIG['SECTIONS'] manuellement :
       ('Nom de la section', 'fichier.pdf', page_debut, page_fin)
""")

    if not sections_to_process and not CONFIG['THEMES']:
        return

    # Charger le modèle une seule fois pour les deux modes
    if not Path(CONFIG['MODEL_PATH']).exists():
        print(f"\n❌ Modèle introuvable : {CONFIG['MODEL_PATH']}")
        print("   Lancez d'abord : python fine_tune_deberta_rst.py")
        return

    print(f"\n🤖 Chargement modèle RST...")
    model, tokenizer, label_encoder, device = load_model(CONFIG['MODEL_PATH'])
    print(f"   ✅ {model.config.num_labels} relations | device={device}")

    # Modèle Sentence-BERT pour S1 (baseline) + métriques indépendantes
    sbert_model = None
    if sections_to_process:
        print(f"\n🤖 Chargement modèle Sentence-BERT ({CONFIG['SBERT_MODEL_NAME']})"
              f" — baseline S1 + métriques indépendantes...")
        sbert_model = load_sbert_model(CONFIG['SBERT_MODEL_NAME'])
        if sbert_model is None:
            print("   ⚠️  sentence-transformers indisponible — S1 et SBERT coh. ignorés.")
        else:
            print("   ✅ Sentence-BERT chargé")

    # ── MODE 1 : ÉVALUATION PAR SECTION (Versions S0/S1/S2) ─────────────────
    # Lit ARTICLES_JSON et N_PILOT depuis CONFIG.
    # AUTO_SCAN=False + ARTICLES_JSON défini → skip extraction PDF complète.
    PILOT_JSON = CONFIG.get('ARTICLES_JSON', None)
    PILOT_N    = CONFIG.get('N_PILOT', None)

    # Charger SBERT si JSON disponible même sans sections_to_process
    if sbert_model is None and PILOT_JSON and Path(PILOT_JSON).exists():
        print(f"\n🤖 Chargement modèle Sentence-BERT ({CONFIG['SBERT_MODEL_NAME']})...")
        sbert_model = load_sbert_model(CONFIG['SBERT_MODEL_NAME'])
        if sbert_model is None:
            print("   ⚠️  sentence-transformers indisponible.")
        else:
            print("   ✅ Sentence-BERT chargé")

    no_json = not PILOT_JSON or not Path(PILOT_JSON).exists()
    if not sections_to_process and no_json:
        print("\n⏭️  Aucune section à traiter — mode évaluation par section ignoré.")
    else:
        run_sections_mode(model, tokenizer, label_encoder, device,
                          sections_to_process, sbert_model,
                          articles_json=PILOT_JSON,
                          n_pilot=PILOT_N)

    # ── MODE 2 : DOCUMENTS COMPOSITES PAR THÈME (HTML/PDF) ────────────────────
    if not CONFIG['THEMES']:
        print("\n⏭️  CONFIG['THEMES'] vide — génération de documents composites ignorée.")
    else:
        generate_theme_documents(model, tokenizer, label_encoder, device)


def run_sections_mode(model, tokenizer, label_encoder, device,
                       sections_to_process, sbert_model=None,
                       articles_json=None, n_pilot=None):
    """
    Mode 1 : pour chaque section, génère 3 versions et leurs scores.
    Conditions :
      S0 : ordre original OpenStax (référence humaine experte)
      S1 : réordonnancement SBERT (baseline indépendante)
      S2 : réordonnancement RST/DeBERTa via graphe complet (méthode proposée)

    Paramètres optionnels :
      articles_json : chemin vers un JSON pré-extrait (skip extraction PDF)
      n_pilot       : nombre de sections à traiter (None = toutes)
    """

    # ── 1. Charger articles (depuis JSON ou extraire depuis PDF) ──────────────
    if articles_json and Path(articles_json).exists():
        print(f"\n📂 Chargement depuis {articles_json} (skip extraction PDF)...")
        with open(articles_json, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        print(f"   ✓ {len(articles)} sections disponibles")

        if n_pilot and n_pilot < len(articles):
            # Sélection équilibrée par manuel
            from collections import defaultdict
            import random as _random
            by_pdf = defaultdict(list)
            for item in articles:
                by_pdf[item['pdf_file']].append(item)
            pdfs = sorted(by_pdf.keys())
            per_pdf = n_pilot // len(pdfs)
            remainder = n_pilot - per_pdf * len(pdfs)
            selected = []
            for i, pdf in enumerate(pdfs):
                n_pick = per_pdf + (1 if i < remainder else 0)
                pool = by_pdf[pdf]
                chosen = _random.sample(pool, min(n_pick, len(pool)))
                selected.extend(chosen)
                print(f"   {pdf:<50s}: {len(chosen)} sections")
            articles = selected
            print(f"   ✓ {len(articles)} sections sélectionnées pour le pilote")
    else:
        print(f"\n📚 Extraction du contenu ({len(sections_to_process)} sections)...")
        articles = []
        for name, pdf_filename, page_start, page_end, heading_text in sections_to_process:
            print(f"   → {name}...", end=' ')
            article = extract_section(
                name, pdf_filename, page_start, page_end,
                CONFIG['PDF_DIR'],
                CONFIG['MIN_WORDS'], CONFIG['MAX_WORDS'],
                CONFIG['MIN_PARAGRAPHS'], CONFIG['MAX_PARAGRAPHS'],
                heading_text=heading_text
            )
            if article:
                articles.append(article)
        print(f"\n   ✅ {len(articles)} sections exploitables")
        if not articles:
            print("\n❌ Aucune section exploitable.")
            return
        Path(CONFIG['OUTPUT_DIR']).mkdir(parents=True, exist_ok=True)
        raw_path = Path(CONFIG['OUTPUT_DIR']) / 'articles_raw.json'
        with open(raw_path, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        print(f"   💾 Articles bruts : {raw_path}")

    # Préchargement GPT-2 une seule fois
    if GPT2_AVAILABLE:
        _load_gpt2()
    else:
        print("   ⚠️  GPT-2 indisponible — perplexité ignorée.")

    # ── 2. Générer S0, S1 (SBERT), S2 (RST) ─────────────────────────────────
    print(f"\n🔀 Génération des trois conditions (S0 / S1-SBERT / S2-RST)...")
    eval_pairs = []

    for article in articles:
        topic = article['topic']
        paras = article['paragraphs']   # texte propre, liste de str
        print(f"\n   ── {topic} ({len(paras)} paragraphes) ──")

        # ── S0 : ordre original ──────────────────────────────────────────────
        rst_s0,   rels_s0   = compute_coherence_score(paras, model, tokenizer,
                                                       label_encoder, device)
        sbert_s0             = sbert_coherence_score(paras, sbert_model)
        perp_s0              = gpt2_perplexity(paras)
        print(f"   S0 original  | RST={rst_s0:.3f}"
              f" | SBERT={sbert_s0:.4f}" if sbert_s0 else
              f"   S0 original  | RST={rst_s0:.3f}")
        if perp_s0:
            print(f"                  GPT-2 perp={perp_s0:.1f}")

        # ── S1 : SBERT (baseline indépendante) ───────────────────────────────
        if sbert_model is not None:
            paras_s1, _ = order_paragraphs_by_similarity(paras, sbert_model)
        else:
            paras_s1 = list(paras)
        rst_s1,   rels_s1   = compute_coherence_score(paras_s1, model, tokenizer,
                                                       label_encoder, device)
        sbert_s1             = sbert_coherence_score(paras_s1, sbert_model)
        perp_s1              = gpt2_perplexity(paras_s1)
        print(f"   S1 SBERT     | RST={rst_s1:.3f}"
              + (f" | SBERT={sbert_s1:.4f}" if sbert_s1 else ""))
        if perp_s1:
            print(f"                  GPT-2 perp={perp_s1:.1f}")

        # ── S2 : RST discourse graph G=(V,E) + ablation α + beam search ────────
        profile    = CONFIG.get('LEARNER_PROFILE', 'novice')
        ALPHAS     = [1.0, 0.7, 0.5, 0.3]
        BEAM_WIDTHS = [1, 3, 5]   # beam=1 ≡ greedy (contrôle)

        print(f"   S2 RST graph | construction G=({len(paras)}V,"
              f"{len(paras)*(len(paras)-1)}E) profil={profile}...")
        graph, adjacency, sbert_matrix = build_rst_discourse_graph(
            paras, model, tokenizer, label_encoder, device,
            sbert_model=sbert_model)

        # ── Greedy α ablation ────────────────────────────────────────────────
        results_alpha = {}
        for alpha in ALPHAS:
            paras_a, trav_a, order_a = traverse_rst_graph(
                paras, graph, adjacency, profile=profile,
                alpha=alpha, sbert_sim=sbert_matrix)
            rst_a,  rels_a  = coherence_from_graph(order_a, graph)
            sbert_a          = sbert_coherence_score(paras_a, sbert_model)
            perp_a           = gpt2_perplexity(paras_a)
            results_alpha[alpha] = {
                'paras': paras_a, 'ordering': order_a,
                'traversal': trav_a,
                'rst': rst_a, 'sbert': sbert_a, 'perp': perp_a,
                'rels': rels_a,
            }
            marker = '✅' if rst_a > rst_s0 else '⚠️ '
            print(f"   Greedy [α={alpha:.1f}] | RST={rst_a:.3f}"
                  + (f" | SBERT={sbert_a:.4f}" if sbert_a else "")
                  + (f" | GPT2={perp_a:.1f}" if perp_a else "")
                  + f" {marker} | "
                  + ' → '.join(f'P{i+1}' for i in order_a))

        # ── Beam Search (RST pur α=1.0) ──────────────────────────────────────
        results_beam = {}
        for bw in BEAM_WIDTHS:
            paras_b, trav_b, order_b = traverse_beam_search(
                paras, graph, profile=profile,
                beam_width=bw, alpha=1.0)
            rst_b   = coherence_from_graph(order_b, graph)[0]
            sbert_b = sbert_coherence_score(paras_b, sbert_model)
            perp_b  = gpt2_perplexity(paras_b)
            results_beam[bw] = {
                'paras': paras_b, 'ordering': order_b,
                'traversal': trav_b,
                'rst': rst_b, 'sbert': sbert_b, 'perp': perp_b,
            }
            marker = '✅' if rst_b > rst_s0 else '⚠️ '
            label  = 'Greedy' if bw == 1 else f'Beam-{bw}'
            print(f"   {label:<12}  | RST={rst_b:.3f}"
                  + (f" | SBERT={sbert_b:.4f}" if sbert_b else "")
                  + (f" | GPT2={perp_b:.1f}" if perp_b else "")
                  + f" {marker} | "
                  + ' → '.join(f'P{i+1}' for i in order_b))

        # ── Beam Search + RE-CLASSEMENT GLOBAL (réponse au Reviewer #1) ────────
        # Sur le plus grand beam (k=5), on garde les k candidats finaux au
        # lieu du seul argmax, puis on les re-classe selon un critère
        # indépendant de DeBERTa (SBERT + option GPT-2), pas seulement selon
        # la moyenne des scores locaux (Eq. 6).
        candidates_5 = traverse_beam_search_candidates(
            paras, graph, profile=profile, beam_width=5, alpha=1.0)
        best_rerank, rerank_report = rerank_beam_candidates_global(
            paras, candidates_5, graph, sbert_model, gamma=0.6, use_gpt2=False)

        order_rr   = best_rerank['ordering']
        paras_rr   = [paras[i] for i in order_rr]
        rst_rr, _  = coherence_from_graph(order_rr, graph)
        sbert_rr   = sbert_coherence_score(paras_rr, sbert_model)
        perp_rr    = gpt2_perplexity(paras_rr)

        results_beam['5_rerank'] = {
            'paras': paras_rr, 'ordering': order_rr,
            'traversal': best_rerank['edges'],
            'rst': rst_rr, 'sbert': sbert_rr, 'perp': perp_rr,
            'rerank_report': rerank_report,
        }
        changed_marker = '🔀 changé' if rerank_report['changed_vs_beam_only'] else '= identique'
        print(f"   {'Beam5+Rerank':<12}  | RST={rst_rr:.3f}"
              + (f" | SBERT={sbert_rr:.4f}" if sbert_rr else "")
              + (f" | GPT2={perp_rr:.1f}" if perp_rr else "")
              + f" | vs. beam-only: {changed_marker} | "
              + ' → '.join(f'P{i+1}' for i in order_rr))

        # Référence : α=1.0 greedy pour S2 principal
        best      = results_alpha[1.0]
        paras_s2  = best['paras']
        ordering_s2 = best['ordering']
        traversal_info = best['traversal']
        rst_s2    = best['rst']
        sbert_s2  = best['sbert']
        perp_s2   = best['perp']
        rels_s2   = best['rels']

        rel_counts = {}
        for edge in graph.values():
            rel_counts[edge['relation']] = rel_counts.get(edge['relation'],0)+1
        top_rels = sorted(rel_counts, key=rel_counts.get, reverse=True)[:3]
        print(f"   top-rels: {top_rels}")

        d_rst   = rst_s2   - rst_s0
        d_sbert = (sbert_s2 - sbert_s0) if (sbert_s2 and sbert_s0) else None
        d_perp  = (perp_s2  - perp_s0)  if (perp_s2  and perp_s0)  else None
        print(f"   Δ Greedy(α=1.0)-S0 RST={d_rst:+.3f}"
              + (f" | SBERT={d_sbert:+.4f}" if d_sbert is not None else "")
              + (f" | GPT-2={d_perp:+.1f}"  if d_perp  is not None else ""))

        common_meta = {
            'topic':    topic,
            'source':   article['source'],
            'pdf_file': article['pdf_file'],
            'pages':    article['pages'],
        }

        graph_json = {f"{i},{j}": v for (i, j), v in graph.items()}

        # Sérialiser ablation α
        alpha_results_json = {
            str(a): {
                'ordering':    [i+1 for i in r['ordering']],
                'score_rst':   round(r['rst'],   4),
                'score_sbert': round(r['sbert'], 4) if r['sbert'] else None,
                'score_perp':  round(r['perp'],  1) if r['perp']  else None,
                'delta_rst_vs_s0':   round(r['rst']  - rst_s0, 4),
                'delta_sbert_vs_s0': round(r['sbert']- sbert_s0, 4)
                                     if (r['sbert'] and sbert_s0) else None,
            }
            for a, r in results_alpha.items()
        }

        # Sérialiser beam search (inclut '5_rerank' avec son rerank_report)
        beam_results_json = {
            str(bw): {
                'ordering':    [i+1 for i in r['ordering']],
                'score_rst':   round(r['rst'],   4),
                'score_sbert': round(r['sbert'], 4) if r['sbert'] else None,
                'score_perp':  round(r['perp'],  1) if r['perp']  else None,
                'delta_rst_vs_s0':   round(r['rst']  - rst_s0, 4),
                'delta_sbert_vs_s0': round(r['sbert']- sbert_s0, 4)
                                     if (r['sbert'] and sbert_s0) else None,
                **({'rerank_report': r['rerank_report']} if 'rerank_report' in r else {}),
            }
            for bw, r in results_beam.items()
        }

        eval_pairs.append({
            **common_meta,
            # Paragraphes — texte brut propre
            'version_S0': paras,
            'version_S1': paras_s1,
            'version_S2': paras_s2,   # α=1.0 (RST pur, référence)
            # Ordre de traversée S2 α=1.0
            'ordering_S2':   [i + 1 for i in ordering_s2],
            'profile_S2':    profile,
            # Graphe RST complet G=(V,E) avec similarités SBERT
            'rst_graph':     graph_json,
            'graph_n_nodes': len(paras),
            'graph_n_edges': len(graph),
            # Scores RST (DeBERTa — circulaire pour S2)
            'score_S0_rst':  round(rst_s0, 4),
            'score_S1_rst':  round(rst_s1, 4),
            'score_S2_rst':  round(rst_s2, 4),
            # Scores SBERT cohérence (indépendant ✅)
            'score_S0_sbert': sbert_s0,
            'score_S1_sbert': sbert_s1,
            'score_S2_sbert': sbert_s2,
            # Perplexité GPT-2 (indépendant ✅)
            'score_S0_perp':  perp_s0,
            'score_S1_perp':  perp_s1,
            'score_S2_perp':  perp_s2,
            # Deltas S2(α=1.0) vs S0
            'delta_S2_S0_rst':   round(rst_s2  - rst_s0,  4),
            'delta_S2_S1_rst':   round(rst_s2  - rst_s1,  4),
            'delta_S2_S0_sbert': round(sbert_s2 - sbert_s0, 4) if (sbert_s2 and sbert_s0) else None,
            'delta_S2_S1_sbert': round(sbert_s2 - sbert_s1, 4) if (sbert_s2 and sbert_s1) else None,
            'delta_S2_S0_perp':  round(perp_s2  - perp_s0,  3) if (perp_s2  and perp_s0)  else None,
            'delta_S2_S1_perp':  round(perp_s2  - perp_s1,  3) if (perp_s2  and perp_s1)  else None,
            # Ablation study α (greedy hybride)
            'alpha_ablation':    alpha_results_json,
            # Beam search (RST pur α=1.0, beam=1/3/5)
            'beam_search':       beam_results_json,
            # Relations RST sur paires adjacentes
            'rst_relations_S0':  rels_s0,
            'rst_relations_S1':  rels_s1,
            'rst_relations_S2':  rels_s2,
            'traversal_info_S2': traversal_info,
        })

    # ── 3. Sauvegarder ───────────────────────────────────────────────────────
    out = Path(CONFIG['OUTPUT_DIR']) / 'evaluation_pairs.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(eval_pairs, f, ensure_ascii=False, indent=2)
    print(f"\n   💾 {out}")

    # ── 4. Résumé ─────────────────────────────────────────────────────────────
    n = len(eval_pairs)
    def mean(key):
        vals = [e[key] for e in eval_pairs if e.get(key) is not None]
        return np.mean(vals) if vals else None

    rst0  = mean('score_S0_rst');   rst1  = mean('score_S1_rst');   rst2  = mean('score_S2_rst')
    sb0   = mean('score_S0_sbert'); sb1   = mean('score_S1_sbert'); sb2   = mean('score_S2_sbert')
    pp0   = mean('score_S0_perp');  pp1   = mean('score_S1_perp');  pp2   = mean('score_S2_perp')

    print(f"""
{'='*72}
  📊 RÉSUMÉ — {n} sections
{'='*72}
  Métrique         S0 (original)  S1 (SBERT)   Greedy(α=1.0)
  ─────────────────────────────────────────────────────────────
  RST auto (↑)     {rst0:.4f}       {rst1:.4f}      {rst2:.4f}
  SBERT coh. (↑)   {f'{sb0:.4f}' if sb0 else 'N/A':12s}   {f'{sb1:.4f}' if sb1 else 'N/A':10s}   {f'{sb2:.4f}' if sb2 else 'N/A'}
  GPT-2 perp. (↓)  {f'{pp0:.1f}' if pp0 else 'N/A':12s}   {f'{pp1:.1f}' if pp1 else 'N/A':10s}   {f'{pp2:.1f}' if pp2 else 'N/A'}
{'='*72}
  ABLATION — Greedy hybride RST+SBERT (α)
  Score(i,j) = α×conf_RST + (1-α)×sim_SBERT
{'='*72}
  {'α':<6} {'RST':>8} {'Δ RST':>8} {'SBERT':>8} {'Δ SBERT':>9} {'GPT-2':>7} {'Δ GPT2':>8}""")

    for alpha_str in ['1.0', '0.7', '0.5', '0.3']:
        rst_a  = np.mean([e['alpha_ablation'][alpha_str]['score_rst']
                          for e in eval_pairs if 'alpha_ablation' in e])
        vals_s = [e['alpha_ablation'][alpha_str]['score_sbert']
                  for e in eval_pairs
                  if e.get('alpha_ablation',{}).get(alpha_str,{}).get('score_sbert')]
        vals_p = [e['alpha_ablation'][alpha_str]['score_perp']
                  for e in eval_pairs
                  if e.get('alpha_ablation',{}).get(alpha_str,{}).get('score_perp')]
        sb_a = np.mean(vals_s) if vals_s else 0
        pp_a = np.mean(vals_p) if vals_p else 0
        dr = rst_a - rst0; ds = sb_a - sb0; dp = pp_a - pp0
        rst_ok = '✅' if rst_a > rst0 else '⚠️ '
        sb_ok  = '✅' if sb_a  > sb0  else '❌'
        gpt_ok = '✅' if pp_a  < pp0  else '❌'
        print(f"  {alpha_str:<6} {rst_a:>8.4f} {dr:>+8.4f} {sb_a:>8.4f} {ds:>+9.4f} "
              f"{pp_a:>7.1f} {dp:>+8.1f}  {rst_ok}{sb_ok}{gpt_ok}")

    print(f"""
{'='*72}
  BEAM SEARCH (RST pur α=1.0) — beam_width ∈ {{1, 3, 5}}
  beam=1 ≡ greedy (contrôle)
{'='*72}
  {'beam':<8} {'RST':>8} {'Δ RST':>8} {'SBERT':>8} {'Δ SBERT':>9} {'GPT-2':>7} {'Δ GPT2':>8}""")

    for bw_str in ['1', '3', '5', '5_rerank']:
        rst_b  = np.mean([e['beam_search'][bw_str]['score_rst']
                          for e in eval_pairs if 'beam_search' in e])
        vals_s = [e['beam_search'][bw_str]['score_sbert']
                  for e in eval_pairs
                  if e.get('beam_search',{}).get(bw_str,{}).get('score_sbert')]
        vals_p = [e['beam_search'][bw_str]['score_perp']
                  for e in eval_pairs
                  if e.get('beam_search',{}).get(bw_str,{}).get('score_perp')]
        sb_b = np.mean(vals_s) if vals_s else 0
        pp_b = np.mean(vals_p) if vals_p else 0
        dr = rst_b - rst0; ds = sb_b - sb0; dp = pp_b - pp0
        rst_ok = '✅' if rst_b > rst0 else '⚠️ '
        sb_ok  = '✅' if sb_b  > sb0  else '❌'
        gpt_ok = '✅' if pp_b  < pp0  else '❌'
        label  = 'beam=5+rerank' if bw_str == '5_rerank' else f"beam={bw_str}"
        print(f"  {label:<13} {rst_b:>8.4f} {dr:>+8.4f} {sb_b:>8.4f} {ds:>+9.4f} "
              f"{pp_b:>7.1f} {dp:>+8.1f}  {rst_ok}{sb_ok}{gpt_ok}")

    n_changed = sum(
        1 for e in eval_pairs
        if e.get('beam_search', {}).get('5_rerank', {})
             .get('rerank_report', {}).get('changed_vs_beam_only')
    )
    print(f"""
  ─────────────────────────────────────────────────────────────────────
  Si beam=3 ou 5 améliore RST et/ou SBERT vs beam=1 (greedy) :
  → le problème est la stratégie de traversée, pas le classifieur.
  → beam search est défendable comme amélioration dans §4.5 et §5.

  RE-CLASSEMENT GLOBAL (beam=5+rerank, gamma=0.6, réponse Reviewer #1) :
  → sélection changée par rapport au choix beam-only sur {n_changed}/{n} sections.
  → si SBERT (et/ou GPT-2) s'améliore ici sans effondrement de RST :
    preuve empirique que le score local moyen (Eq. 6) laissait de la
    marge, et qu'un critère global la récupère au moins partiellement.
{'='*72}
""")


if __name__ == "__main__":
    main()
