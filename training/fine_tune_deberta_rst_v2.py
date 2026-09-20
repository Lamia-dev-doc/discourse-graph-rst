"""
fine_tune_deberta_rst.py

"""

import os
import json
import torch
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, accuracy_score,
    f1_score, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'DATA_FILE':    'data/gum_scidtb_cleaned.json',
    'OUTPUT_DIR':   'deberta_rst_model_v2',

    # DeBERTa-v3-base : meilleur que BERT pour les tâches discursives
    # (Kobayashi 2022 — déjà dans votre related work)
    'MODEL_NAME': 'microsoft/deberta-v3-base',

    # ✅ 128 au lieu de 256 → économise ~40% VRAM (GTX 1650 4GB)
    'MAX_LENGTH':   128,

    'BATCH_SIZE':   4,        # ✅ réduit 8→4 pour GTX 1650 4GB
    'GRAD_ACCUM':   8,        # ✅ compensé : batch effectif = 4×8 = 32
    'EPOCHS':       10,
    'PATIENCE':     3,        # Early stopping sur F1-macro
    'LEARNING_RATE': 2e-5,   # LR plus faible pour DeBERTa
    'WARMUP_RATIO': 0.1,     # 10% des steps en warmup
    'WEIGHT_DECAY': 0.01,

    'TRAIN_SIZE':   0.8,
    'VAL_SIZE':     0.1,
    'TEST_SIZE':    0.1,

    'SEED':         42,
    'DEVICE':       'cuda' if torch.cuda.is_available() else 'cpu'
}

# ============================================================================
# SEED GLOBAL — REPRODUCTIBILITÉ
# ============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CONFIG['SEED'])

print(f"""
{'='*70}
  FINE-TUNING DeBERTa-base — RST DISCOURSE RELATION CLASSIFICATION
  Corpus : GUM + SciDTB fusionnés
{'='*70}

Configuration :
  • Modèle      : {CONFIG['MODEL_NAME']}
  • Données     : {CONFIG['DATA_FILE']}
  • MAX_LENGTH  : {CONFIG['MAX_LENGTH']}
  • Device      : {CONFIG['DEVICE']}
  • Seed        : {CONFIG['SEED']}
  • Epochs      : {CONFIG['EPOCHS']} (early stopping patience={CONFIG['PATIENCE']})
  • Batch size  : {CONFIG['BATCH_SIZE']} × grad_accum={CONFIG['GRAD_ACCUM']} = {CONFIG['BATCH_SIZE']*CONFIG['GRAD_ACCUM']} effectif
  • LR          : {CONFIG['LEARNING_RATE']}
""")

# ============================================================================
# DATASET — tokenizer correct : edu1 + edu2 séparément
# ============================================================================

class RSTDataset(Dataset):
    """
    Paires EDU pour classification RST.
    Utilise tokenizer(edu1, edu2) — HuggingFace gère automatiquement :
    [CLS] edu1 [SEP] edu2 [SEP]
    """
    def __init__(self, pairs, labels, tokenizer, max_length):
        self.pairs   = pairs    # liste de (edu1, edu2)
        self.labels  = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        edu1, edu2 = self.pairs[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            edu1, edu2,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        item = {
            'input_ids':      encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label':          torch.tensor(label, dtype=torch.long)
        }

        # token_type_ids optionnels (DeBERTa n'en a pas besoin)
        if 'token_type_ids' in encoding:
            item['token_type_ids'] = encoding['token_type_ids'].flatten()

        return item

# ============================================================================
# CHARGEMENT DONNÉES
# ============================================================================

def load_corpus(file_path):
    print(f"\n📚 Chargement du corpus...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"   ✓ {len(data):,} paires EDU chargées")

    pairs  = [(item['edu1'], item['edu2']) for item in data]
    labels = [item['relation'] for item in data]

    return pairs, labels

# ============================================================================
# ENTRAÎNEMENT avec gradient accumulation
# ============================================================================

def train_epoch(model, dataloader, optimizer, scheduler,
                loss_fn, device, grad_accum):
    model.train()
    total_loss   = 0
    predictions  = []
    true_labels  = []

    optimizer.zero_grad()
    progress_bar = tqdm(dataloader, desc="Training")

    for step, batch in enumerate(progress_bar):
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['label'].to(device)

        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        if 'token_type_ids' in batch:
            kwargs['token_type_ids'] = batch['token_type_ids'].to(device)

        outputs = model(**kwargs)
        loss = loss_fn(outputs.logits, labels) / grad_accum

        loss.backward()

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum
        preds = torch.argmax(outputs.logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

        progress_bar.set_postfix({'loss': f"{loss.item()*grad_accum:.4f}"})

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    f1_m     = f1_score(true_labels, predictions, average='macro',
                        zero_division=0)
    return avg_loss, accuracy, f1_m

# ============================================================================
# ÉVALUATION
# ============================================================================

def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    predictions = []
    true_labels = []
    total_loss  = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['label'].to(device)

            kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
            if 'token_type_ids' in batch:
                kwargs['token_type_ids'] = batch['token_type_ids'].to(device)

            outputs = model(**kwargs)
            loss    = loss_fn(outputs.logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    avg_loss   = total_loss / len(dataloader)
    accuracy   = accuracy_score(true_labels, predictions)
    f1_weighted = f1_score(true_labels, predictions, average='weighted',
                            zero_division=0)
    f1_macro   = f1_score(true_labels, predictions, average='macro',
                           zero_division=0)

    return predictions, true_labels, avg_loss, accuracy, f1_weighted, f1_macro

# ============================================================================
# MATRICE DE CONFUSION
# ============================================================================

def save_confusion_matrix(true_labels, predictions, class_names, output_dir):
    cm = confusion_matrix(true_labels, predictions)

    # Normaliser par ligne (recall par classe)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig, axes = plt.subplots(1, 2, figsize=(28, 12))

    # Counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=axes[0])
    axes[0].set_title('Confusion Matrix (counts)', fontsize=13)
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].tick_params(axis='y', rotation=0)

    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=axes[1])
    axes[1].set_title('Confusion Matrix (normalized recall)', fontsize=13)
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].tick_params(axis='y', rotation=0)

    plt.tight_layout()
    path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Matrice de confusion : {path}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    # ✅ Vider le cache GPU avant de commencer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"   🧹 Cache GPU vidé")
        free = torch.cuda.get_device_properties(0).total_memory
        print(f"   💾 VRAM totale : {free/1024**3:.1f} GB")

    # ── 1. Charger données ───────────────────────────────────────────────────
    pairs, labels = load_corpus(CONFIG['DATA_FILE'])

    # ── 2. Encoder labels ────────────────────────────────────────────────────
    print("\n🔤 Encodage des relations...")
    label_encoder  = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_labels     = len(label_encoder.classes_)
    print(f"   ✓ {num_labels} relations : {list(label_encoder.classes_)}")

    # ── 3. Split stratifié ───────────────────────────────────────────────────
    print("\n📊 Split train / val / test (stratifié)...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        pairs, encoded_labels,
        test_size=CONFIG['VAL_SIZE'] + CONFIG['TEST_SIZE'],
        random_state=CONFIG['SEED'],
        stratify=encoded_labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=CONFIG['TEST_SIZE'] / (CONFIG['VAL_SIZE'] + CONFIG['TEST_SIZE']),
        random_state=CONFIG['SEED'],
        stratify=y_temp
    )
    print(f"   ✓ Train : {len(X_train):,} | Val : {len(X_val):,} | Test : {len(X_test):,}")

    # ── 4. Weighted Loss (déséquilibre de classes) ───────────────────────────
    print("\n⚖️  Calcul des poids de classes (weighted cross-entropy + boost rare)...")
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )

    # ── Boost des relations rares pédagogiquement critiques ──────────────────
    # Ces relations gouvernent la structure des textes éducatifs mais sont
    # sous-représentées dans GUM+SciDTB (< 50 exemples chacune).
    # Un boost de 2.0-3.0× améliore leur recall sans dégrader les classes
    # fréquentes (elaboration-*, attribution-positive restent dominantes).
    from collections import Counter
    train_counts = Counter(y_train)

    RARE_BOOST = {
        # Relations très rares (< 30 exemples) — boost fort
        'adversative-antithesis':    3.0,   # 20 ex. — opposition logique
        'explanation-justify':       3.0,   # 23 ex. — justification pédagogique
        'restatement-partial':       2.5,   # 28 ex. — résumé/reformulation
        # Relations rares (30-50 exemples) — boost modéré
        'causal-cause':              2.0,   # 38 ex. — causalité
        'causal-result':             2.0,   # 32 ex. — résultat
        'explanation-evidence':      2.0,   # 32 ex. — preuve
        'organization-preparation':  1.8,   # 32 ex. — introduction de liste
        'temporal-sequence':         1.8,   # 41 ex. — progression temporelle
        'contingency-condition':     1.5,   # 50 ex. — condition
    }

    boosted = []
    for i, rel in enumerate(label_encoder.classes_):
        boost = RARE_BOOST.get(rel, 1.0)
        class_weights[i] *= boost
        if boost > 1.0:
            boosted.append((rel, class_weights[i], boost))

    class_weights_tensor = torch.FloatTensor(class_weights).to(CONFIG['DEVICE'])
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)

    print(f"\n   Relations boostées :")
    for rel, w, boost in sorted(boosted, key=lambda x: -x[2]):
        n_ex = train_counts.get(
            list(label_encoder.classes_).index(rel) if rel in label_encoder.classes_ else -1, 0)
        print(f"   {rel:35s} : poids = {w:.3f}  (×{boost})")

    print(f"\n   Top 5 poids finaux :")
    for rel, w in sorted(zip(label_encoder.classes_, class_weights),
                          key=lambda x: x[1], reverse=True)[:5]:
        print(f"   {rel:35s} : poids = {w:.3f}")

    # ── 5. Tokenizer DeBERTa ──────────────────────────────────────────────────
    print(f"\n🔧 Chargement tokenizer {CONFIG['MODEL_NAME']}...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['MODEL_NAME'])
    print(f"   ✓ Vocabulaire : {tokenizer.vocab_size:,} tokens")

    # ── 6. Datasets & DataLoaders ─────────────────────────────────────────────
    train_dataset = RSTDataset(X_train, y_train, tokenizer, CONFIG['MAX_LENGTH'])
    val_dataset   = RSTDataset(X_val,   y_val,   tokenizer, CONFIG['MAX_LENGTH'])
    test_dataset  = RSTDataset(X_test,  y_test,  tokenizer, CONFIG['MAX_LENGTH'])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'],
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=CONFIG['BATCH_SIZE'],
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=CONFIG['BATCH_SIZE'],
                              shuffle=False, num_workers=0)

    # ── 7. Modèle DeBERTa ────────────────────────────────────────────────────
    print(f"\n🤖 Chargement de {CONFIG['MODEL_NAME']}...")
    id2label = {i: lbl for i, lbl in enumerate(label_encoder.classes_)}
    label2id = {lbl: i for i, lbl in enumerate(label_encoder.classes_)}

    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG['MODEL_NAME'],
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    model.to(CONFIG['DEVICE'])

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Paramètres totaux : {total_params/1e6:.1f}M")

    # ── 8. Optimizer & Scheduler ──────────────────────────────────────────────
    optimizer   = AdamW(model.parameters(),
                        lr=CONFIG['LEARNING_RATE'],
                        weight_decay=CONFIG['WEIGHT_DECAY'])
    total_steps = (len(train_loader) // CONFIG['GRAD_ACCUM']) * CONFIG['EPOCHS']
    warmup_steps = int(total_steps * CONFIG['WARMUP_RATIO'])
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    print(f"   ✓ Total steps : {total_steps} | Warmup : {warmup_steps}")

    # ── 9. Entraînement ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  🚀 ENTRAÎNEMENT — sauvegarde sur F1-macro (patience={CONFIG['PATIENCE']})")
    print(f"{'='*70}\n")

    os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)
    best_f1_macro  = 0.0
    patience_count = 0
    history        = []
    start_epoch    = 0

    # ✅ REPRISE DEPUIS CHECKPOINT (si PC a redémarré)
    checkpoint_path = os.path.join(CONFIG['OUTPUT_DIR'], 'checkpoint.pkl')
    if os.path.exists(checkpoint_path):
        print(f"\n🔄 Checkpoint trouvé — reprise de l'entraînement...")
        with open(checkpoint_path, 'rb') as f:
            ckpt = pickle.load(f)
        start_epoch    = ckpt['epoch']
        best_f1_macro  = ckpt['best_f1_macro']
        patience_count = ckpt['patience_count']
        history        = ckpt['history']
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        # Recharger le meilleur modèle sauvegardé
        model = AutoModelForSequenceClassification.from_pretrained(
            CONFIG['OUTPUT_DIR'])
        model.to(CONFIG['DEVICE'])
        print(f"   ✓ Reprise à l'epoch {start_epoch + 1} | "
              f"Meilleur F1-M : {best_f1_macro:.4f}")
    else:
        print(f"\n🚀 Démarrage fresh — aucun checkpoint trouvé")

    for epoch in range(start_epoch, CONFIG['EPOCHS']):
        print(f"\n📅 Epoch {epoch + 1}/{CONFIG['EPOCHS']}")
        print("-" * 70)

        # Train
        train_loss, train_acc, train_f1m = train_epoch(
            model, train_loader, optimizer, scheduler,
            loss_fn, CONFIG['DEVICE'], CONFIG['GRAD_ACCUM']
        )
        print(f"   Train → Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1-M: {train_f1m:.4f}")

        # Validation
        _, _, val_loss, val_acc, val_f1w, val_f1m = evaluate(
            model, val_loader, loss_fn, CONFIG['DEVICE']
        )
        print(f"   Val   → Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1-W: {val_f1w:.4f} | F1-M: {val_f1m:.4f}")

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss, 'train_acc': train_acc,
            'val_loss': val_loss, 'val_acc': val_acc,
            'val_f1w': val_f1w, 'val_f1m': val_f1m
        })

        # ── Sauvegarde sur F1-MACRO (pas accuracy) ──
        if val_f1m > best_f1_macro:
            best_f1_macro  = val_f1m
            patience_count = 0
            model.save_pretrained(CONFIG['OUTPUT_DIR'])
            tokenizer.save_pretrained(CONFIG['OUTPUT_DIR'])
            with open(os.path.join(CONFIG['OUTPUT_DIR'], 'label_encoder.pkl'), 'wb') as f:
                pickle.dump(label_encoder, f)
            print(f"   ⭐ Meilleur modèle sauvegardé ! F1-M = {val_f1m:.4f}")
        else:
            patience_count += 1
            print(f"   ⏳ Pas d'amélioration ({patience_count}/{CONFIG['PATIENCE']})")
            if patience_count >= CONFIG['PATIENCE']:
                print(f"\n🛑 Early stopping à l'époque {epoch + 1}")
                break

        # ✅ SAUVEGARDE CHECKPOINT après chaque epoch
        # → permet de reprendre si le PC redémarre
        ckpt = {
            'epoch':         epoch + 1,
            'best_f1_macro': best_f1_macro,
            'patience_count': patience_count,
            'history':       history,
            'optimizer':     optimizer.state_dict(),
            'scheduler':     scheduler.state_dict(),
        }
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(ckpt, f)
        print(f"   💾 Checkpoint sauvegardé (epoch {epoch + 1})")

    # ── 10. Évaluation finale sur Test ────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  📊 ÉVALUATION FINALE SUR TEST SET")
    print(f"{'='*70}\n")

    # ✅ Supprimer le checkpoint — entraînement terminé normalement
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("   🗑️  Checkpoint supprimé (entraînement terminé)\n")

    # Recharger meilleur modèle
    print("🔄 Rechargement du meilleur modèle...")
    model = AutoModelForSequenceClassification.from_pretrained(CONFIG['OUTPUT_DIR'])
    model.to(CONFIG['DEVICE'])

    preds, true, _, test_acc, test_f1w, test_f1m = evaluate(
        model, test_loader, loss_fn, CONFIG['DEVICE']
    )

    print(f"\n🎯 RÉSULTATS FINAUX:")
    print(f"   • Accuracy   : {test_acc:.4f}  ({test_acc*100:.2f}%)")
    print(f"   • F1-W       : {test_f1w:.4f}")
    print(f"   • F1-Macro   : {test_f1m:.4f}")

    # Rapport détaillé
    print(f"\n📋 RAPPORT PAR RELATION:\n")
    report = classification_report(
        true, preds,
        target_names=label_encoder.classes_,
        digits=4,
        zero_division=0
    )
    print(report)

    # ── 11. Matrice de confusion ──────────────────────────────────────────────
    print("\n📊 Génération de la matrice de confusion...")
    save_confusion_matrix(true, preds, label_encoder.classes_, CONFIG['OUTPUT_DIR'])

    # ── 12. Courbes d'apprentissage ───────────────────────────────────────────
    print("\n📈 Génération des courbes d'apprentissage...")
    epochs_range = [h['epoch'] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs_range, [h['train_loss'] for h in history], 'b-o', label='Train')
    axes[0].plot(epochs_range, [h['val_loss']   for h in history], 'r-o', label='Val')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_range, [h['train_acc'] for h in history], 'b-o', label='Train Acc')
    axes[1].plot(epochs_range, [h['val_acc']   for h in history], 'r-o', label='Val Acc')
    axes[1].plot(epochs_range, [h['val_f1m']   for h in history], 'g-s', label='Val F1-M')
    axes[1].set_title('Accuracy & F1-Macro')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f'DeBERTa-RST Training — {num_labels} relations', fontsize=13)
    plt.tight_layout()
    curve_path = os.path.join(CONFIG['OUTPUT_DIR'], 'training_curves.png')
    plt.savefig(curve_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Courbes : {curve_path}")

    # ── 13. Sauvegarder rapport complet ──────────────────────────────────────
    report_path = os.path.join(CONFIG['OUTPUT_DIR'], 'evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"DeBERTa-RST — Corpus GUM+SciDTB Fusionné\n")
        f.write(f"Modèle : {CONFIG['MODEL_NAME']}\n")
        f.write(f"Seed   : {CONFIG['SEED']}\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"Accuracy     : {test_acc:.4f}  ({test_acc*100:.2f}%)\n")
        f.write(f"F1-Weighted  : {test_f1w:.4f}\n")
        f.write(f"F1-Macro     : {test_f1m:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nHistorique d'entraînement:\n")
        for h in history:
            f.write(f"  Epoch {h['epoch']}: train_loss={h['train_loss']:.4f} "
                    f"val_acc={h['val_acc']:.4f} val_f1m={h['val_f1m']:.4f}\n")

    print(f"\n✅ Rapport    : {report_path}")
    print(f"✅ Modèle     : {CONFIG['OUTPUT_DIR']}/")

    print(f"\n{'='*70}")
    print(f"  🎉 TERMINÉ — Meilleur F1-Macro val : {best_f1_macro:.4f}")
    print(f"{'='*70}\n")

    # ── 14. Comparaison v1 → v2 ──────────────────────────────────────────────
    print("📊 COMPARAISON DeBERTa-v1 → DeBERTa-v2 (poids boostés) :")
    print(f"   DeBERTa-v1 (balanced)  : Accuracy=69.68%  F1-W=0.6951  F1-M=0.6505")
    print(f"   DeBERTa-v2 (boosted)   : Accuracy={test_acc*100:.2f}%  F1-W={test_f1w:.4f}  F1-M={test_f1m:.4f}")
    delta_acc = test_acc*100 - 69.68
    delta_f1m = test_f1m - 0.6505
    print(f"   Δ Accuracy : {delta_acc:+.2f} pp")
    print(f"   Δ F1-Macro : {delta_f1m:+.4f}")
    print(f"\n   Relations rares à surveiller :")
    print(f"   explanation-justify, adversative-antithesis, restatement-partial")

if __name__ == "__main__":
    main()
