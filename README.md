# — LeatherGuard : Détection et segmentation de défauts sur cuir


Modular computer-vision pipeline that checks a leather photo for wrinkle defects, localises them, measures their size and proposes an indicative quality grade. Built with PyTorch, Ultralytics YOLO, segmentation-models-pytorch and Streamlit.

## 1. Contexte et objectif


1. **Module 2 — Classification** : détermine si une pièce de cuir est normale ou présente un défaut (pli).
2. **Module 3 — Détection** : localise précisément chaque défaut via une boîte orientée (OBB).
3. **Module 4 — Segmentation** : trace le contour exact du défaut pour en mesurer la surface.
4. **Module 3b — Quality Grading** : attribue un grade (A/B/C) et une décision (Accept/Rework/Reject) selon le pourcentage de surface touchée.

---

## 2. Module 2 — Classification (normal vs défaut)

**Méthodologie** : comparaison de 3 architectures (EfficientNetV2-S, ResNet50, MobileNetV3-Large) avec pondération de classes et sélection automatique par score composite (0.5×recall + 0.3×f1 + 0.2×fps).

**Résultat final** (test, 76 images : 30 normales, 46 défectueuses, seuil de décision = 0.65) :

| Classe | Precision | Recall | F1-score |
|---|---|---|---|
| Normal | 0.91 | 0.97 | 0.94 |
| Wrinkle (défaut) | 0.98 | 0.93 | 0.96 |
| **Accuracy globale** | | | **0.95** |

**Statut** : module terminé, fiable, aucune limite majeure identifiée.

---

## 3. Module 3 — Détection (localisation par boîte orientée)

**Méthodologie** : YOLO11-OBB, exploration systématique de 4 leviers (architecture, résolution, données, augmentation).

### Chronologie des résultats

| Étape | Configuration | mAP50 | Recall | Precision |
|---|---|---|---|---|
| 0 | Point de départ (yolo11n, imgsz=640, CPU, 60 ep, 300 img) | 0.367 | 0.311 | — |
| 1 | GPU activé (même config) | 0.367 | 0.311 | — (6× plus rapide) |
| 2 | Architecture n→s (150 ep) | 0.362 | 0.338 | — |
| 3 | Architecture s→m (non retenu) | 0.423 | 0.378 | — |
| 4 | **Résolution 640→960px** | **0.465** | 0.347 | 0.623 |
| 5 | Résolution 960→1280px (non retenu, compromis) | 0.441 | 0.493 | 0.467 |
| 6 | **Dataset enrichi +142 images (300→442)** | **0.523** | **0.556** | 0.466 |
| 7 | mixup/copy_paste (non retenu, résultat inférieur) | 0.312 | 0.333 | 0.673 |

**Résultat final retenu** : YOLO11s-OBB, imgsz=960, dataset de 442 images.
- **mAP50 = 0.523 | mAP50-95 = 0.241 | Recall = 0.556 | Precision = 0.466**
- Mesuré sur 68 images test, 99 instances de défauts.

**Ajustement du seuil de confiance** (compromis recall/precision) :
| Seuil | Recall | Precision |
|---|---|---|
| 0.15 | 0.573 | 0.473 |
| 0.25 (défaut) | 0.400 | 0.634 |

**Progression totale** : Recall +79% relatif (0.311 → 0.556) depuis le point de départ.

**Statut** : module finalisé. Le levier le plus efficace a été le volume/diversité des données (+21% de mAP50, +61% de recall), devant l'architecture et la résolution.

---

## 4. Module 4 — Segmentation (contour précis du défaut)

C'est le module ayant nécessité le plus d'itérations, en raison d'un défaut à très faible contraste visuel.

### 4.1 Diagnostic initial

Masques générés automatiquement en remplissant les boîtes de détection (`fillPoly`) :

| Architecture | Dice (masques fillPoly) |
|---|---|
| U-Net++ | 0.433 |
| SegFormer | 0.321 |
| YOLO11-Seg | ~0.08 (Mask mAP50) |

**Cause identifiée** : un pli occupe seulement 10-20% de sa boîte englobante ; remplir toute la boîte crée une fausse vérité terrain (rectangle plein au lieu d'un contour fin). Confirmé par comparaison visuelle et par test contrôlé (changer uniquement le masque a suffi à faire progresser le score).

### 4.2 Tentatives d'automatisation (échouées)

SAM 2 (point-prompt, box-prompt), CLAHE, étirement de contraste global : aucune n'a permis une segmentation automatique fiable, le contraste des plis étant trop faible même pour un modèle de pointe.

### 4.3 Cycle d'annotation manuelle (Label Studio)

| Lot | Total cumulé | Dice obtenu |
|---|---|---|
| Lot 1 (37 images) | 37 | 0.538 |
| Lot 2 (+45, sans augmentation) | 80 | 0.548 (instable) |
| + Augmentation de données | 80 | 0.583 |
| Lot 3 (+50) | 130 | (split instable, non comparable) |
| **Split figé + test dédié (25 images)** | 130 | **0.523** (1ère mesure fiable) |
| + Normalisation ImageNet | 130 | 0.555 |
| + Contraste étiré (en plus) | 130 | 0.597 (val 0.670) |
| + Test-Time Augmentation (TTA) | 130 | 0.636 |
| Après consolidation complète (6+ lots, réparation des masques orphelins/mal nommés, split final propre) | ~200 masques fusionnés | **0.603** (mesure finale définitive) |

### 4.4 Résultat final retenu

**U-Net++**, contraste étiré + normalisation ImageNet + BCE/Dice loss + TTA :

| Métrique | Valeur |
|---|---|
| Dice | **0.603** |
| IoU | 0.445 |
| Pixel Accuracy | 0.913 |
| Precision | 0.595 |
| Recall | 0.677 |
| Vitesse (FPS) | 314.6 |

Comparaison SegFormer sur les mêmes vrais masques : **Dice = 0.436** (non retenu).

**Progression totale** : +39% relatif depuis le point de départ (0.433 → 0.603).

**Statut** : module finalisé. Résultat honnête pour un défaut à contraste intrinsèquement très faible — limite documentée, pas un échec de méthode.

---

## 5. Module 3b — Quality Grading

**Méthodologie** : mesure la surface de défaut segmentée par rapport à la surface totale de l'image, attribue un grade selon des seuils métier.

**Résultat sur 46 images test** (seuils ajustés à 5%/10% pour tenir compte de la precision du modèle) :

| Grade | Décision | Nombre | % |
|---|---|---|---|
| A | Accept | 25 | 54.3% |
| B | Rework | 14 | 30.4% |
| C | Reject | 7 | 15.2% |

**Statistiques des défauts mesurés** : surface moyenne touchée = 7.8%, médiane = 4.7%.

**Limite documentée** : avec une precision de segmentation de 0.595, une partie des grades B/C peut refléter des faux positifs modestes plutôt que de vrais défauts étendus — recommandation de supervision humaine sur les décisions proches des seuils.

---

## 6. Synthèse chiffrée globale

| Module | Métrique principale | Résultat final | Progression depuis le point de départ |
|---|---|---|---|
| Classification | F1-score | 0.94–0.96 | — (résultat direct) |
| Détection | Recall / mAP50 | 0.556 / 0.523 | +79% / +43% |
| Segmentation | Dice | 0.603 | +39% |
| Quality Grading | Taux d'acceptation | 54.3% | (calibré après ajustement des seuils) |

**Volume de données final** :
- Détection : 442 images annotées (boîtes OBB)
- Segmentation : ~200 masques manuels fusionnés sur 6+ lots d'annotation Label Studio

---

## 7. Contributions méthodologiques clés

1. **Diagnostic rigoureux** du problème de masques synthétiques (`fillPoly`) par élimination méthodique (comparaison multi-architectures → inspection visuelle → test contrôlé).
2. **Découverte et exploitation de l'étirement de contraste local** pour rendre un défaut invisible en brut annotable manuellement.
3. **Cycle itératif d'annotation manuelle avec validation rigoureuse** (split figé, test set dédié, anti-fuite train/test).
4. **Identification et correction de plusieurs bugs de pipeline** (double-normalisation de coordonnées, dossiers mal nommés, désynchronisation de conventions de nommage entre versions de scripts) — documentés comme retour d'expérience méthodologique.

---

## 8. Limites et perspectives (résumé)

- **Détection** : recall imparfait (55.6%) limité par le volume de données ; petits défauts sous-représentés.
- **Segmentation** : plafonnée par le contraste intrinsèquement faible du défaut — un changement du dispositif de capture (éclairage rasant) est identifié comme le levier le plus prometteur, non testé faute de nouvelles photos.
- **Perspectives** : boucle d'amélioration continue en production, module d'anomaly detection non supervisé complémentaire, standardisation des conventions de nommage pour fiabiliser le pipeline technique.
**Project structure**
app.py                      Streamlit UI
leatherguard/classifier.py  classifier: architecture auto-detection, loading, prediction
leatherguard/pipeline.py    detection, crops, U-Net++, measurements, overlay
leatherguard/grading.py     severity grid
leatherguard/inference.py   image loading (EXIF-aware), decision logic
leatherguard/weights.py     identifies a .pt file by content, without executing it
scripts/inspect_weights.py  organise and sanity-check the three weight files
tests/ 