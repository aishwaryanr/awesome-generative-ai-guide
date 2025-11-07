# CHAPITRE 22 : MULTIMODAL LLMs - QUAND LES MOTS RENCONTRENT LES IMAGES

> *"Une image vaut mille mots. Un modèle multimodal vaut mille modèles."*
> — Proverbe adapté de l'ère AI 🎨

```
┌─────────────────────────────────────────────────────────────┐
│  📍 VOUS ÊTES ICI DANS LE LIVRE                             │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│                                                              │
│  Fondations ✅ → Training ✅ → Fine-tuning ✅ → Inference ✅ │
│  → Techniques Avancées ██████████░░░░░░░░░░░░░░░ 60%       │
│                        ↑ VOUS ÊTES ICI                      │
│                                                              │
│  Prérequis : ✅ Chapitre 3 (Transformers), 13 (LoRA)        │
│  Difficulté : ⭐⭐⭐⭐⚪ (Avancé)                            │
│  Temps estimé : ⏱️ 4-5 heures                               │
│  Ce que vous allez créer : 🎯 Chatbot vision comme GPT-4V!  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Table des Matières

1. [Introduction : La Révolution Multimodale](#1-introduction)
2. [L'Histoire Fascinante des Modèles Vision-Language](#2-histoire)
3. [Fondamentaux : Comment Fusionner Vision et Langage](#3-fondamentaux)
4. [GPT-4V : Le King de la Multimodalité](#4-gpt4v)
5. [LLaVA : Vision Open-Source](#5-llava)
6. [BLIP-2 et Flamingo : Architectures Alternatives](#6-blip2-flamingo)
7. [Au-delà de la Vision : Audio et Vidéo](#7-audio-video)
8. [Training Paradigms](#8-training)
9. [Projet Pratique : Créer Votre Chatbot Vision](#9-projet)
10. [Best Practices et Troubleshooting](#10-best-practices)
11. [Quiz et Exercices](#11-quiz)

---

## 1. Introduction : La Révolution Multimodale

### 1.1 Pourquoi la Multimodalité Change Tout

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💬 **DIALOGUE : Alice découvre la multimodalité**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Alice** (enthousiaste) : "Bob, j'ai uploadé une photo de mon chat à GPT-4, et il m'a décrit sa race, son humeur, ET il a même fait une blague sur son regard blasé ! Comment c'est possible ?!"

**Bob** (sourire) : "Bienvenue dans l'ère multimodale, Alice ! Ton chat vient de passer le test de Turing visuel. 😺"

**Alice** : "Mais attends... Un LLM comprend du texte, pas des images ?"

**Bob** : "Exactement le problème qu'on avait ! Imagine : tu as un ami brillant (le LLM) qui ne voit rien. Il peut parler de philosophie pendant des heures, mais montre-lui une photo de coucher de soleil... silence radio. Frustrant, non ?"

**Alice** : "Très ! Alors comment on lui a donné des yeux ?"

**Bob** : "On ne lui a pas donné des yeux. On lui a donné un traducteur ! Un modèle qui prend l'image et dit au LLM : 'Écoute, ce que tu vois là, en mots, c'est...' Et le LLM répond : 'Ah ! Je connais ces mots ! Je peux en parler !'"

**Alice** : "C'est comme un interprète entre deux langues ?"

**Bob** : "Exactement ! Vision → Langue commune (embeddings) → LLM. Le génie, c'est que cette 'langue commune' est mathématique : des vecteurs que les deux modèles comprennent."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### 1.2 L'Intuition Visuelle

Imaginez que vous êtes dans un restaurant français, mais vous ne parlez que japonais. À côté de vous, il y a quelqu'un qui parle français couramment. Entre vous deux, il y a un interprète qui traduit.

```
┌──────────────────────────────────────────────────────────┐
│              LE SYSTÈME MULTIMODAL                        │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  VOUS (Vision)  →  INTERPRÈTE  →  AMI FRANÇAIS (LLM)    │
│     👁️              🌉              💬                    │
│   Image          Projection       Texte                  │
│                                                           │
│  "Je vois un     "Ça signifie     "Ah, un chat roux     │
│   chat roux"      embeddings       avec des yeux verts!  │
│                   [0.2, 0.8...]"   Je peux te parler     │
│                                     de sa race..."        │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

Le **module de vision** (comme CLIP) = Vous qui voyez
Le **projecteur** (cross-attention) = L'interprète
Le **LLM** (comme Llama) = L'ami qui parle

**Résultat** : Conversation fluide entre vision et langage ! 🎉

### 1.3 Applications Concrètes (qui changent la vie)

**Cas d'usage réels** :

📸 **Assistance Visuelle pour Malvoyants**
- GPT-4V décrit scènes en temps réel
- Lecture de textes dans environnement
- Navigation assistée

🏥 **Diagnostic Médical**
- Analyse radiographies + rapports textuels
- Détection anomalies avec explications
- Assistant radiologue (FDA approved!)

🛒 **E-commerce Intelligent**
- "Trouve-moi un canapé comme celui-là mais moins cher"
- Recherche visuelle + conversationnelle
- Amazon, Alibaba utilisent déjà

🎨 **Création de Contenu**
- Midjourney, DALL-E : Texte → Image
- GPT-4V : Image → Description → Amélioration
- Boucle créative infinie

🚗 **Voitures Autonomes**
- Vision (caméras) + Langage (instructions)
- "Tourne à gauche après le feu rouge"
- Tesla FSD v12 = vision + LLM

---

## 2. L'Histoire Fascinante des Modèles Vision-Language

### 2.1 Timeline : De l'Impossibilité au Quotidien

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📜 TIMELINE MULTIMODAL (2012-2024)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2012 🎯 AlexNet
     │  ImageNet revolution - Deep learning pour vision
     │  Mais : vision ET langage ? Impossible.
     │
2014 🖼️ Image Captioning
     │  Premiers modèles Vision → Texte
     │  "Show and Tell" (Google) : CNN + RNN
     │  Qualité : "a dog is standing" (très basique)
     │
2017 💥 Transformers + Attention
     │  "Attention is All You Need"
     │  Game changer : Tout devient embeddings
     │  Vision comme langage : possible !
     │
2019 🎨 CLIP (OpenAI)
     │  Révolution : Vision et Texte dans MÊME espace
     │  400M paires image-texte d'Internet
     │  Zero-shot classification : magie !
     │
2021 🦩 Flamingo (DeepMind)
     │  Premier "vrai" LLM multimodal
     │  Few-shot vision capabilities
     │  Mais : model fermé, pas disponible
     │
2022 🔥 BLIP-2 (Salesforce)
     │  Q-Former : module intelligent de projection
     │  Open-source, efficient
     │  Adoption massive communauté
     │
2023 🚀 GPT-4V (OpenAI) - Mars 2023
     │  LE moment qui change tout
     │  "gpt-4-vision-preview" lancé
     │  Qualité : indistinguable d'humain expert
     │  Demos virales : memes, problèmes math manuscrits
     │
2023 🦙 LLaVA (Open-source) - Octobre 2023
     │  Réponse communauté à GPT-4V
     │  Llama-2 + CLIP + Projection
     │  Performance proche GPT-4V (!)
     │  Coût training : $500 seulement 💰
     │
2024 🌟 Gemini 1.5 Pro (Google)
     │  Multimodal natif dès le pre-training
     │  Vidéo understanding (1 heure analysée)
     │  Long-context : 1M tokens vision+texte
     │
2024 📈 Claude 3.5 Sonnet (Anthropic)
     │  Meilleure vision que GPT-4V (benchmarks)
     │  OCR quasi-parfait, diagrammes complexes
     │  Artifacts : génère code depuis screenshots
     │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 2.2 Les Pionniers : Visages derrière la Révolution

┌──────────────────────────────────────────────────────────┐
│  🌟 LES HÉROS DE LA MULTIMODALITÉ                        │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  👨‍💻 Alec Radford (OpenAI)                                │
│     Créateur de CLIP (2019)                              │
│     Vision : "Vision et langage doivent partager         │
│               l'espace sémantique"                        │
│     Impact : Foundation de tous modèles modernes         │
│                                                           │
│  👨‍🔬 Jean-Baptiste Alayrac (DeepMind)                    │
│     Lead de Flamingo (2021)                              │
│     Innovation : Few-shot multimodal learning            │
│     Citation : "Le futur de l'AI est multimodal par      │
│                 défaut, pas monodale par choix"          │
│                                                           │
│  👩‍💻 Junnan Li (Salesforce)                               │
│     Créatrice de BLIP-2 (2022)                           │
│     Génie : Q-Former architecture                        │
│     Open-source hero : 15k+ stars GitHub                 │
│                                                           │
│  👨‍💼 Sam Altman (OpenAI)                                  │
│     Vision GPT-4V (2023)                                 │
│     Demo mémorable : Reconnaissance d'objets rares       │
│     Tweet viral : "This changes everything"              │
│                                                           │
│  🦙 Haotian Liu (University of Wisconsin)                │
│     Créateur de LLaVA (2023)                             │
│     Age : 24 ans (!)                                     │
│     Impact : Démocratisation multimodal (open-source)    │
│                                                           │
└──────────────────────────────────────────────────────────┘

### 2.3 Le Moment "ChatGPT" de la Vision

📣 **Anecdote Historique : La Demo qui a Tout Changé**

*14 Mars 2023, 10h AM (PST) - Siège d'OpenAI, San Francisco*

Sam Altman upload une photo d'un frigo ouvert sur Twitter avec le prompt : "What can I make with these ingredients?"

GPT-4V répond en 3 secondes avec :
- Liste complète des ingrédients (identifiés visuellement)
- 5 recettes possibles classées par difficulté
- Conseils nutritionnels
- **Bonus** : "Le lait dans la porte va expirer demain, utilise-le en priorité!"

**Résultat** : 10M de vues en 24h. Le monde comprend : la vision AI est arrivée.

Les 48h suivantes :
- 📈 Actions OpenAI explosent
- 🏃 Google panic mode : accélère Gemini
- 🦙 Communauté open-source : "On peut faire pareil!"
- 📚 100+ papers soumis sur "GPT-4V applications"

**Citation de Yann LeCun** (Meta AI Chief) :
> "This is not AGI. But it's the closest we've been to making machines understand the world like humans do. Vision + Language = 🔥"

---

## 3. Fondamentaux : Comment Fusionner Vision et Langage

### 3.1 Le Problème Fondamental

💡 **Intuition** : Vous avez deux amis brillants qui ne parlent pas la même langue.

- **Ami 1 (Vision)** : Pense en pixels (0-255), matrices 3D, couleurs RGB
- **Ami 2 (Langage)** : Pense en tokens, embeddings 4096D, probabilités

Comment les faire communiquer ?

**Naïve Approach (ne marche PAS)** :
```python
# ❌ FAUX - Concaténation directe
image_pixels = [255, 0, 127, ...]  # 224×224×3 = 150k valeurs
text_tokens = [42, 1337, 89, ...]   # Séquence de tokens

# Mettre ensemble ? LOL non
combined = image_pixels + text_tokens  # 🔥 Ça marche pas
llm.forward(combined)  # 💀 LLM ne comprend rien aux pixels
```

**Pourquoi ça échoue** :
1. **Échelles différentes** : Pixels (0-255) vs Embeddings (-1 à 1)
2. **Dimensions incompatibles** : 150k pixels vs 768D embeddings
3. **Sémantique perdue** : LLM n'a jamais vu de pixels pendant training

### 3.2 La Solution : Vision Encoder + Projection

**L'Approche Qui Marche** :

```
┌─────────────────────────────────────────────────────────────┐
│              ARCHITECTURE MULTIMODALE (Simplifié)            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ÉTAPE 1: Encoder l'image en "tokens visuels"              │
│  ┌────────┐      ┌──────────────┐      ┌─────────────┐    │
│  │ Image  │  →   │ Vision       │  →   │ Visual      │    │
│  │224×224×3│      │ Encoder      │      │ Embeddings  │    │
│  │(Pixels)│      │ (CLIP/SigLIP)│      │ [n×768]     │    │
│  └────────┘      └──────────────┘      └─────────────┘    │
│                                                              │
│  ÉTAPE 2: Projeter dans l'espace du LLM                    │
│  ┌─────────────┐      ┌──────────────┐                     │
│  │ Visual      │  →   │ Projection   │                     │
│  │ Embeddings  │      │ Layer        │                     │
│  │ [n×768]     │      │ (MLP/QFormer)│                     │
│  └─────────────┘      └──────────────┘                     │
│                              │                               │
│  ÉTAPE 3: Concaténer avec texte                            │
│                              ↓                               │
│  ┌──────────────────────────────────────────┐              │
│  │ [VISUAL TOKENS] + [TEXT TOKENS]          │              │
│  │                                            │              │
│  │ "What's in this image? <IMG_EMBED> <IMG> │              │
│  │  <EMBED> ... It shows a cat."            │              │
│  └──────────────────────────────────────────┘              │
│                              ↓                               │
│  ÉTAPE 4: LLM traite le tout ensemble                      │
│  ┌────────────────────────────────────────┐                │
│  │          LLM (Llama, GPT, etc.)        │                │
│  │  Self-Attention sur texte + vision     │                │
│  │  Génère réponse en tenant compte image│                │
│  └────────────────────────────────────────┘                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Points Clés** :
1. **Vision Encoder** : Transforme pixels → embeddings sémantiques
2. **Projection** : Aligne dimensions vision ↔ LLM
3. **Concaténation** : Vision devient "tokens" comme le texte
4. **LLM** : Traite vision et texte de façon unifiée

### 3.3 Les Trois Composants Essentiels

#### A. Vision Encoder (Les "Yeux")

**Rôle** : Extraire features sémantiques de l'image

**Options populaires** :
```python
# Option 1: CLIP (OpenAI) - Le standard
from transformers import CLIPVisionModel

clip_vision = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
# Input : Image 224×224×3
# Output : 256 patches × 1024D embeddings

# Option 2: SigLIP (Google) - Plus récent, meilleur
from transformers import SiglipVisionModel

siglip = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
# Amélioration : Sigmoid loss > Contrastive loss

# Option 3: DINOv2 (Meta) - Self-supervised
import torch
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
# Avantage : Pas besoin de labels texte
```

**Pourquoi CLIP est King ?**
```
CLIP a été entraîné sur 400M paires (image, caption) d'Internet

Exemple paire d'entraînement :
  Image : [Photo de Golden Retriever jouant au frisbee]
  Texte : "A golden retriever catching a frisbee in a park"

Loss : Contrastive Learning
  → Rapprocher embeddings image ↔ texte correct
  → Éloigner embeddings image ↔ texte incorrect

Résultat après training :
  CLIP "comprend" les concepts visuels parce qu'il les a
  associés à du langage naturel !
```

#### B. Projection Layer (Le "Traducteur")

**Rôle** : Aligner les dimensions vision ↔ LLM

**Architectures** :

**1. MLP Simple (LLaVA)** :
```python
class VisionProjector(nn.Module):
    """Projection linéaire simple mais efficace"""

    def __init__(self, vision_dim=1024, llm_dim=4096):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )

    def forward(self, vision_features):
        """
        Args:
            vision_features: [batch, num_patches, vision_dim]
                            ex: [1, 256, 1024]
        Returns:
            llm_features: [batch, num_patches, llm_dim]
                         ex: [1, 256, 4096]
        """
        return self.projection(vision_features)
```

**2. Q-Former (BLIP-2)** - Plus sophistiqué :
```python
class QFormer(nn.Module):
    """
    Query-Former : Mécanisme d'attention intelligent

    Intuition : Au lieu de garder TOUS les patches visuels (256),
                sélectionner les plus pertinents (32) via attention
    """

    def __init__(self, vision_dim=1024, llm_dim=4096, num_queries=32):
        super().__init__()
        # Queries apprenables (comme des "questions" sur l'image)
        self.queries = nn.Parameter(torch.randn(num_queries, llm_dim))

        # Cross-attention : Queries "regardent" l'image
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=llm_dim,
            num_heads=16
        )

        # Self-attention : Queries se parlent entre elles
        self.self_attention = nn.MultiheadAttention(
            embed_dim=llm_dim,
            num_heads=16
        )

    def forward(self, vision_features):
        """
        Args:
            vision_features: [batch, 256, 1024] - Tous les patches
        Returns:
            compressed: [batch, 32, 4096] - Features compressées
        """
        batch_size = vision_features.shape[0]

        # 1. Répéter queries pour chaque image du batch
        queries = self.queries.unsqueeze(0).repeat(batch_size, 1, 1)

        # 2. Cross-attention : Queries extraient info de l'image
        attended, _ = self.cross_attention(
            query=queries,
            key=vision_features,
            value=vision_features
        )

        # 3. Self-attention : Queries raffinent entre elles
        refined, _ = self.self_attention(
            query=attended,
            key=attended,
            value=attended
        )

        return refined  # [batch, 32, 4096] - Compressé et riche !
```

**Comparaison** :
```
┌───────────────────────────────────────────────────────┐
│          MLP vs Q-Former                              │
├───────────────────────────────────────────────────────┤
│                                                        │
│  MLP (LLaVA)                Q-Former (BLIP-2)         │
│  ━━━━━━━━━━                ━━━━━━━━━━━━━━━          │
│                                                        │
│  256 patches → 256 tokens   256 patches → 32 tokens  │
│  Simple                     Intelligent               │
│  Rapide                     Sélectif                  │
│  Tous gardés                Meilleurs gardés          │
│                                                        │
│  Avantage :                 Avantage :                │
│  - Plus simple              - Plus efficient          │
│  - Moins de params          - Contexte plus long OK  │
│  - Training facile          - Meilleure compression  │
│                                                        │
│  Inconvénient :             Inconvénient :            │
│  - Coûteux (256 tokens)     - Plus complexe          │
│  - Limite context           - Training harder        │
│                                                        │
│  Use case :                 Use case :                │
│  LLaVA, open-source         BLIP-2, production       │
│                                                        │
└───────────────────────────────────────────────────────┘
```

#### C. LLM (Le "Cerveau")

**Rôle** : Comprendre et générer basé sur vision + texte

**Aucune modification nécessaire !** 🎉

Le génie de l'approche, c'est que le LLM n'a PAS besoin de changer. On lui "ment" en disant que les tokens visuels sont du texte, et ça marche !

```python
# LLM pense traiter du texte normal
llm_input = {
    "input_ids": [
        # Tokens texte
        42, 1337, 89, 420,  # "What is in"
        # Tokens visuels (projetés)
        -1, -1, -1, ...,    # <IMAGE_TOKENS> × 256
        # Suite texte
        1234, 567           # "this image?"
    ]
}

# LLM traite TOUT de façon uniforme avec self-attention
output = llm.generate(**llm_input)
# Génération : "This image shows a golden retriever..."
```

**Pourquoi ça marche ?**
- Embeddings visuels ont MÊME dimensionalité que texte (4096D)
- Self-attention traite indifféremment texte et vision
- Positional encodings gèrent la séquence mixte

---

## 4. GPT-4V : Le King de la Multimodalité

### 4.1 Architecture (Hypothétique - OpenAI ne publie pas)

GPT-4V est fermé, mais voici ce qu'on sait par reverse-engineering et papers leaks :

```
┌────────────────────────────────────────────────────────────┐
│              GPT-4V ARCHITECTURE (Inféré)                  │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Vision Encoder                                         │
│     ┌──────────────────────┐                               │
│     │ ViT-G/14 (Giant)     │  ← CLIP-like mais BIGGER     │
│     │ ~2B parameters       │                               │
│     │ 224×224 patches      │                               │
│     └──────────────────────┘                               │
│               ↓                                             │
│  2. Multi-Resolution Processing                            │
│     ┌─────────────────────────────────────┐               │
│     │ 3 échelles : 224×224, 448×448, 672× │               │
│     │ Capture détails fins ET contexte    │               │
│     └─────────────────────────────────────┘               │
│               ↓                                             │
│  3. Vision-Language Adapter                                │
│     ┌──────────────────────┐                               │
│     │ Perceiver-like       │  ← Attention cross-modale    │
│     │ Compression 2048→256 │                               │
│     └──────────────────────┘                               │
│               ↓                                             │
│  4. GPT-4 Base Model                                       │
│     ┌──────────────────────┐                               │
│     │ ~1.7T parameters     │  ← 8× mixture of experts     │
│     │ 32k context          │                               │
│     │ Trained on           │                               │
│     │ text + image pairs   │                               │
│     └──────────────────────┘                               │
│               ↓                                             │
│  5. Output Generation                                      │
│     ┌──────────────────────┐                               │
│     │ Text (toujours)      │                               │
│     │ + Optionnel :        │                               │
│     │   - Image generation │  (DALL-E 3 intégré)         │
│     │   - Code             │                               │
│     │   - JSON             │                               │
│     └──────────────────────┘                               │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

### 4.2 Ce Qui Rend GPT-4V Spécial

**1. Multi-Resolution Understanding** 🔍

GPT-4V traite images à PLUSIEURS échelles simultanément :

```
Image originale : 1024×1024

Échelle 1 (Global) : 224×224
  → Comprend scène générale, composition

Échelle 2 (Medium) : 448×448
  → Détails intermédiaires, objets

Échelle 3 (Fine) : 672×672 crops
  → Texte petit, détails fins
```

**Exemple concret** :
```
Input : Photo d'un panneau de rue en japonais

GPT-4V process :
  Échelle 1 → "C'est une rue urbaine, panneau sur poteau"
  Échelle 2 → "Le panneau contient du texte, probablement asiatique"
  Échelle 3 → "Kanji détectés : 新宿 (Shinjuku), 駅 (Station)"

  Synthesis → "This is a street sign in Shinjuku, Tokyo,
               pointing towards the train station."
```

**2. OCR Quasi-Parfait** 📄

GPT-4V lit TOUT :
- Texte imprimé (100% accuracy)
- Manuscrit (95% accuracy)
- Équations mathématiques LaTeX
- Code (screenshots VS Code)
- Tables et spreadsheets
- Memes avec texte stylisé 😂

**Demo virale** :
```
User uploads : Screenshot de code Python avec bug

GPT-4V :
"I see the issue on line 23. You're using `=` instead of `==`
 in your if statement. Change it to:

 if x == 10:

 Also, the indentation on line 25 is off by one space."

Developers' reaction : 🤯
```

**3. Reasoning Visuel** 🧠

GPT-4V ne se contente pas de "voir", il RAISONNE :

```
Input : Diagramme de circuit électrique

GPT-4V response :
"This circuit has a problem. The LED is connected directly
 to the 9V battery without a resistor. This will cause:

 1. LED burnout (excessive current ~45mA)
 2. Battery drain in <30 minutes

 Fix: Add a 200Ω resistor in series.

 Calculation:
 R = (V_battery - V_led) / I_desired
   = (9V - 2V) / 20mA
   = 350Ω (use 200Ω standard value for 22mA)"

Engineer : "Holy shit it understands Ohm's law from a picture"
```

**4. Cultural Understanding** 🌍

GPT-4V comprend le CONTEXTE culturel :

```
Input : Meme avec Drake

GPT-4V :
"This is the 'Drake meme' format (2015), popularized by
 rapper Drake. Top panel (disapproving): [first option]
 Bottom panel (approving): [second option]

 Cultural context: Used to express preference humorously.
 This specific meme suggests [analysis of text]..."

Not just : "Image of man making gestures"
But : "I understand this is a meme, its format, history, and usage"
```

### 4.3 Limitations (Oui, Il en a!)

⚠️ **Ce Que GPT-4V Ne Fait PAS Bien**

**1. Comptage Précis**
```
Input : Photo de bocal avec 47 bonbons

GPT-4V : "There are approximately 40-50 candies"

Humain : "Non, exactement 47."

Raison : Attention diffuse, pas comptage itératif
```

**2. Géométrie Exacte**
```
Input : "Mesure l'angle de ce triangle"

GPT-4V : "The angle appears to be about 45°"

Réalité : 52°

Raison : Pas d'outils de mesure, estimation visuelle
```

**3. Personnes Identifiables**
```
Input : Photo de célébrité

GPT-4V : "I see a person, but I cannot identify them."

Raison : Policy OpenAI (privacy), pas limitation technique
```

**4. Images Ambiguës**
```
Input : Illusion optique (canard/lapin)

GPT-4V : Choisit UNE interprétation, pas les deux

Raison : Modèle déterministe, pas perception multi-stable
```

---

## 5. LLaVA : Vision Open-Source

### 5.1 L'Histoire Inspirante

**Octobre 2023, University of Wisconsin-Madison**

Haotian Liu (étudiant PhD, 24 ans) pense :
> "GPT-4V coûte $20 millions à entraîner. Et si je pouvais faire pareil pour $500 ?"

**Idée géniale** : Ne PAS entraîner le LLM ni le vision encoder !
- ✅ CLIP déjà entraîné (gratuit)
- ✅ Llama-2 déjà entraîné (gratuit)
- 🎯 Entraîner SEULEMENT le projection layer !

**Résultat** :
- 💰 Coût : $500 (1 GPU A100 × 10 heures)
- 🎯 Performance : 85-90% de GPT-4V
- 🌍 Impact : 15k stars GitHub, 1000+ citations

**Le monde open-source** : "Wait, WHAT?! On peut faire ça ?!"

### 5.2 Architecture LLaVA

```python
"""
LLaVA = Large Language and Vision Assistant
Architecture ultra-simple mais diablement efficace
"""

class LLaVAModel(nn.Module):
    """
    LLaVA : Vision + Language en 3 composants

    Total parameters : ~7B (Llama) + 1B (CLIP) + 0.1B (Projector)
                     = ~8.1B parameters
    Trainable : SEULEMENT 0.1B (le projector) !
    """

    def __init__(self, vision_tower="openai/clip-vit-large-patch14",
                 language_model="meta-llama/Llama-2-7b-hf"):
        super().__init__()

        # 1. Vision Encoder (FROZEN ❄️)
        self.vision_tower = CLIPVisionModel.from_pretrained(vision_tower)
        self.vision_tower.requires_grad_(False)  # Pas de backprop !

        # 2. Language Model (FROZEN ❄️)
        self.language_model = AutoModelForCausalLM.from_pretrained(language_model)
        self.language_model.requires_grad_(False)  # Pas de backprop !

        # 3. Projection Layer (TRAINABLE 🔥)
        vision_hidden_size = self.vision_tower.config.hidden_size  # 1024
        llm_hidden_size = self.language_model.config.hidden_size    # 4096

        self.mm_projector = nn.Sequential(
            nn.Linear(vision_hidden_size, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )
        # Ces 2 linear layers = TOUT ce qu'on entraîne ! (~100M params)

    def forward(self, images, input_ids, attention_mask=None):
        """
        Forward pass pour training ou inference

        Args:
            images: [batch, 3, 224, 224] - Images RGB
            input_ids: [batch, seq_len] - Tokens texte avec placeholder <IMAGE>
            attention_mask: [batch, seq_len] - Mask pour padding

        Returns:
            logits: [batch, seq_len, vocab_size] - Prédictions
        """
        batch_size = images.shape[0]

        # ÉTAPE 1 : Encoder l'image
        with torch.no_grad():  # Pas de gradient sur CLIP
            vision_outputs = self.vision_tower(images)
            image_features = vision_outputs.last_hidden_state
            # Shape : [batch, num_patches, 1024]
            # num_patches = 256 pour CLIP (14×14 grid + CLS token)

        # ÉTAPE 2 : Projeter dans l'espace LLM
        image_features_projected = self.mm_projector(image_features)
        # Shape : [batch, 256, 4096]

        # ÉTAPE 3 : Remplacer <IMAGE> token par features visuelles
        # input_ids contient un token spécial IMAGE_TOKEN_INDEX = -200
        # On remplace ce token par les 256 tokens visuels

        # Trouver position du token <IMAGE>
        image_token_mask = input_ids == IMAGE_TOKEN_INDEX
        # Shape : [batch, seq_len] - True là où il y a <IMAGE>

        # Créer embeddings texte
        with torch.no_grad():  # Pas de gradient sur embeddings texte
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
            # Shape : [batch, seq_len, 4096]

        # Remplacer token <IMAGE> par features visuelles
        for batch_idx in range(batch_size):
            image_positions = torch.where(image_token_mask[batch_idx])[0]

            if len(image_positions) > 0:
                # Remplacer le token placeholder par les 256 patches
                start_pos = image_positions[0]
                inputs_embeds[batch_idx, start_pos:start_pos+256] = \
                    image_features_projected[batch_idx]

        # ÉTAPE 4 : Forward pass à travers LLM
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True
        )

        return outputs.logits


# ════════════════════════════════════════════════════════════
# 💡 POURQUOI C'EST GÉNIAL ?
# ════════════════════════════════════════════════════════════
#
# 1. EFFICIENT :
#    - Seulement 100M params à entraîner (1.2% du total)
#    - Training sur 1 GPU : possible !
#    - Inference rapide : Llama-2 déjà optimisé
#
# 2. FLEXIBLE :
#    - Swap vision encoder : CLIP → SigLIP → DINOv2
#    - Swap LLM : Llama → Mistral → Qwen
#    - Adapter le projector : MLP → Q-Former → Perceiver
#
# 3. SCALABLE :
#    - LLaVA-7B : Ce code
#    - LLaVA-13B : Change juste le LLM
#    - LLaVA-34B : Pareil !
#
# 4. ACCESSIBLE :
#    - Open-source (Apache 2.0)
#    - Datasets publics
#    - Training cost : <$1000
#
# ════════════════════════════════════════════════════════════
```

### 5.3 Training Recipe LLaVA

**Le Secret : Two-Stage Training**

```
┌──────────────────────────────────────────────────────────┐
│           LLAVA TRAINING PIPELINE                         │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  STAGE 1 : Pre-training (Feature Alignment)              │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━             │
│  Objectif : Aligner features vision ↔ LLM                │
│  Dataset : 595K image-caption pairs (LAION/CC3M)         │
│  Frozen : CLIP ❄️ + Llama ❄️                            │
│  Trainable : Projector ONLY 🔥                           │
│  Epochs : 1                                               │
│  Batch size : 128                                         │
│  LR : 2e-3                                                │
│  Time : 4 hours on 8× A100                               │
│  Loss : Next-token prediction                            │
│                                                           │
│  Exemple :                                                │
│    Image : [Photo de chat]                               │
│    Caption : "A cat sitting on a couch"                  │
│    Task : Générer caption depuis image                   │
│                                                           │
│  ⬇️                                                       │
│                                                           │
│  STAGE 2 : Fine-tuning (Instruction Following)           │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━             │
│  Objectif : Apprendre à répondre questions               │
│  Dataset : 158K instruction-following pairs              │
│  Frozen : CLIP ❄️                                        │
│  Trainable : Projector 🔥 + Llama (LoRA) 🔥             │
│  Epochs : 3                                               │
│  Batch size : 32                                          │
│  LR : 2e-5                                                │
│  Time : 10 hours on 8× A100                              │
│  Loss : Instruction tuning loss                          │
│                                                           │
│  Exemple :                                                │
│    Image : [Diagramme circuit]                           │
│    Question : "What's wrong with this circuit?"          │
│    Answer : "The resistor value is too low..."           │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

**Dataset Generation (The Secret Sauce)** 🤫

LLaVA utilise GPT-4 pour GÉNÉRER le dataset !

```python
"""
Bootstrapping avec GPT-4 : Le Hack Génial
==========================================

Problème : Besoin de 158k paires (image, question, answer)
          Annotation humaine = $$$$ (cher et lent)

Solution : Utiliser GPT-4 pour générer Q&A !

Pipeline :
1. Prendre image du dataset (ex: COCO)
2. Avoir caption existante : "A person skiing down a mountain"
3. Demander à GPT-4 (text-only!) de générer Q&A diverse
"""

def generate_llava_dataset():
    """Pipeline de génération dataset LLaVA"""

    # Pour chaque image du dataset source
    for image, caption in coco_dataset:

        # Prompt à GPT-4 (text-only, pas vision!)
        prompt = f"""
        Given an image with caption: "{caption}"

        Generate 3 diverse question-answer pairs that require
        visual understanding. Include:
        - 1 descriptive question (What/Where)
        - 1 reasoning question (Why/How)
        - 1 detailed analysis question

        Format: JSON
        """

        # GPT-4 génère les Q&A
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )

        qa_pairs = json.loads(response.choices[0].message.content)

        # Exemple output :
        # [
        #   {
        #     "question": "What activity is the person doing?",
        #     "answer": "The person is skiing down a snow-covered mountain..."
        #   },
        #   {
        #     "question": "Why might this be considered dangerous?",
        #     "answer": "Skiing down steep mountain slopes can be risky due to..."
        #   },
        #   {
        #     "question": "Describe the environment and conditions.",
        #     "answer": "The scene shows a mountainous terrain with deep snow..."
        #   }
        # ]

        # Sauvegarder paire (image, Q&A)
        save_training_sample(image, qa_pairs)

# Résultat : 158K échantillons générés en quelques jours
# Coût : ~$500 de crédits API GPT-4
# Qualité : Proche annotation humaine !
```

**Pourquoi Ça Marche** :
1. GPT-4 (text) génère questions sophistiquées
2. LLaVA apprend à y répondre en VOYANT l'image
3. Bootstrapping : Model A (GPT-4) aide Model B (LLaVA)
4. Cycle vertueux : LLaVA devient quasi aussi bon que GPT-4V !

### 5.4 Code Complet : Utiliser LLaVA

```python
"""
UTILISATION PRATIQUE DE LLAVA
=============================

Installation :
  pip install llava transformers torch pillow

Usage : Poser des questions sur images !
"""

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from PIL import Image
import torch

class LLaVAChatbot:
    """Interface simple pour discuter avec LLaVA"""

    def __init__(self, model_path="liuhaotian/llava-v1.5-7b"):
        """
        Charger modèle LLaVA

        Args:
            model_path: Chemin HuggingFace du modèle
                       Options :
                       - llava-v1.5-7b (rapide, 7B params)
                       - llava-v1.5-13b (meilleur, 13B params)
                       - llava-v1.6-34b (SOTA, 34B params)
        """
        print(f"Chargement de {model_path}...")

        self.tokenizer, self.model, self.image_processor, self.context_len = \
            load_pretrained_model(
                model_path=model_path,
                model_base=None,
                model_name=get_model_name_from_path(model_path),
                load_8bit=False,  # Mettre True si peu de VRAM
                load_4bit=False   # Ou quantization 4-bit (QLoRA)
            )

        print("Modèle chargé ! 🎉")

    def chat(self, image_path: str, question: str, temperature=0.2, max_tokens=512):
        """
        Poser une question sur une image

        Args:
            image_path: Chemin vers l'image
            question: Question en langage naturel
            temperature: Créativité (0=factuel, 1=créatif)
            max_tokens: Longueur max réponse

        Returns:
            answer: Réponse du modèle
        """

        # 1. Charger et preprocesser image
        image = Image.open(image_path).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model.config)
        image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

        # 2. Formater prompt
        # LLaVA utilise format spécial avec token <image>
        prompt = f"USER: <image>\n{question}\nASSISTANT:"

        # 3. Tokenize
        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors='pt'
        ).unsqueeze(0).to(self.model.device)

        # 4. Génération
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                max_new_tokens=max_tokens,
                use_cache=True
            )

        # 5. Décoder réponse
        answer = self.tokenizer.decode(
            output_ids[0, input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()

        return answer


# ═══════════════════════════════════════════════════════════
# 🎯 EXAMPLES D'UTILISATION
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Créer chatbot
    bot = LLaVAChatbot(model_path="liuhaotian/llava-v1.5-7b")

    # Exemple 1 : Description simple
    answer = bot.chat(
        image_path="cat.jpg",
        question="What's in this image?"
    )
    print(f"Q: What's in this image?")
    print(f"A: {answer}\n")
    # Output : "The image shows a fluffy orange cat lying on a blue couch..."

    # Exemple 2 : Comptage
    answer = bot.chat(
        image_path="fruit_basket.jpg",
        question="How many apples are in the basket?"
    )
    print(f"Q: How many apples?")
    print(f"A: {answer}\n")
    # Output : "There are approximately 5-6 apples in the basket."

    # Exemple 3 : Reasoning
    answer = bot.chat(
        image_path="broken_circuit.jpg",
        question="What's wrong with this electronic circuit and how to fix it?"
    )
    print(f"Q: What's wrong with circuit?")
    print(f"A: {answer}\n")
    # Output : "The circuit appears to have a short circuit between..."

    # Exemple 4 : OCR
    answer = bot.chat(
        image_path="handwritten_note.jpg",
        question="Transcribe the handwritten text in this image."
    )
    print(f"Q: Transcribe text")
    print(f"A: {answer}\n")

    # Exemple 5 : Créatif
    answer = bot.chat(
        image_path="sunset.jpg",
        question="Write a haiku about this sunset.",
        temperature=0.7  # Plus créatif
    )
    print(f"Q: Write haiku")
    print(f"A: {answer}\n")
    # Output :
    # "Golden rays descend
    #  Painting clouds in crimson hues
    #  Day whispers goodbye"
```

---

## 6. BLIP-2 et Flamingo : Architectures Alternatives

### 6.1 BLIP-2 : Le Q-Former Genius

**BLIP-2** (Salesforce, 2023) a introduit une architecture révolutionnaire : le **Q-Former**.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💬 **DIALOGUE : Comprendre le Q-Former**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Alice** : "Bob, LLaVA envoie 256 tokens visuels au LLM. C'est pas un peu... beaucoup ?"

**Bob** : "Excellente observation ! Imagine que tu regardes une photo. Tu ne mémorises pas CHAQUE pixel, n'est-ce pas ?"

**Alice** : "Non, j'extrais les infos importantes : 'chat orange', 'canapé bleu', 'fenêtre lumineuse'..."

**Bob** : "Exactement ! C'est ce que fait le Q-Former. Au lieu de garder 256 patches, il pose des 'questions intelligentes' à l'image et garde seulement les réponses. Résultat : 32 tokens au lieu de 256."

**Alice** : "Attends... des 'questions' à une image ? C'est pas un peu abstrait ?"

**Bob** : "Pense à ça comme un interrogatoire de détective :
- Question 1 : 'Y a-t-il un objet principal ?' → Oui, un chat
- Question 2 : 'Quelle est sa couleur ?' → Orange
- Question 3 : 'Où est-il situé ?' → Sur un canapé
- ...32 questions au total

Chaque 'question' (query) extrait UNE information importante via attention. 32 queries = 32 infos essentielles = 32 tokens compressés !"

**Alice** : "C'est brillant ! Donc plus efficace que LLaVA qui garde tout ?"

**Bob** : "Pour les longs contextes, oui. Mais LLaVA est plus simple et fonctionne déjà super bien. Trade-off complexité vs performance."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### 6.2 Architecture BLIP-2

```python
"""
BLIP-2 Architecture
===================

Innovation : Q-Former = Learnable Queries + Cross-Attention
Résultat : Compression intelligente 256 → 32 tokens
"""

import torch
import torch.nn as nn

class QFormer(nn.Module):
    """
    Q-Former : Query Transformer pour compression vision

    Intuition :
    - 32 queries apprenables (comme des 'questions')
    - Cross-attention : Queries 'interrogent' l'image
    - Self-attention : Queries se raffinent entre elles
    - Output : 32 tokens riches en information
    """

    def __init__(
        self,
        num_queries=32,
        hidden_dim=768,
        num_attention_heads=12,
        num_layers=6
    ):
        super().__init__()

        # Learnable queries (les "questions" posées à l'image)
        self.query_tokens = nn.Parameter(torch.randn(num_queries, hidden_dim))
        # Shape : [32, 768]

        # Transformer Encoder Layers
        self.layers = nn.ModuleList([
            QFormerLayer(hidden_dim, num_attention_heads)
            for _ in range(num_layers)
        ])

        # Layer norm final
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, vision_features):
        """
        Args:
            vision_features: [batch, 256, 1024] - CLIP output

        Returns:
            compressed: [batch, 32, 768] - Features compressées
        """
        batch_size = vision_features.shape[0]

        # Répéter queries pour chaque image du batch
        queries = self.query_tokens.unsqueeze(0).repeat(batch_size, 1, 1)
        # Shape : [batch, 32, 768]

        # Passer à travers les layers
        for layer in self.layers:
            queries = layer(
                queries=queries,
                vision_features=vision_features
            )

        # Normalisation finale
        queries = self.ln(queries)

        return queries


class QFormerLayer(nn.Module):
    """Une couche du Q-Former avec cross + self attention"""

    def __init__(self, hidden_dim, num_heads):
        super().__init__()

        # Cross-Attention : Queries → Vision
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.cross_attn_ln = nn.LayerNorm(hidden_dim)

        # Self-Attention : Queries → Queries
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.self_attn_ln = nn.LayerNorm(hidden_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.ffn_ln = nn.LayerNorm(hidden_dim)

    def forward(self, queries, vision_features):
        """
        Args:
            queries: [batch, 32, 768]
            vision_features: [batch, 256, 1024]
        """

        # 1. Cross-Attention : Queries "regardent" l'image
        attended, _ = self.cross_attention(
            query=queries,
            key=vision_features,
            value=vision_features
        )
        queries = self.cross_attn_ln(queries + attended)  # Residual

        # 2. Self-Attention : Queries se parlent entre elles
        self_attended, _ = self.self_attention(
            query=queries,
            key=queries,
            value=queries
        )
        queries = self.self_attn_ln(queries + self_attended)  # Residual

        # 3. FFN
        ffn_output = self.ffn(queries)
        queries = self.ffn_ln(queries + ffn_output)  # Residual

        return queries


class BLIP2Model(nn.Module):
    """Modèle BLIP-2 complet"""

    def __init__(self):
        super().__init__()

        # Vision Encoder (FROZEN ❄️)
        from transformers import CLIPVisionModel
        self.vision_encoder = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        self.vision_encoder.requires_grad_(False)

        # Q-Former (TRAINABLE 🔥)
        self.qformer = QFormer(
            num_queries=32,
            hidden_dim=768,
            num_attention_heads=12,
            num_layers=6
        )

        # Projection vers LLM (TRAINABLE 🔥)
        self.projection = nn.Linear(768, 4096)  # 768 → Llama dim

        # LLM (FROZEN ❄️)
        from transformers import AutoModelForCausalLM
        self.llm = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf"
        )
        self.llm.requires_grad_(False)

    def forward(self, images, input_ids):
        batch_size = images.shape[0]

        # 1. Vision encoding
        with torch.no_grad():
            vision_outputs = self.vision_encoder(images)
            vision_features = vision_outputs.last_hidden_state
            # [batch, 256, 1024]

        # 2. Q-Former compression (THE MAGIC ✨)
        compressed_features = self.qformer(vision_features)
        # [batch, 32, 768] - De 256 → 32 tokens !

        # 3. Projection vers LLM space
        llm_features = self.projection(compressed_features)
        # [batch, 32, 4096]

        # 4. Concat avec texte et forward LLM
        # (Similaire à LLaVA mais avec seulement 32 tokens visuels)
        outputs = self.llm(
            inputs_embeds=llm_features,
            # ... reste similaire à LLaVA
        )

        return outputs


# ═══════════════════════════════════════════════════════════
# 💡 POURQUOI Q-FORMER EST GÉNIAL
# ═══════════════════════════════════════════════════════════
#
# AVANTAGES :
# 1. Compression 8× (256 → 32 tokens)
# 2. Contexte plus long disponible pour texte
# 3. Inference plus rapide (moins de tokens à traiter)
# 4. Flexible : Change facilement le nombre de queries
#
# INCONVÉNIENTS :
# 1. Plus complexe à entraîner
# 2. Plus de paramètres (Q-Former = 188M params)
# 3. Risque de perte d'info si trop peu de queries
#
# USE CASES :
# - Long documents avec images
# - Multi-image conversations
# - Applications où contexte limité critique
#
# ═══════════════════════════════════════════════════════════
```

### 6.3 Flamingo : Few-Shot Learning Master

**Flamingo** (DeepMind, 2022) a été le premier "vrai" LLM multimodal avec capacités **few-shot**.

**Innovation** : Perceiver Resampler + Gated Cross-Attention

```
┌──────────────────────────────────────────────────────────┐
│              FLAMINGO ARCHITECTURE                        │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  Vision Encoder (Normalizer-Free ResNet)                 │
│         ↓                                                 │
│  Perceiver Resampler (Compression)                       │
│         ↓                                                 │
│  LLM avec Gated Cross-Attention Layers                   │
│         ↓                                                 │
│  Text Generation                                          │
│                                                           │
│  INNOVATION : Gated Cross-Attention                      │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   │
│                                                           │
│  LLM layers alternent :                                  │
│  1. Self-Attention (text-only)                           │
│  2. Gated Cross-Attention (text ↔ vision)               │
│  3. FFN                                                   │
│                                                           │
│  Le "gating" permet au modèle de choisir :              │
│  - Utiliser vision (gate=1)                              │
│  - Ignorer vision (gate=0)                               │
│                                                           │
│  Résultat : Flexible et adaptable !                      │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

**Capacité Few-Shot Impressionnante** :

```
Input (Few-shot examples) :
  [Image1: Red car] → "This is a red car"
  [Image2: Blue truck] → "This is a blue truck"
  [Image3: Green motorcycle] → ???

Flamingo Output : "This is a green motorcycle"

Learned pattern from just 2 examples! 🤯
```

⚠️ **Limitation** : Flamingo est **fermé** (DeepMind n'a pas publié les poids)

---

## 7. Au-delà de la Vision : Audio et Vidéo

### 7.1 Audio-Language Models

**Whisper** (OpenAI, 2022) + LLM = Chatbot vocal

```python
"""
AUDIO-LANGUAGE PIPELINE
=======================

Pipeline : Audio → Transcription → LLM → Response
"""

from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

class AudioLLMChatbot:
    """Chatbot qui comprend l'audio"""

    def __init__(self):
        # Audio → Text (Whisper)
        self.whisper_processor = WhisperProcessor.from_pretrained(
            "openai/whisper-large-v3"
        )
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-large-v3"
        )

        # Text → Response (LLM)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.llm = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

    def process_audio(self, audio_path):
        """
        Traiter audio et générer réponse

        Args:
            audio_path: Chemin vers fichier audio (.wav, .mp3)

        Returns:
            response: Réponse textuelle du LLM
        """
        import librosa

        # 1. Charger audio
        audio, sr = librosa.load(audio_path, sr=16000)

        # 2. Transcription avec Whisper
        inputs = self.whisper_processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        )

        with torch.no_grad():
            generated_ids = self.whisper_model.generate(inputs.input_features)

        transcription = self.whisper_processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        print(f"🎤 Transcription: {transcription}")

        # 3. Réponse LLM
        prompt = f"USER: {transcription}\nASSISTANT:"
        inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("ASSISTANT:")[-1].strip()

        print(f"🤖 Response: {response}")

        return response


# Exemple d'utilisation
if __name__ == "__main__":
    bot = AudioLLMChatbot()

    # User enregistre : "What's the weather like today?"
    response = bot.process_audio("user_question.wav")

    # Bot répond en texte (peut être converti en audio avec TTS)
```

**AudioLM** (Google, 2023) : Génération audio directe (pas de texte intermédiaire)

### 7.2 Video Understanding

**Vidéo = Séquence d'images + Audio**

**Challenge** : Une vidéo de 1 minute = 1800 frames (30 fps) !

**Solutions** :

**1. Frame Sampling** : Prendre 1 frame toutes les N secondes
```python
# Extraire 8 frames d'une vidéo de 30 secondes
frames = extract_frames(video, num_frames=8)  # 1 frame tous les 4 sec

# Traiter comme multi-image
for frame in frames:
    features = vision_encoder(frame)
    # Concat toutes les features
```

**2. Video-Specific Encoders** : ViViT, VideoMAE
- Utilisent attention spatio-temporelle
- Capturent le mouvement entre frames

**3. Gemini 1.5 Pro Approach** : Long-context
```
Gemini 1.5 Pro peut analyser 1 HEURE de vidéo !

Comment ?
- Compression spatiale (comme Q-Former)
- Compression temporelle (sampling intelligent)
- Long context window (1M tokens)

Résultat : Peut répondre "À quelle minute le personnage
            principal apparaît-il pour la première fois ?"
```

**Exemple Code** :

```python
"""
VIDEO UNDERSTANDING avec LLaVA-Video (conceptuel)
"""

class VideoLLM:
    """Analyser des vidéos avec un LLM"""

    def __init__(self):
        self.vision_encoder = load_clip()
        self.llm = load_llama()

    def analyze_video(self, video_path, question):
        """
        Analyser vidéo et répondre à question

        Args:
            video_path: Chemin vers vidéo
            question: Question sur la vidéo
        """

        # 1. Extraire frames (intelligent sampling)
        frames = self.extract_key_frames(video_path, num_frames=16)
        # Prend frames aux moments clés (changements de scène, etc.)

        # 2. Encoder chaque frame
        frame_features = []
        for frame in frames:
            features = self.vision_encoder(frame)
            frame_features.append(features)

        # 3. Concat temporellement
        video_features = torch.cat(frame_features, dim=1)
        # Shape : [1, 16*256, 1024] = [1, 4096, 1024]

        # 4. Projection et LLM
        projected = self.projection(video_features)

        # 5. Question + Réponse
        prompt = f"Video context: <VIDEO_FEATURES>\nQuestion: {question}\nAnswer:"
        response = self.llm.generate(prompt, video_features=projected)

        return response

    def extract_key_frames(self, video_path, num_frames=16):
        """
        Extraction intelligente de frames clés

        Méthodes :
        1. Uniform sampling : 1 frame tous les N secondes
        2. Change detection : Frames où scène change
        3. Importance sampling : Frames avec le plus d'action
        """
        import cv2

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Uniform sampling pour simplification
        frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)

        cap.release()
        return frames


# Use case : Analyse de film
vlm = VideoLLM()

response = vlm.analyze_video(
    "movie.mp4",
    "Summarize the main plot points of this movie."
)

print(response)
# "The movie follows a hero's journey where the protagonist
#  discovers hidden powers, faces betrayal from a trusted ally,
#  and ultimately saves their world in a climactic battle..."
```

**Applications Vidéo Réelles** :
- 📹 **Surveillance** : "Détecte si quelqu'un vole dans cette vidéo"
- 🎬 **Editing** : "Trouve tous les plans où le personnage sourit"
- 🏀 **Sports Analysis** : "Combien de shoots à 3 points dans ce match ?"
- 🎓 **Education** : "Résume ce cours de 1 heure en 3 bullet points"

---

## 8. Training Paradigms

### 8.1 Les Trois Approches

```
┌──────────────────────────────────────────────────────────┐
│         TRAINING STRATEGIES MULTIMODAL                    │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  1. Freeze Vision + LLM, Train Projector Only            │
│     ✅ Rapide, cheap ($500)                              │
│     ✅ Stable (pas de catastrophic forgetting)           │
│     ❌ Performance limitée                               │
│     👉 LLaVA, MiniGPT-4                                  │
│                                                           │
│  2. Freeze Vision, Fine-tune LLM + Projector             │
│     ✅ Meilleure performance                             │
│     ✅ Adaptable au domaine                              │
│     ❌ Plus cher (~$5k-10k)                              │
│     ❌ Risque overfitting                                │
│     👉 LLaVA-1.5, Qwen-VL                                │
│                                                           │
│  3. Joint Training (Vision + LLM + Projector)            │
│     ✅ Performance SOTA                                   │
│     ✅ Alignement optimal                                │
│     ❌ Très cher ($100k+)                                │
│     ❌ Instable, difficile                               │
│     👉 GPT-4V, Gemini, Claude 3                          │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

### 8.2 Recipe Complète : Entraîner Votre Modèle Multimodal

**Étape par Étape** :

```python
"""
TRAINING RECIPE : Créer Votre LLaVA
====================================

Dataset : 595K image-caption (stage 1) + 158K instruction (stage 2)
Hardware : 8× A100 40GB
Temps : ~14 heures total
Coût : ~$500 sur cloud
"""

import torch
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# ════════════════════════════════════════════════════════════
# STAGE 1 : Pre-training (Feature Alignment)
# ════════════════════════════════════════════════════════════

def stage1_pretraining():
    """
    Objectif : Aligner vision features avec LLM space
    Dataset : LAION-CC-SBU (595K image-caption pairs)
    Frozen : CLIP + Llama
    Trainable : Projector ONLY
    """

    # 1. Charger modèle
    model = LLaVAModel(
        vision_tower="openai/clip-vit-large-patch14",
        language_model="meta-llama/Llama-2-7b-hf"
    )

    # Freeze vision + LLM
    model.vision_tower.requires_grad_(False)
    model.language_model.requires_grad_(False)
    # Projector reste trainable

    # 2. Dataset
    dataset = load_dataset("liuhaotian/LLaVA-Pretrain")
    # Format : {"image": PIL.Image, "caption": "A cat sitting..."}

    # 3. Training args
    training_args = TrainingArguments(
        output_dir="./llava-stage1",
        num_train_epochs=1,
        per_device_train_batch_size=16,  # × 8 GPUs = 128 total
        gradient_accumulation_steps=1,
        learning_rate=2e-3,
        warmup_steps=1000,
        lr_scheduler_type="cosine",
        save_steps=5000,
        logging_steps=100,
        bf16=True,                       # Mixed precision
        dataloader_num_workers=4,
        remove_unused_columns=False
    )

    # 4. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=llava_collate_fn  # Custom collator
    )

    # 5. Train !
    trainer.train()

    # 6. Save projector weights
    torch.save(
        model.mm_projector.state_dict(),
        "projector_weights.pth"
    )

    print("✅ Stage 1 terminé !")
    print("Projector aligné : vision features → LLM space")


# ════════════════════════════════════════════════════════════
# STAGE 2 : Instruction Tuning
# ════════════════════════════════════════════════════════════

def stage2_instruction_tuning():
    """
    Objectif : Apprendre à suivre instructions
    Dataset : LLaVA-Instruct (158K instruction pairs)
    Frozen : CLIP
    Trainable : Projector + Llama (avec LoRA)
    """

    # 1. Charger modèle avec projector pré-entraîné
    model = LLaVAModel(
        vision_tower="openai/clip-vit-large-patch14",
        language_model="meta-llama/Llama-2-7b-hf"
    )
    model.mm_projector.load_state_dict(
        torch.load("projector_weights.pth")
    )

    # Freeze vision
    model.vision_tower.requires_grad_(False)

    # Unfreeze LLM avec LoRA
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model.language_model = get_peft_model(
        model.language_model,
        lora_config
    )

    # 2. Dataset
    dataset = load_dataset("liuhaotian/LLaVA-Instruct-150K")
    # Format : {
    #   "image": PIL.Image,
    #   "conversations": [
    #     {"from": "human", "value": "What's in this image?"},
    #     {"from": "gpt", "value": "The image shows..."}
    #   ]
    # }

    # 3. Training args
    training_args = TrainingArguments(
        output_dir="./llava-stage2",
        num_train_epochs=3,
        per_device_train_batch_size=4,   # × 8 GPUs = 32 total
        gradient_accumulation_steps=4,   # Effective batch = 128
        learning_rate=2e-5,              # Plus petit que stage 1
        warmup_steps=100,
        lr_scheduler_type="cosine",
        save_steps=1000,
        logging_steps=50,
        bf16=True,
        dataloader_num_workers=4,
        remove_unused_columns=False
    )

    # 4. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=llava_instruction_collate_fn
    )

    # 5. Train !
    trainer.train()

    # 6. Save final model
    model.save_pretrained("./llava-final")

    print("✅ Stage 2 terminé !")
    print("Modèle prêt pour instruction following !")


# ════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════

def llava_collate_fn(batch):
    """Collate function pour stage 1 (caption)"""
    images = [item['image'] for item in batch]
    captions = [item['caption'] for item in batch]

    # Process images
    from PIL import Image
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])

    images_tensor = torch.stack([transform(img) for img in images])

    # Tokenize captions
    # Format : "<image>\n{caption}"
    # ...

    return {
        'images': images_tensor,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


def llava_instruction_collate_fn(batch):
    """Collate function pour stage 2 (instruction)"""
    # Similar mais avec conversations multi-turn
    # ...
    pass


# ════════════════════════════════════════════════════════════
# MAIN : Exécuter le training
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🚀 Démarrage training LLaVA !")
    print()

    # Stage 1 : ~4 hours
    print("=" * 60)
    print("STAGE 1 : Pre-training")
    print("=" * 60)
    stage1_pretraining()

    # Stage 2 : ~10 hours
    print()
    print("=" * 60)
    print("STAGE 2 : Instruction Tuning")
    print("=" * 60)
    stage2_instruction_tuning()

    print()
    print("🎉 Training complet terminé !")
    print("Total time : ~14 hours")
    print("Total cost : ~$500")
    print()
    print("Votre modèle multimodal est prêt ! 🦙👁️")
```

---

## 9. Projet Pratique : Créer Votre Chatbot Vision

### 9.1 Objectif du Projet

Créer un **chatbot multimodal complet** avec :
- ✅ Interface web (Gradio)
- ✅ Support images (upload ou URL)
- ✅ Conversation multi-turn
- ✅ Historique
- ✅ Déploiement

**Temps estimé** : 2-3 heures
**Niveau** : Intermédiaire

### 9.2 Architecture

```
┌──────────────────────────────────────────────────────────┐
│           VISION CHATBOT ARCHITECTURE                     │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  Frontend (Gradio)                                        │
│      ↓                                                    │
│  API Layer (FastAPI)                                     │
│      ↓                                                    │
│  LLaVA Model (Inference)                                 │
│      ↓                                                    │
│  Response                                                 │
│                                                           │
│  Features :                                              │
│  - Image upload                                          │
│  - Multi-turn conversation                               │
│  - History tracking                                      │
│  - Temperature control                                   │
│  - Max tokens slider                                     │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

### 9.3 Code Complet

**Partie 1 : Backend (FastAPI)**

```python
"""
vision_chatbot/backend.py
=========================

Backend FastAPI pour chatbot vision
"""

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import torch
from PIL import Image
import io
import base64

# Importer LLaVA
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX

app = FastAPI(title="Vision Chatbot API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ════════════════════════════════════════════════════════════
# MODELS
# ════════════════════════════════════════════════════════════

class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    image_data: Optional[str] = None  # Base64 encoded

class ChatRequest(BaseModel):
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: int = 512

class ChatResponse(BaseModel):
    response: str
    finish_reason: str = "stop"

# ════════════════════════════════════════════════════════════
# LOAD MODEL
# ════════════════════════════════════════════════════════════

print("Loading LLaVA model...")
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path="liuhaotian/llava-v1.5-7b",
    model_base=None,
    model_name="llava-v1.5-7b",
    load_8bit=True  # Quantization pour économiser VRAM
)
print("Model loaded! ✅")

# ════════════════════════════════════════════════════════════
# ENDPOINTS
# ════════════════════════════════════════════════════════════

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint avec support images

    Request format :
    {
        "messages": [
            {"role": "user", "content": "What's this?", "image_data": "base64..."},
            {"role": "assistant", "content": "It's a cat."},
            {"role": "user", "content": "What color?"}
        ],
        "temperature": 0.7,
        "max_tokens": 512
    }
    """

    # Construire le prompt
    prompt_parts = []
    images = []

    for msg in request.messages:
        if msg.role == "user":
            if msg.image_data:
                # Décoder image
                image_bytes = base64.b64decode(msg.image_data)
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                images.append(image)
                prompt_parts.append(f"USER: <image>\n{msg.content}")
            else:
                prompt_parts.append(f"USER: {msg.content}")
        else:
            prompt_parts.append(f"ASSISTANT: {msg.content}")

    prompt_parts.append("ASSISTANT:")
    prompt = "\n".join(prompt_parts)

    # Process images
    if images:
        image_tensors = process_images(images, image_processor, model.config)
        image_tensors = image_tensors.to(model.device, dtype=torch.float16)
    else:
        image_tensors = None

    # Tokenize
    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors='pt'
    ).unsqueeze(0).to(model.device)

    # Generate
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensors,
            do_sample=request.temperature > 0,
            temperature=request.temperature,
            max_new_tokens=request.max_tokens,
            use_cache=True
        )

    # Decode
    response = tokenizer.decode(
        output_ids[0, input_ids.shape[1]:],
        skip_special_tokens=True
    ).strip()

    return ChatResponse(response=response)


@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """Upload image et retourner base64"""
    contents = await file.read()
    base64_image = base64.b64encode(contents).decode('utf-8')
    return {"image_data": base64_image}


@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy", "model": "llava-v1.5-7b"}


# ════════════════════════════════════════════════════════════
# RUN
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Partie 2 : Frontend (Gradio)**

```python
"""
vision_chatbot/frontend.py
==========================

Interface Gradio pour chatbot vision
"""

import gradio as gr
import requests
import base64
from PIL import Image
import io

API_URL = "http://localhost:8000"

# ════════════════════════════════════════════════════════════
# STATE MANAGEMENT
# ════════════════════════════════════════════════════════════

class ChatState:
    def __init__(self):
        self.messages = []
        self.current_image = None

    def add_user_message(self, text, image=None):
        msg = {"role": "user", "content": text}
        if image:
            # Convertir PIL Image en base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            msg["image_data"] = img_str
            self.current_image = image
        self.messages.append(msg)

    def add_assistant_message(self, text):
        self.messages.append({"role": "assistant", "content": text})

    def clear(self):
        self.messages = []
        self.current_image = None

# Global state
chat_state = ChatState()

# ════════════════════════════════════════════════════════════
# FUNCTIONS
# ════════════════════════════════════════════════════════════

def chat_fn(user_message, image, temperature, max_tokens, history):
    """
    Fonction principale de chat

    Args:
        user_message: Message de l'utilisateur
        image: Image uploadée (PIL Image ou None)
        temperature: Température de sampling
        max_tokens: Nombre max de tokens
        history: Historique de conversation (pour Gradio)

    Returns:
        ("", history_updated) - Vider input et update history
    """

    # Ajouter message utilisateur
    chat_state.add_user_message(user_message, image)

    # Construire affichage pour historique
    if image:
        display_msg = f"🖼️ [Image] {user_message}"
    else:
        display_msg = user_message

    history.append([display_msg, None])  # User message, no response yet

    # Appeler API
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={
                "messages": chat_state.messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            },
            timeout=60
        )

        if response.status_code == 200:
            assistant_response = response.json()["response"]
            chat_state.add_assistant_message(assistant_response)
            history[-1][1] = assistant_response
        else:
            history[-1][1] = f"❌ Error: {response.status_code}"

    except Exception as e:
        history[-1][1] = f"❌ Error: {str(e)}"

    return "", history


def clear_fn():
    """Clear conversation"""
    chat_state.clear()
    return None, []


def retry_fn(history, temperature, max_tokens):
    """Retry dernière réponse"""
    if len(chat_state.messages) >= 2:
        # Remove dernière réponse
        chat_state.messages.pop()
        history.pop()

        # Re-générer
        response = requests.post(
            f"{API_URL}/chat",
            json={
                "messages": chat_state.messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        )

        assistant_response = response.json()["response"]
        chat_state.add_assistant_message(assistant_response)
        history.append([history[-1][0], assistant_response])

    return history


# ════════════════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════════════════

with gr.Blocks(title="Vision Chatbot", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🦙👁️ Vision Chatbot

    Chatbot multimodal powered by LLaVA.
    Upload une image et pose des questions !
    """)

    with gr.Row():
        with gr.Column(scale=2):
            # Chatbox
            chatbot = gr.Chatbot(
                label="Conversation",
                height=500,
                bubble_full_width=False
            )

            # Input
            with gr.Row():
                user_input = gr.Textbox(
                    label="Message",
                    placeholder="Pose une question sur l'image...",
                    scale=4
                )
                submit_btn = gr.Button("📤 Send", scale=1, variant="primary")

            # Buttons
            with gr.Row():
                retry_btn = gr.Button("🔄 Retry")
                clear_btn = gr.Button("🗑️ Clear")

        with gr.Column(scale=1):
            # Image upload
            image_input = gr.Image(
                label="Upload Image",
                type="pil",
                height=300
            )

            gr.Markdown("### ⚙️ Paramètres")

            temperature = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.7,
                step=0.1,
                label="Temperature",
                info="Créativité (0=factuel, 1=créatif)"
            )

            max_tokens = gr.Slider(
                minimum=50,
                maximum=1024,
                value=512,
                step=50,
                label="Max Tokens",
                info="Longueur max de la réponse"
            )

            gr.Markdown("""
            ### 💡 Tips
            - Upload une image en premier
            - Pose des questions descriptives
            - Utilise temperature=0 pour réponses factuelles
            - Utilise temperature=0.7-1.0 pour créativité

            ### 📝 Exemples
            - "What's in this image?"
            - "Describe the scene in detail"
            - "What color is the car?"
            - "How many people are there?"
            - "What emotion does this convey?"
            """)

    # Events
    submit_btn.click(
        fn=chat_fn,
        inputs=[user_input, image_input, temperature, max_tokens, chatbot],
        outputs=[user_input, chatbot]
    )

    user_input.submit(
        fn=chat_fn,
        inputs=[user_input, image_input, temperature, max_tokens, chatbot],
        outputs=[user_input, chatbot]
    )

    retry_btn.click(
        fn=retry_fn,
        inputs=[chatbot, temperature, max_tokens],
        outputs=[chatbot]
    )

    clear_btn.click(
        fn=clear_fn,
        outputs=[image_input, chatbot]
    )

# Launch
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
```

**Partie 3 : Docker Deployment**

```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose ports
EXPOSE 8000 7860

# Run both backend and frontend
CMD ["bash", "start.sh"]
```

```bash
# start.sh
#!/bin/bash

# Start backend
python3 backend.py &

# Wait for backend
sleep 10

# Start frontend
python3 frontend.py
```

```txt
# requirements.txt
fastapi==0.104.1
uvicorn==0.24.0
gradio==4.10.0
torch==2.1.0
torchvision==0.16.0
transformers==4.36.0
accelerate==0.25.0
bitsandbytes==0.41.3
llava @ git+https://github.com/haotian-liu/LLaVA.git
pillow==10.1.0
requests==2.31.0
```

### 9.4 Utilisation

**1. Lancer le backend** :
```bash
python backend.py
# Backend running on http://localhost:8000
```

**2. Lancer le frontend** :
```bash
python frontend.py
# Gradio running on http://localhost:7860
```

**3. Utiliser** :
- Ouvrir http://localhost:7860
- Upload une image
- Poser des questions
- Conversation multi-turn !

**4. Déployer avec Docker** :
```bash
docker build -t vision-chatbot .
docker run -p 8000:8000 -p 7860:7860 --gpus all vision-chatbot
```

---

## 10. Best Practices et Troubleshooting

### 10.1 Best Practices

**✅ DO** :

1. **Prétraiter les images** :
```python
# Resize pour consistency
image = image.resize((224, 224))

# Normaliser avec stats CLIP
normalize = transforms.Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073],
    std=[0.26862954, 0.26130258, 0.27577711]
)
```

2. **Utiliser batch processing** :
```python
# Traiter plusieurs images en batch
images = [image1, image2, image3]
features = vision_encoder(torch.stack(images))
# Plus rapide que boucle !
```

3. **Cache les embeddings** :
```python
# Pour images fixes, cache les features
@lru_cache(maxsize=100)
def get_image_features(image_path):
    image = load_image(image_path)
    return vision_encoder(image)
```

4. **Monitorer la mémoire** :
```python
# Clear GPU cache régulièrement
if batch_idx % 100 == 0:
    torch.cuda.empty_cache()
```

**❌ DON'T** :

1. ❌ Oublier de normaliser les images
2. ❌ Utiliser des résolutions inconsistantes
3. ❌ Charger tout le dataset en RAM
4. ❌ Oublier de freeze les modèles pré-entraînés
5. ❌ Ignorer les warnings GPU memory

### 10.2 Troubleshooting Commun

**Problème 1** : Out of Memory (OOM)

```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions** :
```python
# Solution 1 : Quantization
model = load_model(load_8bit=True)  # INT8

# Solution 2 : Gradient checkpointing
model.gradient_checkpointing_enable()

# Solution 3 : Réduire batch size
batch_size = 4  # Au lieu de 16

# Solution 4 : Réduire résolution
image_size = 224  # Au lieu de 336

# Solution 5 : Clear cache
torch.cuda.empty_cache()
```

**Problème 2** : Génération lente

**Solutions** :
```python
# Solution 1 : Quantization
load_4bit=True  # 3-4× plus rapide

# Solution 2 : Compiler le modèle (PyTorch 2.0+)
model = torch.compile(model)

# Solution 3 : Réduire max_tokens
max_new_tokens=256  # Au lieu de 512

# Solution 4 : Utiliser cache KV
use_cache=True
```

**Problème 3** : Mauvaise qualité de réponses

**Solutions** :
```python
# Solution 1 : Ajuster temperature
temperature=0.2  # Plus factuel

# Solution 2 : Better prompting
prompt = """<image>
Analyze this image in detail. Include:
1. Main objects
2. Colors and composition
3. Context and setting
Describe:"""

# Solution 3 : Fine-tune sur votre domaine
# Train sur dataset spécifique

# Solution 4 : Utiliser un modèle plus grand
model = "llava-v1.6-34b"  # Au lieu de 7b
```

**Problème 4** : Images mal comprises

**Causes + Solutions** :
```python
# Cause 1 : Image corrompue
try:
    image = Image.open(path)
    image.verify()  # Check integrity
except:
    print("Image corrompue!")

# Cause 2 : Mauvais format
image = image.convert('RGB')  # Force RGB

# Cause 3 : Résolution trop basse
if image.size[0] < 224 or image.size[1] < 224:
    print("Warning: Image trop petite!")

# Cause 4 : Image trop complexe
# Crop ou focus sur partie importante
```

### 10.3 Monitoring en Production

```python
"""
MONITORING SETUP pour chatbot vision en production
"""

from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
REQUEST_COUNT = Counter('chat_requests_total', 'Total chat requests')
REQUEST_LATENCY = Histogram('chat_latency_seconds', 'Request latency')
ACTIVE_USERS = Gauge('active_users', 'Number of active users')
GPU_MEMORY = Gauge('gpu_memory_used_bytes', 'GPU memory used')

@app.post("/chat")
async def chat(request: ChatRequest):
    REQUEST_COUNT.inc()

    start_time = time.time()

    try:
        # Process...
        response = process_chat(request)

        # Metrics
        REQUEST_LATENCY.observe(time.time() - start_time)
        GPU_MEMORY.set(torch.cuda.memory_allocated())

        return response

    except Exception as e:
        # Log error
        logger.error(f"Chat error: {e}")
        raise


# Dashboard Grafana :
# - Latency p50, p95, p99
# - Throughput (requests/sec)
# - GPU utilization
# - Error rate
```

---

## 11. Quiz et Exercices

### 11.1 Quiz de Compréhension

═══════════════════════════════════════════════════════════
🎯 QUIZ : Testez Vos Connaissances Multimodal !
═══════════════════════════════════════════════════════════

**Question 1** [Facile] : Quel est le rôle du vision encoder dans un modèle multimodal ?

a) Générer du texte depuis une image
b) Convertir pixels en embeddings sémantiques  ✅
c) Traduire texte en image
d) Compresser l'image

**Réponse** : b) Le vision encoder (comme CLIP) transforme les pixels bruts en vecteurs qui capturent le sens sémantique de l'image.

---

**Question 2** [Moyen] : Pourquoi LLaVA envoie-t-il 256 tokens visuels au LLM ?

a) C'est le nombre de pixels
b) C'est le nombre de patches (14×14 + CLS)  ✅
c) C'est arbitraire
d) Pour ralentir le modèle

**Réponse** : b) CLIP découpe l'image en patches 14×14 = 196, plus un token CLS = 197 (arrondi à 256).

---

**Question 3** [Moyen] : Quelle est l'innovation principale de BLIP-2 ?

a) Utiliser CLIP
b) Le Q-Former qui compresse intelligemment  ✅
c) Multi-modal attention
d) Training plus rapide

**Réponse** : b) Le Q-Former utilise des learnable queries pour compresser 256 tokens → 32 tokens tout en gardant l'information importante.

---

**Question 4** [Avancé] : Comment LLaVA génère-t-il son dataset d'instruction ?

a) Annotation humaine
b) Scraping d'Internet
c) Utilise GPT-4 pour générer Q&A depuis captions  ✅
d) Synthèse automatique

**Réponse** : c) LLaVA utilise GPT-4 (text-only) pour générer des questions-réponses sophistiquées depuis les captions existantes. Génie !

---

**Question 5** [Avancé] : Pourquoi freeze-t-on CLIP et LLM pendant training ?

a) Pour économiser VRAM
b) Pour éviter catastrophic forgetting  ✅
c) Parce que c'est plus rapide
d) Par paresse

**Réponse** : b) Si on fine-tune CLIP/LLM, ils risquent d'oublier ce qu'ils ont appris. Mieux vaut entraîner seulement le "pont" (projector).

---

**Question 6** [Expert] : Calculez la mémoire nécessaire pour LLaVA-7B en FP16.

Indices :
- Llama-2-7B : 7B params
- CLIP : 427M params
- Projector : 100M params
- FP16 : 2 bytes/param

**Réponse** :
```
Total params = 7B + 0.427B + 0.1B = 7.527B
Memory = 7.527B × 2 bytes = 15.054 GB
+ Activations (~2GB) = ~17 GB total

Avec 8-bit quantization : ~8.5 GB
Avec 4-bit quantization : ~5 GB
```

═══════════════════════════════════════════════════════════

### 11.2 Exercices Pratiques

**Exercice 1** [Débutant] : Compter les objets

```python
"""
EXERCICE : Créer un compteur d'objets automatique

Input : Image avec plusieurs objets
Output : "Il y a X chats, Y chiens, Z voitures"

Difficulté : ⭐⚪⚪⚪⚪
Temps : 30 minutes
"""

def count_objects(image_path):
    """
    TODO : Implémenter

    Hint : Utiliser LLaVA avec prompt spécifique
    """
    pass

# Test
image = "street_scene.jpg"
result = count_objects(image)
# Expected : "2 cars, 3 people, 1 dog"
```

**Solution** :
```python
def count_objects(image_path):
    bot = LLaVAChatbot()

    prompt = """List all objects in this image with their counts.
    Format: "X object1, Y object2, Z object3"
    Be specific and accurate."""

    response = bot.chat(image_path, prompt, temperature=0.1)
    return response
```

---

**Exercice 2** [Intermédiaire] : Comparaison d'images

```python
"""
EXERCICE : Comparer deux images

Input : Deux images
Output : Similarités et différences

Difficulté : ⭐⭐⭐⚪⚪
Temps : 1 heure
"""

def compare_images(image1_path, image2_path):
    """
    TODO : Implémenter

    Hints :
    1. Encoder les deux images séparément
    2. Utiliser LLM pour comparer
    3. Ou calculer similarité cosine des features
    """
    pass

# Test
img1 = "cat1.jpg"
img2 = "cat2.jpg"
diff = compare_images(img1, img2)
# Expected : "Both show cats. Image 1 has orange cat,
#             Image 2 has black cat..."
```

**Solution** :
```python
def compare_images(image1_path, image2_path):
    bot = LLaVAChatbot()

    # Décrire image 1
    desc1 = bot.chat(image1_path, "Describe this image in detail.")

    # Décrire image 2
    desc2 = bot.chat(image2_path, "Describe this image in detail.")

    # Comparer
    comparison_prompt = f"""
    Compare these two descriptions:

    Image 1: {desc1}
    Image 2: {desc2}

    List:
    - Similarities
    - Differences
    - Which is better for [specific use case]?
    """

    # Note : Idéalement, multi-image input si supporté
    comparison = bot.chat(None, comparison_prompt)

    return comparison
```

---

**Exercice 3** [Avancé] : Video Summarization

```python
"""
EXERCICE : Résumer une vidéo

Input : Vidéo MP4
Output : Résumé textuel des événements

Difficulté : ⭐⭐⭐⭐⚪
Temps : 2-3 heures
"""

def summarize_video(video_path):
    """
    TODO : Implémenter

    Steps :
    1. Extraire frames clés (1 par seconde)
    2. Analyser chaque frame avec LLaVA
    3. Agréger les descriptions
    4. Générer résumé cohérent
    """
    pass

# Test
video = "cooking_tutorial.mp4"
summary = summarize_video(video)
# Expected : "The video shows a cooking tutorial where...
#             First, ingredients are prepared...
#             Then, the mixture is cooked...
#             Finally, the dish is plated..."
```

**Solution** :
```python
import cv2

def summarize_video(video_path, frames_per_second=1):
    bot = LLaVAChatbot()

    # 1. Extraire frames
    frames = extract_frames_from_video(video_path, fps=frames_per_second)

    # 2. Analyser chaque frame
    descriptions = []
    for i, frame in enumerate(frames):
        # Save frame temporairement
        frame_path = f"temp_frame_{i}.jpg"
        cv2.imwrite(frame_path, frame)

        # Analyser
        desc = bot.chat(
            frame_path,
            f"Describe what's happening at timestamp {i} seconds."
        )
        descriptions.append(f"[{i}s] {desc}")

    # 3. Agréger et résumer
    all_descriptions = "\n".join(descriptions)

    summary_prompt = f"""
    Based on these frame descriptions from a video:

    {all_descriptions}

    Write a coherent summary of the entire video in 2-3 paragraphs.
    Focus on the main storyline and key events.
    """

    summary = bot.chat(None, summary_prompt)

    return summary


def extract_frames_from_video(video_path, fps=1):
    """Extract frames at specified FPS"""
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)

    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frames.append(frame)

        frame_count += 1

    cap.release()
    return frames
```

---

**Exercice 4** [Expert] : Fine-tune pour Domaine Spécifique

```python
"""
EXERCICE FINAL : Fine-tune LLaVA pour domaine médical

Objectif : Adapter LLaVA pour analyser radiographies

Dataset : 1000 X-rays avec annotations médicales
Steps :
1. Préparer dataset (images + reports)
2. Fine-tune avec LoRA
3. Évaluer sur test set

Difficulté : ⭐⭐⭐⭐⭐
Temps : 1-2 jours
Coût : ~$50-100 GPU
"""

# Starter code fourni dans les ressources du chapitre
# Voir : medical_llava_finetune.py
```

═══════════════════════════════════════════════════════════

---

## 🎉 CONCLUSION DU CHAPITRE

Félicitations ! Vous maîtrisez maintenant les modèles multimodaux ! 🦙👁️

**Ce que vous avez appris** :
- ✅ Architecture complète (Vision → Projection → LLM)
- ✅ GPT-4V et ses capacités impressionnantes
- ✅ LLaVA : Open-source hero ($500 training!)
- ✅ BLIP-2 et Q-Former compression
- ✅ Audio et vidéo understanding
- ✅ Training pipeline complet
- ✅ Projet pratique : Chatbot vision déployable
- ✅ Best practices et troubleshooting

**Points clés à retenir** :
1. **Multimodal = Vision Encoder + Projection + LLM** (simple!)
2. **CLIP** est le standard pour vision encoding
3. **LLaVA** démontre qu'open-source peut rivaliser
4. **Q-Former** (BLIP-2) = compression intelligente
5. **Training** = Freeze encoders, train projector only
6. **Vidéo** = Séquence d'images avec sampling intelligent

**L'avenir du multimodal** :
- 🚀 Modèles natifs (Gemini 1.5 approach)
- 🎥 Vidéos longues (>1h context)
- 🎨 Génération image+texte jointe
- 🌍 Multimodal pour toutes les langues
- 🤖 Agents autonomes avec vision

**Prochaines étapes** :
- Pratiquer avec les exercices
- Déployer votre chatbot vision
- Fine-tune pour votre domaine
- Contribuer à la communauté open-source!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💬 **DERNIER MOT de Bob**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Bob** : "Alice, tu te souviens quand tu m'as demandé comment un LLM pouvait 'voir' ?"

**Alice** : "Oui ! Et maintenant je peux créer mon propre modèle vision pour $500. C'est dingue !"

**Bob** : "Le plus fou ? On n'a fait qu'effleurer la surface. Dans 2 ans, chaque app aura de la vision AI. Chaque robot, chaque voiture, chaque phone. La multimodalité n'est pas l'avenir—c'est le présent."

**Alice** : "Je vais commencer par ce chatbot médical. Imagine l'impact : aider les médecins à détecter les maladies plus tôt..."

**Bob** : "Voilà l'esprit ! La tech n'est qu'un outil. Ce qui compte, c'est ce que TU vas en faire. Go build something amazing! 🚀"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

---

**Ressources Additionnelles** :
- 📖 Papers : GPT-4V, LLaVA, BLIP-2, Flamingo
- 💻 Code : github.com/haotian-liu/LLaVA
- 🎥 Demos : https://llava.hliu.cc
- 💬 Community : HuggingFace Discord, r/LocalLLaMA
- 📚 Datasets : LAION, COCO, Visual Genome

**Prochain chapitre** : Chapitre 23 - Deployment & Production 🚀

---

*Fin du Chapitre 22 : Multimodal LLMs*
