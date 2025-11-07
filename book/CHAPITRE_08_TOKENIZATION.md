# CHAPITRE 8 : TOKENIZATION - L'ART DE DÉCOUPER LE LANGAGE

> *« Avant qu'un LLM puisse comprendre 'Hello', il doit d'abord apprendre à le **découper**. La tokenization est l'interface invisible entre le langage humain et les mathématiques des réseaux de neurones. »*

---

## 📖 Table des matières

1. [Introduction : Le Problème du Découpage](#1-introduction)
2. [Tokenization Character-Level](#2-character-level)
3. [Tokenization Word-Level](#3-word-level)
4. [Subword Tokenization : Le Standard Moderne](#4-subword)
5. [Byte Pair Encoding (BPE)](#5-bpe)
6. [WordPiece (BERT)](#6-wordpiece)
7. [SentencePiece & Unigram](#7-sentencepiece)
8. [Tiktoken (GPT-3/4)](#8-tiktoken)
9. [Tokenizers Multilingues](#9-multilingue)
10. [Problèmes et Limitations](#10-problemes)
11. [Implémentation from Scratch](#11-implementation)
12. [Quiz Interactif](#12-quiz)
13. [Exercices Pratiques](#13-exercices)
14. [Conclusion](#14-conclusion)
15. [Ressources](#15-ressources)

---

## 1. Introduction : Le Problème du Découpage {#1-introduction}

### 🎭 Dialogue : Le Dilemme de la Tokenization

**Alice** : Bob, pourquoi ChatGPT compte "strawberry" comme 2 tokens mais "apple" comme 1 ?

**Bob** : Excellente question ! C'est parce que "apple" est un mot fréquent dans les données d'entraînement, donc il a son propre token. "strawberry" est moins fréquent, donc il est découpé en "straw" + "berry".

**Alice** : Mais pourquoi découper ? Pourquoi ne pas donner un token à chaque mot du dictionnaire ?

**Bob** : Imagine :
- Anglais : ~170,000 mots
- Français : ~100,000 mots
- Noms propres, argot, typos : infini !

Un vocabulaire de millions de mots rendrait les embeddings gigantesques et le modèle incapable de gérer des mots inconnus.

**Alice** : Et si on utilisait des caractères ? 26 lettres + ponctuation = ~100 tokens !

**Bob** : Séquences trop longues ! "Hello world" = 11 caractères. Un article de 500 mots = 3000 caractères. L'attention en O(n²) exploserait.

**Alice** : Donc on a besoin d'un **compromis** ?

**Bob** : Exactement. Les **subword tokens** (sous-mots) sont le sweet spot : vocabulaire ~50k tokens, séquences raisonnables, zéro mot inconnu.

### 📊 Comparaison des Approches

| Méthode | Vocab Size | Sequence Length | OOV (Out-of-Vocab) | Modèles |
|---------|------------|-----------------|-------------------|---------|
| **Character** | ~100 | Très longue | ❌ Aucun | CharRNN (obsolète) |
| **Word** | 100k-1M | Courte | ❌ Beaucoup | Word2Vec (obsolète) |
| **Subword** | 30k-100k | Moyenne | ✅ Aucun | GPT, BERT, T5 |
| **Byte** | 256 | Très longue | ✅ Aucun | ByT5, CANINE |

### 🎯 Anecdote : La Naissance du BPE en NLP

**2015, Université d'Édimbourg**

Rico Sennrich et ses collègues travaillent sur la traduction automatique. Problème : les mots rares (noms propres, composés allemands) ne sont pas dans le vocabulaire.

*Sennrich* : "Et si on adaptait la compression de données ? Le Byte Pair Encoding (1994) merge les paires fréquentes de bytes..."

*Collègue* : "En NLP, on merge des caractères au lieu de bytes !"

**Résultat** : BPE devient le standard de facto. GPT-2 (2019), BERT (2018), presque tous les LLMs modernes l'utilisent ou en dérivent.

**Impact** :
- Vocabulaire réduit de 1M → 50k tokens
- Zéro OOV (tout mot peut être décomposé)
- Performances BLEU +2-3 points en traduction

### 🎯 Objectifs du Chapitre

À la fin de ce chapitre, vous saurez :

- ✅ Comprendre pourquoi la tokenization est cruciale
- ✅ Implémenter BPE from scratch
- ✅ Utiliser les tokenizers de HuggingFace
- ✅ Comparer BPE, WordPiece, SentencePiece, Unigram
- ✅ Diagnostiquer et résoudre les problèmes de tokenization
- ✅ Optimiser le vocabulaire pour votre domaine

**Difficulté** : 🟡🟡🟡⚪⚪ (Intermédiaire)
**Prérequis** : Python, bases des LLMs
**Temps de lecture** : ~100 minutes

---

## 2. Tokenization Character-Level {#2-character-level}

### 2.1 Principe

**Idée** : Chaque caractère = 1 token.

```
"Hello" → ['H', 'e', 'l', 'l', 'o']
Token IDs: [72, 101, 108, 108, 111]
```

### 2.2 Implémentation

```python
class CharTokenizer:
    """
    Tokenizer caractère par caractère.
    """
    def __init__(self):
        # Vocabulaire : tous les caractères ASCII imprimables
        self.chars = sorted(list(set(chr(i) for i in range(32, 127))))
        self.vocab_size = len(self.chars)
        self.char_to_id = {ch: i for i, ch in enumerate(self.chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text: str) -> list[int]:
        """Convertit texte en IDs."""
        return [self.char_to_id.get(ch, 0) for ch in text]

    def decode(self, ids: list[int]) -> str:
        """Convertit IDs en texte."""
        return ''.join([self.id_to_char.get(i, '?') for i in ids])


# Test
tokenizer = CharTokenizer()
text = "Hello, World!"
ids = tokenizer.encode(text)
reconstructed = tokenizer.decode(ids)

print(f"Vocab size: {tokenizer.vocab_size}")  # 95
print(f"Original: {text}")
print(f"Token IDs: {ids}")
print(f"Decoded: {reconstructed}")
```

### 2.3 Avantages et Inconvénients

**✅ Avantages** :
- Vocabulaire minimal (~100 tokens)
- Aucun OOV (tous les caractères supportés)
- Simple à implémenter

**❌ Inconvénients** :
- Séquences très longues (×5-10 vs subword)
- Modèle doit apprendre orthographe ("c-a-t" → "cat")
- Attention O(n²) coûteuse
- Perte de structure morphologique

**Verdict** : Rarement utilisé en 2024, sauf cas spéciaux (ByT5 pour langues avec peu de données).

---

## 3. Tokenization Word-Level {#3-word-level}

### 3.1 Principe

**Idée** : Chaque mot = 1 token.

```
"Hello, world!" → ["Hello", ",", "world", "!"]
```

### 3.2 Implémentation Simple

```python
import re
from collections import Counter

class WordTokenizer:
    """
    Tokenizer mot par mot avec vocabulaire fixe.
    """
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.word_to_id = {}
        self.id_to_word = {}
        self.unk_token = "<UNK>"
        self.unk_id = 0

    def fit(self, corpus: list[str]):
        """
        Construit le vocabulaire à partir d'un corpus.
        """
        # Tokenisation simple par espaces et ponctuation
        all_words = []
        for text in corpus:
            words = re.findall(r'\w+|[^\w\s]', text.lower())
            all_words.extend(words)

        # Compter les fréquences
        word_counts = Counter(all_words)

        # Prendre les N plus fréquents
        most_common = word_counts.most_common(self.vocab_size - 1)

        # Construire vocabulaire
        self.word_to_id = {self.unk_token: self.unk_id}
        self.id_to_word = {self.unk_id: self.unk_token}

        for i, (word, _) in enumerate(most_common, start=1):
            self.word_to_id[word] = i
            self.id_to_word[i] = word

        print(f"Vocabulaire construit: {len(self.word_to_id)} mots")

    def encode(self, text: str) -> list[int]:
        """Encode un texte."""
        words = re.findall(r'\w+|[^\w\s]', text.lower())
        return [self.word_to_id.get(w, self.unk_id) for w in words]

    def decode(self, ids: list[int]) -> str:
        """Décode des IDs."""
        words = [self.id_to_word.get(i, self.unk_token) for i in ids]
        return ' '.join(words)


# Test
corpus = [
    "The cat sat on the mat.",
    "The dog sat on the log.",
    "Cats and dogs are animals."
]

tokenizer = WordTokenizer(vocab_size=50)
tokenizer.fit(corpus)

text = "The cat and dog played."
ids = tokenizer.encode(text)
decoded = tokenizer.decode(ids)

print(f"Original: {text}")
print(f"IDs: {ids}")
print(f"Decoded: {decoded}")
# "played" n'est pas dans vocab → <UNK>
```

### 3.3 Problèmes du Word-Level

**Exemple** : Mots composés en allemand
```
"Donaudampfschifffahrtsgesellschaft" (Compagnie de bateaux à vapeur du Danube)
→ <UNK> (mot inconnu !)
```

**Variants orthographiques** :
```
"run", "running", "runs", "ran", "runner"
→ 5 tokens différents (pas de partage d'informations)
```

**Verdict** : Obsolète pour LLMs modernes. Remplacé par subword tokenization.

---

## 4. Subword Tokenization : Le Standard Moderne {#4-subword}

### 4.1 Philosophie

**Idée clé** : Découper les mots en **unités sous-lexicales** fréquentes.

**Exemple** :
```
"unhappiness" → ["un", "happiness"]
                ou ["un", "happi", "ness"]
```

**Avantages** :
- ✅ Vocabulaire gérable (30k-100k)
- ✅ Zéro OOV (tout mot décomposable en caractères)
- ✅ Partage morphologique ("happy", "unhappy", "happiness")
- ✅ Séquences raisonnables (3-5× moins longues que caractères)

### 4.2 Les 4 Algorithmes Majeurs

| Algorithme | Principe | Modèles | Année |
|------------|----------|---------|-------|
| **BPE** | Merge pairs fréquentes (bottom-up) | GPT-2, GPT-3, RoBERTa | 2015 |
| **WordPiece** | Merge par maximum likelihood | BERT, DistilBERT | 2016 |
| **Unigram LM** | Probabilités de sous-mots (top-down) | ALBERT, T5, mBART | 2018 |
| **SentencePiece** | Language-agnostic (UTF-8 bytes) | XLM-R, T5, LLaMA | 2018 |

---

## 5. Byte Pair Encoding (BPE) {#5-bpe}

### 5.1 Algorithme

**Étapes** :

1. **Initialisation** : Vocabulaire = tous les caractères + `</w>` (end-of-word)
2. **Itération** :
   - Compter toutes les paires adjacentes de tokens
   - Merger la paire la plus fréquente
   - Ajouter le nouveau token au vocabulaire
3. **Répéter** jusqu'à atteindre la taille de vocabulaire souhaitée

### 5.2 Exemple Pas à Pas

**Corpus** :
```
"low" : 5 fois
"lower" : 2 fois
"newest" : 6 fois
"widest" : 3 fois
```

**Initialisation** :
```
Vocabulaire: {l, o, w, e, r, n, s, t, i, d, </w>}
Corpus tokenisé:
  low</w> : 5
  lower</w> : 2
  newest</w> : 6
  widest</w> : 3
```

**Itération 1** : Paire la plus fréquente = "e" + "s" (6+3=9 occurrences)
```
Merge: es
Vocabulaire: {l, o, w, e, r, n, s, t, i, d, </w>, es}
Corpus:
  low</w> : 5
  lower</w> : 2
  newest</w> : 6   → n ew es t </w>
  widest</w> : 3   → w id es t </w>
```

**Itération 2** : Paire la plus fréquente = "es" + "t" (9 occurrences)
```
Merge: est
Vocabulaire: {..., es, est}
Corpus:
  newest</w> : 6   → n ew est </w>
  widest</w> : 3   → w id est </w>
```

**Continuer** jusqu'à vocab_size = 1000 (ou autre cible).

### 5.3 Implémentation BPE

```python
from collections import Counter, defaultdict
import re

class BPETokenizer:
    """
    Byte Pair Encoding tokenizer (simplifié).
    """
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.vocab = set()
        self.merges = {}  # (pair) -> merged_token

    def get_vocab(self, corpus):
        """Extrait le vocabulaire de caractères."""
        vocab = Counter()
        for word in corpus:
            vocab[' '.join(word) + ' </w>'] += 1
        return vocab

    def get_stats(self, vocab):
        """Compte les paires de tokens adjacents."""
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs

    def merge_vocab(self, pair, vocab):
        """Merge une paire dans le vocabulaire."""
        new_vocab = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)

        for word in vocab:
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = vocab[word]

        return new_vocab

    def train(self, corpus):
        """
        Entraîne le BPE tokenizer.

        Args:
            corpus: Liste de mots (strings)
        """
        # Initialiser vocabulaire avec caractères
        vocab = self.get_vocab(corpus)

        # Itérer jusqu'à atteindre vocab_size
        for i in range(self.vocab_size - 256):  # 256 = caractères de base
            pairs = self.get_stats(vocab)
            if not pairs:
                break

            # Trouver la paire la plus fréquente
            best_pair = max(pairs, key=pairs.get)

            # Merger
            vocab = self.merge_vocab(best_pair, vocab)
            self.merges[best_pair] = ''.join(best_pair)

            if (i + 1) % 100 == 0:
                print(f"Merge {i+1}: {best_pair} → {''.join(best_pair)}")

        # Construire vocabulaire final
        self.vocab = set()
        for word in vocab:
            self.vocab.update(word.split())

        print(f"BPE training complete. Vocab size: {len(self.vocab)}")

    def encode(self, word):
        """Encode un mot avec BPE."""
        word = ' '.join(word) + ' </w>'
        symbols = word.split()

        while len(symbols) > 1:
            # Trouver la première paire qui existe dans merges
            pairs = [(symbols[i], symbols[i+1]) for i in range(len(symbols)-1)]
            valid_pairs = [p for p in pairs if p in self.merges]

            if not valid_pairs:
                break

            # Prendre la première paire valide
            bigram = valid_pairs[0]
            first, second = bigram

            # Merger
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == first and symbols[i+1] == second:
                    new_symbols.append(first + second)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1

            symbols = new_symbols

        return symbols


# Test
corpus = ["low", "lower", "newest", "widest"] * 10  # Répéter pour fréquences
tokenizer = BPETokenizer(vocab_size=300)
tokenizer.train(corpus)

# Encoder un mot
word = "lowest"
tokens = tokenizer.encode(word)
print(f"\n'{word}' → {tokens}")
```

### 5.4 BPE en Production (GPT-2)

```python
from transformers import GPT2Tokenizer

# Tokenizer GPT-2 (50k vocab, BPE)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

text = "Hello, world! How are you doing today?"
tokens = tokenizer.tokenize(text)
ids = tokenizer.encode(text)

print(f"Text: {text}")
print(f"Tokens: {tokens}")
# ['Hello', ',', 'Ġworld', '!', 'ĠHow', 'Ġare', 'Ġyou', 'Ġdoing', 'Ġtoday', '?']
# Ġ = espace (représenté par un caractère spécial)

print(f"IDs: {ids}")
# [15496, 11, 995, 0, 1374, 389, 345, 1804, 1909, 30]

# Décoder
decoded = tokenizer.decode(ids)
print(f"Decoded: {decoded}")
```

### 💡 Analogie : Le Puzzle

Imaginez que les mots sont des puzzles :
- **Caractères** : 1000 pièces minuscules (long à assembler)
- **Mots complets** : Puzzles pré-assemblés (mais millions de puzzles différents)
- **Subwords (BPE)** : Sections du puzzle (10-50 pièces) réutilisables

BPE trouve les "sections" les plus communes et les assemble intelligemment !

---

## 6. WordPiece (BERT) {#6-wordpiece}

### 6.1 Différence avec BPE

**BPE** : Merge la paire **la plus fréquente**
**WordPiece** : Merge la paire qui **maximise la vraisemblance** du corpus

**Score WordPiece** :
```
score(pair) = freq(pair) / (freq(first) × freq(second))
```

**Intuition** : Favorise les paires qui apparaissent ensemble **plus souvent que par hasard**.

### 6.2 Tokens Spéciaux BERT

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokens spéciaux
print(tokenizer.special_tokens_map)
# {
#   'unk_token': '[UNK]',
#   'sep_token': '[SEP]',
#   'pad_token': '[PAD]',
#   'cls_token': '[CLS]',
#   'mask_token': '[MASK]'
# }

# Tokenization
text = "Hello, how are you?"
tokens = tokenizer.tokenize(text)
print(f"Tokens: {tokens}")
# ['hello', ',', 'how', 'are', 'you', '?']

# Avec tokens spéciaux
encoded = tokenizer.encode(text, add_special_tokens=True)
print(f"Encoded: {encoded}")
# [101, 7592, 1010, 2129, 2024, 2017, 1029, 102]
#  [CLS]  hello   ,    how   are   you    ?   [SEP]

# Décodage
decoded = tokenizer.decode(encoded)
print(f"Decoded: {decoded}")
# [CLS] hello, how are you? [SEP]
```

### 6.3 Préfixe "##" pour Continuation

**Convention BERT** : Subwords non-initiaux ont préfixe "##"

```python
text = "unhappiness"
tokens = tokenizer.tokenize(text)
print(tokens)
# ['un', '##hap', '##pi', '##ness']
#         ^^       ^^       ^^
#    Signifie: continuation du mot précédent
```

**Utilité** : Distinguer "un happy" (deux mots) de "unhappy" (un seul mot).

---

## 7. SentencePiece & Unigram {#7-sentencepiece}

### 7.1 SentencePiece : Language-Agnostic

**Problème** : BPE et WordPiece supposent que les espaces séparent les mots (faux pour chinois, japonais, thaï, etc.).

**Solution SentencePiece** :
1. Traiter le texte comme une séquence de **bytes UTF-8**
2. Pas de pré-tokenization
3. Apprend directement depuis les caractères Unicode

**Exemple** :
```python
import sentencepiece as spm

# Entraîner un modèle SentencePiece
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='m',
    vocab_size=8000,
    character_coverage=0.9995,  # Couverture des caractères Unicode
    model_type='bpe'  # ou 'unigram'
)

# Charger et utiliser
sp = spm.SentencePieceProcessor(model_file='m.model')

text = "Hello, world! 你好世界"
tokens = sp.encode(text, out_type=str)
print(f"Tokens: {tokens}")
# ['▁Hello', ',', '▁world', '!', '▁', '你', '好', '世', '界']
# ▁ = espace (symbole Unicode)

ids = sp.encode(text, out_type=int)
print(f"IDs: {ids}")

decoded = sp.decode(ids)
print(f"Decoded: {decoded}")
```

### 7.2 Unigram Language Model

**Algorithme** (top-down, inverse de BPE) :

1. **Initialisation** : Vocabulaire très large (tous les substrings fréquents)
2. **Itération** :
   - Calculer la perte (loss) du modèle avec vocab actuel
   - Retirer le token qui augmente le moins la perte
3. **Répéter** jusqu'à atteindre vocab_size

**Formule** :
```
P(texte) = ∏ P(token_i)
```

**Avantage** : Tokenization **probabiliste** (plusieurs découpage possibles).

**Exemple** :
```
"unhappiness" peut être tokenisé comme:
- ["un", "happiness"] avec probabilité 0.6
- ["un", "happi", "ness"] avec probabilité 0.3
- ["unhappiness"] avec probabilité 0.1
```

**Modèles** : T5, ALBERT, mBART (via SentencePiece)

---

## 8. Tiktoken (GPT-3/4) {#8-tiktoken}

### 8.1 Spécificités

**Tiktoken** (OpenAI) est une bibliothèque de tokenization optimisée pour GPT-3.5/4.

**Différences** :
- Basé sur BPE
- Optimisé pour **vitesse** (Rust backend)
- Gère mieux le **code** (patterns spéciaux pour Python, JS, etc.)
- Vocabulaire ~100k tokens (vs 50k pour GPT-2)

### 8.2 Utilisation

```python
import tiktoken

# Charger le tokenizer GPT-4
encoding = tiktoken.encoding_for_model("gpt-4")

text = "Hello, world! Let's tokenize this text."
tokens = encoding.encode(text)
print(f"Tokens: {tokens}")
# [9906, 11, 1917, 0, 6914, 596, 4037, 553, 420, 1495, 13]

# Nombre de tokens
print(f"Number of tokens: {len(tokens)}")

# Décoder
decoded = encoding.decode(tokens)
print(f"Decoded: {decoded}")

# Afficher les tokens en string
for token in tokens:
    print(f"{token} → {encoding.decode([token])!r}")
```

### 8.3 Comptage de Tokens pour Tarification

```python
def count_tokens(text, model="gpt-4"):
    """
    Compte le nombre de tokens (pour estimer le coût).
    """
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Exemple
article = "..." * 1000  # Long article
num_tokens = count_tokens(article, model="gpt-4")
cost_per_1k = 0.03  # $0.03 / 1k tokens (GPT-4 input)
estimated_cost = (num_tokens / 1000) * cost_per_1k

print(f"Tokens: {num_tokens:,}")
print(f"Estimated cost: ${estimated_cost:.4f}")
```

---

## 9. Tokenizers Multilingues {#9-multilingue}

### 9.1 Défis du Multilingue

**Problème** : Langues différentes ont fréquences différentes.

**Exemple** : Corpus 90% anglais, 10% chinois
- Vocabulaire dominé par tokens anglais
- Texte chinois sur-fragmenté (chaque caractère = token)
- Inefficacité et perte de performance

### 9.2 Solutions

#### A) SentencePiece avec Character Coverage

```python
spm.SentencePieceTrainer.train(
    input='multilingual_corpus.txt',
    model_prefix='multilingual',
    vocab_size=250000,  # Plus grand pour couvrir plusieurs langues
    character_coverage=0.9995,  # 99.95% des caractères Unicode
    model_type='unigram'
)
```

#### B) XLM-RoBERTa (Facebook)

**Stratégie** :
- Entraîné sur 100 langues
- Vocab size: 250k (vs 50k pour BERT)
- SentencePiece Unigram
- Échantillonnage pondéré par langue

```python
from transformers import XLMRobertaTokenizer

tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

# Test multilingue
texts = [
    "Hello, how are you?",  # Anglais
    "Bonjour, comment allez-vous ?",  # Français
    "你好，你好吗？",  # Chinois
    "مرحبا، كيف حالك؟"  # Arabe
]

for text in texts:
    tokens = tokenizer.tokenize(text)
    print(f"{text}")
    print(f"  → {tokens}\n")
```

### 9.3 Métriques de Qualité Multilingue

**Fertility** : Nombre moyen de tokens par mot

```python
def calculate_fertility(tokenizer, texts, language):
    """
    Mesure l'efficacité du tokenizer pour une langue.
    """
    total_words = 0
    total_tokens = 0

    for text in texts:
        words = text.split()
        tokens = tokenizer.tokenize(text)
        total_words += len(words)
        total_tokens += len(tokens)

    fertility = total_tokens / total_words if total_words > 0 else 0
    print(f"{language}: {fertility:.2f} tokens/word")
    return fertility

# Test
en_texts = ["Hello world", "How are you", "Good morning"]
zh_texts = ["你好世界", "你好吗", "早上好"]

fertility_en = calculate_fertility(tokenizer, en_texts, "English")
fertility_zh = calculate_fertility(tokenizer, zh_texts, "Chinese")

# Idéal: fertility similaire entre langues (~1.2-1.5)
```

---

## 10. Problèmes et Limitations {#10-problemes}

### 10.1 Tokenization Inconsistente

**Problème** : Espaces et casse affectent la tokenization.

```python
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Avec/sans espace
print(tokenizer.tokenize("Hello"))      # ['Hello']
print(tokenizer.tokenize(" Hello"))     # ['ĠHello']  (différent!)

# Casse
print(tokenizer.tokenize("HELLO"))      # ['HE', 'LLO']
print(tokenizer.tokenize("hello"))      # ['hello']
```

**Conséquence** : Modèle peut être sensible à des variations triviales !

### 10.2 Mots Rares Sur-Fragmentés

```python
# Mot rare (nom propre)
rare_word = "Grzybowski"
tokens = tokenizer.tokenize(rare_word)
print(tokens)
# ['G', 'rzy', 'bow', 'ski']  (4 tokens pour 1 mot!)

# Mot commun
common_word = "computer"
tokens = tokenizer.tokenize(common_word)
print(tokens)
# ['computer']  (1 token)
```

**Impact** : Noms propres, termes techniques fragmentés → contexte limité.

### 10.3 Le Problème "SolidGoldMagikarp"

**Anecdote** : Token 41,523 dans GPT-2 = "SolidGoldMagikarp" (nom d'utilisateur Reddit rare).

**Problème** :
- Token existe dans vocab, mais **presque jamais vu durant l'entraînement**
- Embedding non initialisé/random
- Comportement bizarre du modèle quand ce token apparaît

**Test** :
```python
# Demander à GPT-3 de définir "SolidGoldMagikarp"
# Résultat: refus, erreurs, outputs incohérents
```

**Leçon** : Vocab size ≠ tokens utilisables. Certains tokens sont "glitch".

### 10.4 Biais de Tokenization

**Exemple** : Noms africains vs européens

```python
names_european = ["Smith", "Johnson", "Williams"]
names_african = ["Adebayo", "Okonkwo", "Nkrumah"]

for name in names_european:
    print(f"{name}: {tokenizer.tokenize(name)}")
# Smith: ['Smith']
# Johnson: ['Johnson']
# Williams: ['Williams']

for name in names_african:
    print(f"{name}: {tokenizer.tokenize(name)}")
# Adebayo: ['Ade', 'bay', 'o']
# Okonkwo: ['Ok', 'onk', 'wo']
# Nkrumah: ['N', 'k', 'rum', 'ah']
```

**Conséquence** :
- Noms africains utilisent 3× plus de tokens
- Coût d'API 3× plus élevé
- Contexte disponible réduit
- Biais potentiel du modèle

**Solution** : Vocabulaire équilibré, ou fine-tuning avec données diversifiées.

### 10.5 Trailing Spaces

```python
# Espace final change la tokenization!
text1 = "Hello world"
text2 = "Hello world "  # Espace final

print(tokenizer.encode(text1))
print(tokenizer.encode(text2))
# Différents!
```

**Piège** : Prompts avec/sans espace final donnent résultats différents.

---

## 11. Implémentation from Scratch {#11-implementation}

### 11.1 BPE Complet

```python
import re
from collections import Counter, defaultdict

class SimpleBPETokenizer:
    """
    Implémentation complète de BPE from scratch.
    """
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.token_to_id = {}
        self.id_to_token = {}
        self.merges = []

    def _get_stats(self, words):
        """Compte les paires de tokens."""
        pairs = defaultdict(int)
        for word, freq in words.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def _merge_pair(self, pair, words):
        """Merge une paire dans le vocabulaire."""
        new_words = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)

        for word, freq in words.items():
            new_word = word.replace(bigram, replacement)
            new_words[new_word] = freq

        return new_words

    def train(self, corpus):
        """
        Entraîne le tokenizer BPE.

        Args:
            corpus: Liste de phrases (strings)
        """
        # 1. Pré-tokenization : découper en mots
        word_freqs = Counter()
        for text in corpus:
            words = re.findall(r'\w+', text.lower())
            word_freqs.update(words)

        # 2. Convertir en format BPE (caractères séparés)
        words = {}
        for word, freq in word_freqs.items():
            words[' '.join(word) + ' </w>'] = freq

        # 3. Vocabulaire initial : caractères
        vocab = set()
        for word in words:
            vocab.update(word.split())

        # 4. Itérer pour créer les merges
        while len(vocab) < self.vocab_size:
            pairs = self._get_stats(words)
            if not pairs:
                break

            # Trouver la paire la plus fréquente
            best_pair = max(pairs, key=pairs.get)

            # Merger
            words = self._merge_pair(best_pair, words)
            self.merges.append(best_pair)

            # Ajouter au vocabulaire
            vocab.add(''.join(best_pair))

        # 5. Construire token_to_id et id_to_token
        for i, token in enumerate(sorted(vocab)):
            self.token_to_id[token] = i
            self.id_to_token[i] = token

        print(f"Training complete. Vocab size: {len(self.token_to_id)}")

    def encode(self, text):
        """
        Encode un texte en token IDs.
        """
        words = re.findall(r'\w+', text.lower())
        tokens = []

        for word in words:
            # Initialiser avec caractères
            word_tokens = list(word) + ['</w>']

            # Appliquer les merges
            for merge in self.merges:
                i = 0
                while i < len(word_tokens) - 1:
                    if (word_tokens[i], word_tokens[i + 1]) == merge:
                        # Merger
                        word_tokens[i:i+2] = [''.join(merge)]
                    else:
                        i += 1

            tokens.extend(word_tokens)

        # Convertir en IDs
        ids = [self.token_to_id.get(t, 0) for t in tokens]
        return ids

    def decode(self, ids):
        """
        Décode des token IDs en texte.
        """
        tokens = [self.id_to_token.get(i, '<UNK>') for i in ids]
        text = ''.join(tokens).replace('</w>', ' ')
        return text.strip()


# Test complet
corpus = [
    "The cat sat on the mat.",
    "The cat ate the fish.",
    "The dog sat on the log.",
    "A quick brown fox jumps over the lazy dog."
] * 100  # Répéter pour avoir des fréquences

tokenizer = SimpleBPETokenizer(vocab_size=500)
tokenizer.train(corpus)

# Test encoding/decoding
test_text = "The cat sat on the mat"
ids = tokenizer.encode(test_text)
decoded = tokenizer.decode(ids)

print(f"\nOriginal: {test_text}")
print(f"Token IDs: {ids}")
print(f"Decoded: {decoded}")

# Inspecter quelques merges
print(f"\nFirst 10 merges:")
for i, merge in enumerate(tokenizer.merges[:10], 1):
    print(f"{i}. {merge[0]} + {merge[1]} → {merge[0]}{merge[1]}")
```

---

## 12. Quiz Interactif {#12-quiz}

### Question 1 : Vocabulaire BPE

**Pourquoi BPE utilise-t-il un vocabulaire de ~50k tokens au lieu de 1M mots ?**

A) 50k est plus facile à mémoriser
B) Réduire la taille des embeddings et éliminer OOV
C) Accélérer l'entraînement
D) C'est une contrainte GPU

<details>
<summary>Voir la réponse</summary>

**Réponse : B) Réduire la taille des embeddings et éliminer OOV**

**Explications** :
- 1M mots → embedding matrix de 1M × 768 = 768M paramètres (énorme!)
- Beaucoup de mots rares jamais vus (OOV)
- 50k subwords → tout mot peut être décomposé → zéro OOV
- Taille embeddings raisonnable : 50k × 768 = 38M params

**Bonus** : Partage morphologique ("happy", "unhappy", "happiness" partagent "happi").
</details>

---

### Question 2 : BPE vs WordPiece

**Quelle est la différence principale entre BPE et WordPiece ?**

A) BPE est plus rapide
B) WordPiece utilise maximum likelihood, BPE utilise fréquence
C) BPE est pour GPT, WordPiece pour BERT (pas de différence algorithme)
D) WordPiece gère mieux le multilingue

<details>
<summary>Voir la réponse</summary>

**Réponse : B) WordPiece utilise maximum likelihood, BPE utilise fréquence**

**BPE** : Merge la paire la plus **fréquente**
```
score = freq(pair)
```

**WordPiece** : Merge la paire qui maximise la **vraisemblance**
```
score = freq(pair) / (freq(first) × freq(second))
```

**Intuition** : WordPiece favorise paires qui apparaissent ensemble plus que par hasard.
</details>

---

### Question 3 : Trailing Spaces

**Pourquoi "Hello" et " Hello" donnent-ils des tokens différents ?**

A) Bug du tokenizer
B) L'espace est traité comme un token spécial
C) Pour distinguer début de phrase vs milieu
D) C'est un design choice (espaces font partie du token)

<details>
<summary>Voir la réponse</summary>

**Réponse : D) C'est un design choice (espaces font partie du token)**

Dans GPT-2, l'espace est **encodé dans le token** (Ġ = espace en début).

**Exemple** :
- "Hello" → token "Hello" (ID: 15496)
- " Hello" → token "ĠHello" (ID: 18435) différent!

**Raison** : Permet de distinguer "new york" (deux mots) de "newyork" (un mot).

**Piège** : Prompts doivent être cohérents avec/sans espaces !
</details>

---

### Question 4 : Multilingue

**Pourquoi XLM-RoBERTa a un vocabulaire de 250k (vs 50k pour BERT) ?**

A) Erreur de conception
B) Pour couvrir 100 langues avec suffisamment de tokens par langue
C) Plus de tokens = modèle plus intelligent
D) Pour supporter les emojis

<details>
<summary>Voir la réponse</summary>

**Réponse : B) Pour couvrir 100 langues avec suffisamment de tokens par langue**

Avec 50k tokens pour 100 langues → ~500 tokens/langue (insuffisant !).
Avec 250k tokens → ~2500 tokens/langue (raisonnable).

**Problème** : Si vocab trop petit pour multilingue, langues rares sont sur-fragmentées (inefficace).
</details>

---

### Question 5 : SolidGoldMagikarp

**Quel est le problème avec les tokens ultra-rares comme "SolidGoldMagikarp" ?**

A) Ils prennent trop de mémoire
B) Leur embedding n'est pas bien entraîné (peu d'exemples)
C) Ils causent des bugs GPU
D) Ils sont illégaux

<details>
<summary>Voir la réponse</summary>

**Réponse : B) Leur embedding n'est pas bien entraîné (peu d'exemples)**

Si un token apparaît 1 fois dans 300B tokens d'entraînement :
- Son embedding n'est presque jamais mis à jour
- Reste proche de l'initialisation random
- Comportement imprévisible/bizarre du modèle

**Exemple réel** : GPT-3 refuse de définir "SolidGoldMagikarp" ou donne sorties incohérentes.

**Leçon** : Vocab size ≠ tokens utilisables. Filtrer tokens ultra-rares.
</details>

---

### Question 6 : Tiktoken vs GPT-2

**Pourquoi Tiktoken (GPT-4) a ~100k vocab vs 50k pour GPT-2 ?**

A) Meilleure gestion du **code source** (Python, JS)
B) Plus de langues supportées
C) Réduire le nombre de tokens par texte (économies)
D) Toutes les réponses

<details>
<summary>Voir la réponse</summary>

**Réponse : D) Toutes les réponses**

**100k vocab permet** :
- Tokens spéciaux pour code (indentation, keywords)
- Meilleure couverture multilingue
- Moins de tokens par texte → contexte plus long, coût réduit

**Exemple** : Code Python "def function():"
- GPT-2 (50k): 5 tokens
- GPT-4 (100k): 3 tokens
</details>

---

## 13. Exercices Pratiques {#13-exercices}

### Exercice 1 : Comparer Tokenizers

**Objectif** : Évaluer fertility (tokens/word) pour différents tokenizers.

```python
from transformers import GPT2Tokenizer, BertTokenizer, XLMRobertaTokenizer

def compare_tokenizers(text):
    """
    Compare 3 tokenizers sur le même texte.
    """
    tokenizers = {
        "GPT-2": GPT2Tokenizer.from_pretrained("gpt2"),
        "BERT": BertTokenizer.from_pretrained("bert-base-uncased"),
        "XLM-R": XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    }

    # TODO: Pour chaque tokenizer:
    # 1. Tokeniser le texte
    # 2. Compter tokens
    # 3. Calculer fertility (tokens / nombre de mots)
    # 4. Afficher résultats

    pass

# Test
text = "The quick brown fox jumps over the lazy dog."
compare_tokenizers(text)
```

<details>
<summary>Voir la solution</summary>

```python
def compare_tokenizers(text):
    """
    Compare 3 tokenizers.
    """
    tokenizers = {
        "GPT-2": GPT2Tokenizer.from_pretrained("gpt2"),
        "BERT": BertTokenizer.from_pretrained("bert-base-uncased"),
        "XLM-R": XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    }

    words = text.split()
    num_words = len(words)

    print(f"Text: {text}")
    print(f"Words: {num_words}\n")

    for name, tokenizer in tokenizers.items():
        tokens = tokenizer.tokenize(text)
        num_tokens = len(tokens)
        fertility = num_tokens / num_words

        print(f"{name}:")
        print(f"  Tokens: {tokens}")
        print(f"  Count: {num_tokens}")
        print(f"  Fertility: {fertility:.2f}\n")

# Test multilingue
texts = {
    "English": "Hello, how are you doing today?",
    "French": "Bonjour, comment allez-vous aujourd'hui ?",
    "Chinese": "你好，你今天好吗？",
    "Arabic": "مرحبا، كيف حالك اليوم؟"
}

for lang, text in texts.items():
    print(f"=== {lang} ===")
    compare_tokenizers(text)
    print()
```
</details>

---

### Exercice 2 : Détecter Tokenization Bias

**Objectif** : Mesurer le biais entre noms européens vs africains.

```python
def measure_tokenization_bias(tokenizer, names_group1, names_group2):
    """
    Compare fertility moyenne entre deux groupes de noms.
    """
    # TODO:
    # 1. Tokeniser tous les noms de chaque groupe
    # 2. Calculer fertility moyenne par groupe
    # 3. Calculer le ratio (bias factor)
    # 4. Afficher résultats et interprétation
    pass

# Données
european_names = ["Smith", "Johnson", "Williams", "Brown", "Jones"]
african_names = ["Adebayo", "Okonkwo", "Nkrumah", "Mbeki", "Tutu"]

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
measure_tokenization_bias(tokenizer, european_names, african_names)
```

<details>
<summary>Voir la solution</summary>

```python
def measure_tokenization_bias(tokenizer, names_group1, names_group2,
                              label1="Group 1", label2="Group 2"):
    """
    Mesure le biais de tokenization entre deux groupes.
    """
    def avg_fertility(names):
        total_tokens = 0
        for name in names:
            tokens = tokenizer.tokenize(name)
            total_tokens += len(tokens)
        return total_tokens / len(names)

    fertility1 = avg_fertility(names_group1)
    fertility2 = avg_fertility(names_group2)
    bias_ratio = fertility2 / fertility1

    print(f"{label1} average fertility: {fertility1:.2f} tokens/name")
    print(f"{label2} average fertility: {fertility2:.2f} tokens/name")
    print(f"Bias ratio: {bias_ratio:.2f}x")

    if bias_ratio > 1.5:
        print("⚠️ Significant bias detected!")
    elif bias_ratio > 1.2:
        print("⚠️ Moderate bias detected")
    else:
        print("✅ Low bias")

    # Détails
    print(f"\n{label1} examples:")
    for name in names_group1[:3]:
        print(f"  {name}: {tokenizer.tokenize(name)}")

    print(f"\n{label2} examples:")
    for name in names_group2[:3]:
        print(f"  {name}: {tokenizer.tokenize(name)}")

# Test
european_names = ["Smith", "Johnson", "Williams", "Brown", "Jones"]
african_names = ["Adebayo", "Okonkwo", "Nkrumah", "Mbeki", "Tutu"]

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
measure_tokenization_bias(tokenizer, european_names, african_names,
                          "European", "African")
```
</details>

---

### Exercice 3 : Optimiser Vocabulaire pour un Domaine

**Objectif** : Entraîner un tokenizer BPE spécialisé pour du code Python.

```python
# Corpus de code Python
code_corpus = [
    "def hello_world():\n    print('Hello, World!')",
    "import numpy as np\nimport pandas as pd",
    "for i in range(10):\n    print(i)",
    # ... ajouter 1000+ exemples
]

# TODO:
# 1. Entraîner un tokenizer BPE avec tokenizers library
# 2. Comparer avec GPT-2 tokenizer
# 3. Mesurer:
#    - Nombre de tokens pour un fichier Python
#    - Coverage des keywords Python (def, import, for, etc.)
#    - Fertility
```

<details>
<summary>Voir la solution</summary>

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# 1. Créer et entraîner un BPE tokenizer
def train_code_tokenizer(corpus, vocab_size=5000):
    """
    Entraîne un tokenizer BPE spécialisé pour code Python.
    """
    # Initialiser BPE tokenizer
    tokenizer = Tokenizer(models.BPE())

    # Pre-tokenizer (ne pas merger à travers whitespace/ponctuation)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    )

    # Entraîner sur le corpus
    tokenizer.train_from_iterator(corpus, trainer)

    return tokenizer

# 2. Corpus de code Python
code_corpus = [
    "def hello_world():\n    print('Hello, World!')",
    "import numpy as np",
    "for i in range(10):\n    print(i)",
    "class MyClass:\n    def __init__(self):\n        pass",
    "if __name__ == '__main__':\n    main()"
] * 200  # Répéter pour avoir assez de données

# 3. Entraîner
custom_tokenizer = train_code_tokenizer(code_corpus, vocab_size=2000)

# 4. Comparer avec GPT-2
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

test_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

# Tokeniser avec les deux
custom_tokens = custom_tokenizer.encode(test_code).tokens
gpt2_tokens = gpt2_tokenizer.tokenize(test_code)

print(f"Custom tokenizer: {len(custom_tokens)} tokens")
print(f"GPT-2 tokenizer: {len(gpt2_tokens)} tokens")
print(f"Improvement: {len(gpt2_tokens) / len(custom_tokens):.2f}x fewer tokens")

print(f"\nCustom tokens: {custom_tokens[:20]}")
print(f"GPT-2 tokens: {gpt2_tokens[:20]}")
```
</details>

---

## 14. Conclusion {#14-conclusion}

### 🎭 Dialogue Final : L'Interface Invisible

**Alice** : Finalement, la tokenization est plus importante que je pensais !

**Bob** : Absolument. C'est **l'interface** entre nous (humains avec langage) et le modèle (mathématiques). Une mauvaise tokenization = modèle handicapé.

**Alice** : Quels sont les choix clés ?

**Bob** :
1. **Méthode** : BPE (GPT), WordPiece (BERT), Unigram (T5)
2. **Vocab size** : 30k-100k (trade-off séquence length vs expressivité)
3. **Corpus** : Représentatif du domaine cible
4. **Multilingue** : SentencePiece + character coverage élevé

**Alice** : Et les pièges ?

**Bob** :
- Tokens ultra-rares (SolidGoldMagikarp)
- Biais (noms africains sur-fragmentés)
- Inconsistances (espaces, casse)
- Domaine mismatch (tokenizer général sur code médical)

**Alice** : En 2026, vers quoi on va ?

**Bob** :
- **Vocabulaires plus grands** (100k-1M) grâce à hardware
- **Tokenization multimodale** (texte + images + audio)
- **Apprentissage end-to-end** (le modèle apprend sa propre tokenization)
- **Byte-level models** (ByT5, CANINE) : plus de tokenization du tout !

### 🎯 Points Clés à Retenir

| Concept | Essence |
|---------|---------|
| **Character-level** | Vocab minimal, séquences très longues → obsolète |
| **Word-level** | OOV problem, vocab explosion → obsolète |
| **Subword (BPE)** | Sweet spot : 30k-100k vocab, zéro OOV |
| **WordPiece** | BPE avec maximum likelihood (BERT) |
| **SentencePiece** | Language-agnostic, UTF-8 bytes (T5, LLaMA) |
| **Unigram** | Probabilistic, top-down (ALBERT, mBART) |
| **Tiktoken** | Optimisé vitesse + code (GPT-3.5/4) |

### 📊 Recommandations Pratiques

**Pour un nouveau projet** :
1. Partir d'un tokenizer pré-entraîné (GPT-2, BERT, T5)
2. Si domaine spécifique → fine-tune sur corpus domaine
3. Mesurer fertility et bias
4. Vocab size : 32k-50k (équilibre performance/coût)

**Red flags** :
- Fertility > 2.0 pour une langue (sur-fragmentation)
- Tokens ultra-rares (< 10 occurrences) dans le vocab
- Bias ratio > 2.0 entre groupes démographiques

---

## 15. Ressources {#15-ressources}

### 📚 Papers Fondamentaux

1. **"Neural Machine Translation of Rare Words with Subword Units"** (Sennrich et al., 2016)
   - Introduction du BPE en NLP

2. **"Google's Neural Machine Translation System"** (Wu et al., 2016)
   - WordPiece algorithm

3. **"SentencePiece: A simple and language independent approach"** (Kudo & Richardson, 2018)

4. **"Subword Regularization"** (Kudo, 2018)
   - Unigram Language Model

5. **"ByT5: Towards a token-free future with pre-trained byte-to-byte models"** (Xue et al., 2021)

### 🛠️ Bibliothèques

```bash
# HuggingFace Tokenizers (production)
pip install tokenizers transformers

# SentencePiece
pip install sentencepiece

# Tiktoken (OpenAI)
pip install tiktoken

# BPE from scratch (éducatif)
# https://github.com/karpathy/minbpe
```

### 🔗 Outils Interactifs

- **Tokenizer Playground** : https://platform.openai.com/tokenizer
- **HuggingFace Tokenizers Docs** : https://huggingface.co/docs/tokenizers
- **SentencePiece Demo** : https://github.com/google/sentencepiece

### 📖 Tutoriels

- **"Let's build the GPT Tokenizer"** (Andrej Karpathy) : https://www.youtube.com/watch?v=zduSFxRajkE
- **HuggingFace NLP Course - Tokenizers** : https://huggingface.co/course/chapter6

---

**🎓 Bravo !** Vous maîtrisez maintenant la tokenization, l'interface invisible entre langage et mathématiques. Prochain chapitre : **Fine-Tuning** pour adapter un LLM à votre tâche ! 🚀

