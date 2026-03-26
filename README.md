<div align="center">

# 🔁 Deep Learning — Part 3: Recurrent Neural Networks (RNN)

### Basics of RNN Implementation — Sentiment Classification & Hidden State Visualization

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.x-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![NumPy](https://img.shields.io/badge/NumPy-1.x-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

<br/>

> **A foundational deep learning notebook on Recurrent Neural Networks (RNN) — covering text preprocessing, embedding layers, SimpleRNN-based binary sentiment classification, and an advanced intermediate model to visualize hidden states at every timestep.**

</div>

---

## 📋 Table of Contents
- [About the Project](#-about-the-project)
- [Demo](#-demo)
- [Tech Stack](#-tech-stack)
- [Features](#-features)
- [Dataset](#-dataset)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Contributing](#-contributing)
- [License](#-license)

---

## 📌 About the Project

**Deep Learning — Part 3** introduces **Recurrent Neural Networks (RNN)** — the architecture designed for sequential data like text, time series, and speech.

This notebook builds a complete **binary sentiment classifier** on a custom 30-sentence dataset (positive vs. negative), covering every step from raw text to trained model — and then goes further by building an **intermediate inspection model** to visualize how the RNN's hidden state evolves at each token timestep.

| Section | What It Covers |
|---------|----------------|
| 📝 **Text Preprocessing** | Tokenization, vocabulary building, sequence padding |
| 🧩 **Embedding Layer** | Learnable word embeddings (`embed_dim=16`) |
| 🔁 **SimpleRNN Layer** | Recurrent processing with `rnn_units=8` |
| 🎯 **Binary Classification** | Sigmoid output — Positive (1) vs Negative (0) sentiment |
| 🔬 **Intermediate Model** | Extract & inspect embedding outputs + per-timestep hidden states |
| 📊 **Hidden State Visualization** | Print hidden state matrix `(timesteps × units)` for any sentence |

> 💡 The intermediate inspection model reuses the trained weights from the main model and reconstructs the RNN with `return_sequences=True` — allowing you to see exactly what the RNN "remembers" at each word in a sentence.

---

## 🌐 Demo

### 🎯 Main Model — Sentiment Classifier

**Architecture:**
```
Input (token ids, padded) → Embedding(2000, 16) → SimpleRNN(8) → Dense(1, Sigmoid)
```

**Training Config:**

| Parameter | Value |
|-----------|-------|
| Vocab Size | 2000 |
| Embedding Dim | 16 |
| RNN Units | 8 |
| Epochs | 25 |
| Batch Size | 8 |
| Optimizer | Adam |
| Loss | Binary Cross-Entropy |

---

### 🔬 Intermediate Inspection Model

After training, an **intermediate model** is built to expose internal RNN mechanics:

```python
# Output 1 — Embedding layer activations
model.get_layer('embed').output        # shape: (1, maxlen, 16)

# Output 2 — Hidden states at every timestep (return_sequences=True)
inspect_model.predict(example_seq)    # shape: (1, maxlen, 8)
```

**Example output for sentence `"I love this product"`:**
```
Sentence    : I love this product
Token ids   : [[3, 5, 2, 4, 0, 0, 0]]     ← padded to maxlen
Hidden states per timestep shape: (1, 7, 8)
Hidden states (timesteps × units):
[[ 0.123  -0.045   0.312 ... ]   ← after token "I"
 [ 0.256   0.178  -0.091 ... ]   ← after token "love"
 [ 0.489  -0.203   0.445 ... ]   ← after token "this"
 [ 0.521   0.312  -0.178 ... ]   ← after token "product"
 [ 0.521   0.312  -0.178 ... ]   ← padded (no change)
 ...
]
```

---

## 🛠️ Tech Stack

| Technology | Role |
|------------|------|
| **Python 3.10** | Core language |
| **TensorFlow / Keras** | Model building, training, Embedding, SimpleRNN, Dense layers |
| **Keras Tokenizer** | Text tokenization and vocabulary building |
| **Keras pad_sequences** | Sequence padding to uniform length |
| **Keras Functional API** | Intermediate model construction for hidden state inspection |
| **NumPy** | Array operations, label creation, rounding output |

---

## ✨ Features

<details open>
<summary><b>📝 Text Preprocessing Pipeline</b></summary>
<br/>

- **Custom 30-sentence dataset** — 15 positive + 15 negative sentiment sentences
- `Tokenizer(num_words=2000, oov_token="<OOV>")` — handles out-of-vocabulary words
- `texts_to_sequences()` — converts sentences to integer token ID sequences
- `pad_sequences(padding='post')` — pads all sequences to uniform `maxlen`
- Labels defined as `[1]*15 + [0]*15` → converted to NumPy array

</details>

<details open>
<summary><b>🧩 Embedding + SimpleRNN Architecture (Functional API)</b></summary>
<br/>

- Built using **Keras Functional API** (`Input → Embedding → SimpleRNN → Dense`)
- `Embedding(input_dim=2000, output_dim=16, mask_zero=True)` — masks padding tokens
- `SimpleRNN(units=8, return_sequences=False, return_state=False)` — returns final hidden state only
- `Dense(1, activation='sigmoid')` — binary output
- `model.summary()` printed for architecture inspection

</details>

<details open>
<summary><b>🎯 Training & Evaluation</b></summary>
<br/>

- Trained for **25 epochs** with `batch_size=8`
- **Adam optimizer** + **Binary Cross-Entropy** loss
- `verbose=1` training logs per epoch
- Entire dataset used for training (small educational dataset)

</details>

<details open>
<summary><b>🔬 Intermediate Model — Embedding & Hidden State Inspection</b></summary>
<br/>

- **Intermediate model 1** — outputs both `embed` layer and `simple_rnn` layer activations using `model.get_layer()`
- **Inspection model** — rebuilds RNN with `return_sequences=True` to capture hidden state at **every timestep**
- Trained weights **copied** from original RNN layer via `get_weights()` / `set_weights()`
- Prints `(timesteps × units)` hidden state matrix rounded to 3 decimal places
- Shows exactly how the RNN's internal memory evolves word by word

</details>

---

## 📊 Dataset

### Custom Sentiment Dataset

| Property | Value |
|----------|-------|
| **Total Samples** | 30 sentences |
| **Positive (label=1)** | 15 sentences |
| **Negative (label=0)** | 15 sentences |
| **Vocab Size** | 2000 (with `<OOV>` token) |
| **Sequence Length** | Padded to `maxlen` (longest sentence) |

**Sample Sentences:**

| Sentence | Label |
|----------|-------|
| "I love this product" | ✅ Positive (1) |
| "Absolutely fantastic experience" | ✅ Positive (1) |
| "Amazing quality and value" | ✅ Positive (1) |
| "I hate this product" | ❌ Negative (0) |
| "Terrible experience overall" | ❌ Negative (0) |
| "Utterly frustrating and annoying" | ❌ Negative (0) |

---

## 🧠 How It Works

### Main Model Pipeline

```
  Raw Sentences (30 samples)
        │
        ▼
  ┌──────────────────────────────────────────┐
  │  Tokenizer                               │
  │  num_words=2000, oov_token="<OOV>"       │
  │  fit_on_texts() → texts_to_sequences()   │
  └──────────────────────────────────────────┘
        │
        ▼
  ┌──────────────────────────────────────────┐
  │  Padding                                 │
  │  pad_sequences(padding='post')           │
  │  All sequences → uniform maxlen          │
  └──────────────────────────────────────────┘
        │
        ▼
  ┌──────────────────────────────────────────┐
  │  Keras Functional Model                  │
  │  Input(maxlen, int32)                    │
  │  → Embedding(2000, 16, mask_zero=True)   │
  │  → SimpleRNN(8, return_sequences=False)  │
  │  → Dense(1, Sigmoid)                     │
  └──────────────────────────────────────────┘
        │
        ▼
  ┌──────────────────────────────────────────┐
  │  Compile & Train                         │
  │  Optimizer : Adam                        │
  │  Loss      : Binary Cross-Entropy        │
  │  Epochs    : 25  |  Batch Size : 8       │
  └──────────────────────────────────────────┘
        │
        ▼
  ┌──────────────────────────────────────────┐
  │  Predict → Sigmoid output (0.0 – 1.0)   │
  │  > 0.5 → Positive  |  ≤ 0.5 → Negative  │
  └──────────────────────────────────────────┘
```

### Intermediate Inspection Model Pipeline

```
  Trained Model (embed + simple_rnn layers)
        │
        ▼
  ┌──────────────────────────────────────────┐
  │  Reuse trained Embedding weights         │
  │  model.get_layer('embed')                │
  └──────────────────────────────────────────┘
        │
        ▼
  ┌──────────────────────────────────────────┐
  │  Rebuild SimpleRNN                       │
  │  return_sequences=True                   │
  │  Copy weights via get_weights /          │
  │  set_weights                             │
  └──────────────────────────────────────────┘
        │
        ▼
  ┌──────────────────────────────────────────┐
  │  inspect_model.predict(example_seq)      │
  │  Output shape: (1, maxlen, 8)            │
  │  → Hidden state at every token timestep  │
  └──────────────────────────────────────────┘
        │
        ▼
  ┌──────────────────────────────────────────┐
  │  Print hidden state matrix               │
  │  Shape: (timesteps × rnn_units)          │
  │  np.round(hidden_seq[0], 3)              │
  └──────────────────────────────────────────┘
```

### How SimpleRNN Processes a Sentence

```
  Word:      "I"      "love"    "this"   "product"   <PAD>  <PAD>
              │          │         │          │          │      │
              ▼          ▼         ▼          ▼          ▼      ▼
  h_0 ──► [RNN] ──► [RNN] ──► [RNN] ──►  [RNN] ──► [RNN] ──► [RNN]
              │          │         │          │          │      │
             h_1        h_2       h_3        h_4        h_5   h_6 (final)
                                                               │
                                                               ▼
                                                         Dense(Sigmoid)
                                                         → Sentiment Label
```

---

## 📁 Project Structure

```
Deep_Learning_Part_3/
│
├── 📓 Basics_of_RNN_Implenemtation.ipynb    # Main notebook — full RNN pipeline + inspection model
│
└── 📖 README.md                             # This file
```

### Key Files Explained

| File | Purpose |
|------|---------|
| `Basics_of_RNN_Implenemtation.ipynb` | Complete RNN notebook — text preprocessing, Embedding + SimpleRNN model, binary sentiment classification, intermediate model for hidden state visualization |

---

## ⚙️ Getting Started

### Prerequisites
- Python 3.10+
- Jupyter Notebook or JupyterLab
- pip

### 1 — Clone the Repository

```bash
git clone https://github.com/YourUsername/Deep_Learning_Part_3.git
cd Deep_Learning_Part_3
```

### 2 — Install Dependencies

```bash
pip install tensorflow numpy jupyter
```

### 3 — Launch Jupyter

```bash
jupyter notebook
```

### 4 — Run the Notebook

```
Basics_of_RNN_Implenemtation.ipynb
```

> ✅ No external dataset needed — the 30-sentence sentiment dataset is defined directly inside the notebook. Just run all cells in order.

---

## 🤝 Contributing

Contributions are welcome! Ideas to extend this notebook:

- Add **LSTM / GRU** layers and compare against SimpleRNN
- Visualize hidden states as a **heatmap** using Matplotlib/Seaborn
- Train on a larger dataset like **IMDB reviews**
- Add **bidirectional RNN** for improved context understanding

Steps to contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

<div align="center">

**Made with ❤️ for Deep Learning enthusiasts and beginners**

⭐ If this helped your learning journey, please give it a star!

</div>
