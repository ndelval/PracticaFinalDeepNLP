# Multi-Task NER & Sentiment Analysis with Intelligent Alert Generation

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.2.0-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/NLP-Deep%20Learning-orange.svg" alt="NLP">
</p>

A sophisticated multi-task deep learning system that simultaneously performs **Named Entity Recognition (NER)** and **Sentiment Analysis (SA)**, leveraging syntactic dependencies through Graph Convolutional Networks (GCN) and a custom Syntax-aware LSTM architecture. The system generates contextual alerts from news articles and social media content using a DeepSeek-powered language model.

## Key Features

- **Multi-Task Learning**: Joint training for NER and Sentiment Analysis with shared representations
- **Syntax-Aware LSTM (SynLSTM)**: Custom LSTM architecture that incorporates syntactic dependency information
- **GCN Integration**: Graph Convolutional Networks for capturing syntactic relationships between words
- **Character-Level Embeddings**: BiLSTM-based character embeddings for better handling of rare and OOV words
- **Intelligent Alert Generation**: LLM-powered contextual alert creation using DeepSeek-R1-Distill-Qwen-7B
- **Modular Architecture**: Configurable model variants for ablation studies and performance comparison

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Input Text                                   │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Embedding Layer                                   │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ │
│  │  GloVe (100d)   │ +  │  Char BiLSTM    │ +  │  Dependency     │ │
│  │  Word Embeddings│    │  Embeddings     │    │  Embeddings     │ │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    GCN Layer (Syntactic Dependencies)                │
│              Captures structural relationships via GCNConv           │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                              ▼
┌─────────────────────────────┐    ┌─────────────────────────────────┐
│      SynLSTM (NER)          │    │     BiLSTM (Sentiment)          │
│  Custom LSTM with GCN       │    │  2-layer Bidirectional          │
│  context integration        │    │  with pooling strategies        │
└─────────────────────────────┘    └─────────────────────────────────┘
                    │                              │
                    ▼                              ▼
┌─────────────────────────────┐    ┌─────────────────────────────────┐
│      NER Output Head        │    │    Sentiment Output Head        │
│  LayerNorm + Dropout + FC   │    │  Avg/Max/Last Pooling + FC      │
└─────────────────────────────┘    └─────────────────────────────────┘
                    │                              │
                    └──────────────┬──────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Alert Generation Module                           │
│         DeepSeek-R1-Distill-Qwen-7B (4-bit Quantized)               │
│    Generates contextual alerts: "theme: entity" format               │
└─────────────────────────────────────────────────────────────────────┘
```

## Model Variants

The architecture supports five configurable model variants through three boolean flags:

| Model       | `use_char_embs` | `use_separate_lstms` | `use_syn_lstm` | Description                         |
| ----------- | --------------- | -------------------- | -------------- | ----------------------------------- |
| **Model A** | ✅              | ✅                   | ✅             | Full architecture with all features |
| **Model B** | ❌              | ✅                   | ✅             | No character embeddings             |
| **Model C** | ❌              | ❌                   | ✅             | Shared LSTM, no char embeddings     |
| **Model D** | ❌              | ❌                   | ❌             | Standard LSTM baseline              |
| **Model E** | ✅              | ❌                   | ✅             | Shared SynLSTM with char embeddings |

## Project Structure

```
├── data/
│   ├── conll3/              # CoNLL-2003 NER dataset
│   │   ├── train.txt
│   │   ├── valid.txt
│   │   └── test.txt
│   └── twitter/             # Twitter NER dataset
│       ├── train.txt
│       ├── valid.txt
│       └── test.txt
├── models/                  # Saved model checkpoints
│   ├── combined_best_model.pt
│   ├── ner_best_model.pt
│   └── sa_best_model.pt
├── results/                 # Training results and metrics
├── runs/                    # TensorBoard logs
├── src/
│   ├── __init__.py
│   ├── alert_generation.py  # LLM-based alert generation
│   ├── config.py            # Model configuration
│   ├── dataset_stats.py     # Dataset analysis utilities
│   ├── evaluate.py          # Model evaluation
│   ├── models.py            # Neural network architectures
│   ├── new_prediction.py    # Inference pipeline
│   ├── train.py             # Training loop
│   └── utils.py             # Data loading and utilities
├── stats/                   # Dataset statistics and visualizations
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- 16GB+ RAM (for LLM-based alert generation)

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/ndelval/PracticaFinalDeepNLP.git
cd PracticaFinalDeepNLP
```

2. **Create and activate virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

4. **Download GloVe embeddings**

Download pre-trained GloVe embeddings (100-dimensional):

```bash
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
mv glove.6B.100d.txt ./
```

The file structure should look like:

```
./
├── glove.6B.100d.txt
├── src/
├── data/
└── ...
```

## Usage

### Training

Train the multi-task model:

```bash
python -m src.train
```

**Training Features:**

- AdamW optimizer with separate learning rates for NER and SA components
- ReduceLROnPlateau scheduler with validation metric monitoring
- Gradient accumulation (effective batch size: 128)
- Gradient clipping (max norm: 1.0)
- Task-specific early stopping
- TensorBoard logging

**Hyperparameters:**
| Parameter | Value |
|-----------|-------|
| Epochs | 30 |
| Batch Size | 64 |
| Learning Rate | 1e-3 (SA: 3e-3) |
| Weight Decay | 0.05 |
| Hidden Dimension | 128 |
| GCN Hidden Dimension | 50 |
| Dropout | 0.3 - 0.4 |

### Evaluation

Evaluate trained models:

```bash
python -m src.evaluate
```

Compare multiple models:

```bash
python -m src.evaluate --models combined_best_model ner_best_model sa_best_model
```

### Inference & Alert Generation

Generate predictions and alerts for custom text:

```bash
python -m src.new_prediction
```

Example output:

```
Text: German Chancellor Angela Merkel visited Paris to discuss climate change initiatives.

Predicted Entities:
  German: B-MISC
  Chancellor: O
  Angela: B-PER
  Merkel: I-PER
  visited: O
  Paris: B-LOC
  ...

Predicted Sentiment: positive

Generated Alert: Diplomatic visit: Angela Merkel
```

### Dataset Analysis

Generate comprehensive dataset statistics:

```bash
python -m src.dataset_stats
```

This creates visualizations in `stats/` including:

- Sentence length distributions
- NER tag distributions
- Sentiment class balance analysis

## Performance Metrics

| Task          | Metric   | Value  |
| ------------- | -------- | ------ |
| **NER**       | Accuracy | ~88.4% |
| **NER**       | F1 Score | ~88.4% |
| **Sentiment** | Accuracy | ~71.7% |
| **Sentiment** | F1 Score | ~72.0% |

_Results on validation set with learning rate 1e-3_

## Technical Details

### SynLSTM (Syntax-aware LSTM)

The custom LSTM cell integrates syntactic information from GCN outputs:

```python
# Modified LSTM cell equations
i = σ(W_i[h_{t-1}, x_t])           # Input gate
o = σ(W_o[h_{t-1}, x_t, m_t])      # Output gate (with GCN context)
f = σ(W_f[h_{t-1}, x_t, m_t])      # Forget gate (with GCN context)
u = tanh(W_u[h_{t-1}, x_t])        # Cell candidate

# Syntax-aware updates
ii = σ(W_{ii}[h_{t-1}, m_t])       # Syntax input gate
uu = tanh(W_{uu}[h_{t-1}, m_t])    # Syntax cell candidate

C_t = i ⊙ u + ii ⊙ uu + f ⊙ C_{t-1}
h_t = o ⊙ tanh(C_t)
```

### Alert Generation Pipeline

1. **Entity Reconstruction**: BIO tags are converted to entity spans
2. **Multi-Generation**: 30 alert candidates are generated with varied sampling
3. **Validation**: Alerts are filtered for relevance and format compliance
4. **Selection**: LLM selects the best alert based on context relevance

## Datasets

### CoNLL-2003

Standard NER benchmark with entity types:

- **PER**: Person names
- **ORG**: Organizations
- **LOC**: Locations
- **MISC**: Miscellaneous entities

### Sentiment Labels

Generated using `siebert/sentiment-roberta-large-english`:

- **Positive** (1)
- **Negative** (0)

## Dependencies

| Package           | Version | Purpose                    |
| ----------------- | ------- | -------------------------- |
| PyTorch           | 2.2.0   | Deep learning framework    |
| PyTorch Geometric | 2.6.1   | GCN implementation         |
| spaCy             | ≥3.5.0  | Dependency parsing         |
| Transformers      | ≥4.36.0 | Sentiment model & LLM      |
| Accelerate        | ≥0.21.0 | Model loading optimization |
| BitsAndBytes      | ≥0.41.0 | 4-bit quantization         |
| TensorBoard       | 2.15.1  | Training visualization     |

## Monitoring Training

Launch TensorBoard to monitor training:

```bash
tensorboard --logdir=runs
```

View metrics at `http://localhost:6006`:

- Training/Validation Loss
- NER Accuracy & F1
- Sentiment Accuracy & F1
- Learning Rate Schedule

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or suggestions, please open an issue or contact the repository maintainer.

## Documentation

For a comprehensive analysis of the methodology, related work, experimental results, ablation studies, and detailed discussion of the architecture, please refer to the **[Technical Report (PDF)](PracticaFinalDeepNLP.pdf)** included in this repository.

## Acknowledgments

- GloVe embeddings by Stanford NLP
- CoNLL-2003 shared task dataset
- Hugging Face Transformers library
- DeepSeek for the distilled Qwen model
- PyTorch Geometric team
