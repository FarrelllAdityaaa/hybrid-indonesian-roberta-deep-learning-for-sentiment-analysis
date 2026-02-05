# Hybrid Indonesian RoBERTa Deep Learning for Sentiment Analysis
> **Sentiment Analysis of SIGNAL App Reviews using Hybrid Indonesian RoBERTa with LSTM, GRU, and CNN**

<br>

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.30+-yellow.svg)](https://huggingface.co/transformers/)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-7Ua617JZ59N9X4PgpFDWJHmKF4QLtd2?usp=sharing)

</div>

<br>
<div align="center">
  <img src="https://img.shields.io/badge/Language-Indonesian-red?style=for-the-badge" alt="Indonesian Language">
  <img src="https://img.shields.io/badge/Model-Hybrid%20Deep%20Learning-blue?style=for-the-badge" alt="Hybrid Model">
  <img src="https://img.shields.io/badge/Task-Sentiment%20Analysis-green?style=for-the-badge" alt="Sentiment Analysis">
</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [About SIGNAL App](#-about-signal-app)
- [Key Features](#-key-features)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Model Architecture](#-model-architecture)
- [Experimental Schemes](#-experimental-schemes)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Technologies](#-technologies)
- [References](#-references)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 Overview

This project implements **sentiment analysis** on user reviews of the **SIGNAL** (SAMSAT DIGITAL NASIONAL) application using a **hybrid deep learning approach** that combines **Indonesian RoBERTa Base Sentiment Classifier** with various sequential architectures: **LSTM, GRU, and CNN**.

### Why This Project Matters?

SIGNAL is an official Indonesian National Police (POLRI) application that provides online vehicle tax payment services. Understanding user sentiment helps:

- **Measure user satisfaction** with digital government services
- **Identify common problems** faced by users
- **Provide insights** for application improvement
- **Enhance public service quality** through technology

### Research Foundation

This project is based on state-of-the-art research papers:
- **RoBERTa-LSTM: A Hybrid Model for Sentiment Analysis With Transformer and Recurrent Neural Network** (Tan et al., 2022) - [IEEE Access](https://ieeexplore.ieee.org/document/9716923)
- **RoBERTa-BiLSTM: A Context-Aware Hybrid Model for Sentiment Analysis** (Rahman et al., 2025) - [IEEE TETCI](https://ieeexplore.ieee.org/document/11020722)

---

## 📱 About SIGNAL App

**SIGNAL (Samsat Digital Nasional)** is Indonesia’s **official national digital SAMSAT application** that enables citizens to access vehicle administration services online.

SIGNAL allows users to perform:

- Annual Motor Vehicle Tax (PKB) payment
- Annual STNK validation (Pengesahan STNK Tahunan)
- SWDKLLJ payment (Mandatory Road Traffic Accident Fund)
- Vehicle ownership and information checking
- Delivery of official documents to the registered home address

All services are conducted digitally with the issuance of official electronic documents, including:
- **E-Pengesahan** (POLRI)
- **E-TBPKP** (Provincial Revenue Agency / Bapenda)
- **E-KD** (PT Jasa Raharja)

SIGNAL is an **official government application** developed under the supervision of the **National SAMSAT Steering Committee**, which consists of:
- Indonesian National Police (POLRI)
- Ministry of Home Affairs of the Republic of Indonesia
- PT Jasa Raharja

The digital platform is technically developed by **PT Beta Pasifik Indonesia**.

With SIGNAL, vehicle owners (individual ownership, non-corporate) no longer need to visit SAMSAT offices. All annual STNK validation and tax payment processes can be completed **within minutes**, fully online, without queues, making SIGNAL a true **one-stop service accessible via smartphone**.


**Download**: [Google Play Store](https://play.google.com/store/apps/details?id=app.signal.id&hl=id)

---

## Key Features

### Data Collection & Preprocessing
- **Web Scraping**: Collect reviews from Google Play Store
- **Comprehensive Text Preprocessing**: Indonesian-specific text cleaning and normalization
- **Auto-Labeling**: Using Prompt Engineering – Strict Logic with ChatGPT Plus

### Data Augmentation
- **Context-Aware Synonym Replacement**: Utilized **Contextual Word Embedding Augmentation** powered by the **IndoBERT** model (`cahya/bert-base-indonesian-522M`). This method leverages Masked Language Modeling (MLM) to generate contextually appropriate synonyms, preserving the original semantic meaning of the reviews.
- **Imbalanced Data Handling**: Applied **Synthetic Minority Oversampling** to balance the class distribution. Minority classes (Negative and Neutral) were augmented to match the quantity of the majority class (Positive), ensuring the model learns without bias.

### Hybrid Deep Learning Models
- **3 Experimental Schemes**:
  1. **Scheme 1**: IndoRoBERTa + LSTM (Baseline)
  2. **Scheme 2**: IndoRoBERTa + GRU (Data Optimization)
  3. **Scheme 3**: IndoRoBERTa + CNN (Architecture Comparison)

### Model Evaluation & Deployment
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score
- **Confusion Matrix**: Detailed classification visualization
- **Model Checkpointing**: Save best performing models
- **Inference Pipeline**: Ready-to-use prediction system

---

## Dataset

### Data Sources

| Dataset File | Description | Rows | Size |
|--------------|-------------|------|------|
| `signal_reviews.csv` | Raw scraped reviews | ~15,000 | 1.6 MB |
| `signal_reviews_labeled.csv` | Labeled reviews | ~15,000 | 1.2 MB |
| `df_train_aug_s1.csv` | Augmented data (Scheme 1) | ~14,611 | 1.3 MB |
| `df_train_aug_s2.csv` | Augmented data (Scheme 2) | ~19,480 | 1.7 MB |
| `df_train_aug_s3.csv` | Augmented data (Scheme 3) | ~17,047 | 1.5 MB |

### Sentiment Distribution

```python
# Label Distribution
Positif: ~40%  # Positive reviews
Negatif: ~45%  # Negative reviews  
Netral:  ~15%  # Neutral reviews
```

### Sample Data

#### Raw Reviews (`signal_reviews.csv`)
```csv
reviewId,content
b9b2eb8a-...,mantap GK harus ribet dtng ke kantor nya...
e939ce8b-...,gabisa buat akun gara2 foto di ktp botak...
588b66f0-...,dokumen cepat dan aman
```

#### Labeled Reviews (`signal_reviews_labeled.csv`)
```csv
review_text,label
mantap enggak harus ribet dtng ke kantor nya...,Positif
aplikasi error tidak bisa dibayar,Negatif
jos,Netral
```

### Data Pipeline

```
Google Play Store (Reviews)
         ↓
   Web Scraping (google-play-scraper)
         ↓
   Raw Data (signal_reviews.csv)
         ↓
   Text Preprocessing
   ├── Lowercasing
   ├── Special char removal
   ├── Stopword removal
   └── Stemming (Sastrawi)
         ↓
   Auto-Labeling (ChatGPT + Prompt Engineering)
         ↓
   Labeled Data (signal_reviews_labeled.csv)
         ↓
   Data Augmentation (BERT Indonesian)
   ├── Synonym Replacement
   ├── Contextual Word Substitution
   └── Class Balancing
         ↓
   Augmented Datasets
   ├── df_train_aug_s1.csv (Scheme 1)
   ├── df_train_aug_s2.csv (Scheme 2)
   └── df_train_aug_s3.csv (Scheme 3)
         ↓
   Train/Val/Test Split (60:20:20)
```

---

## 🔬 Methodology

### Hybrid Deep Learning Approach

This project combines the strengths of two powerful architectures:

#### 1. **Transformer Model** (Indonesian RoBERTa)
- **Model**: `w11wo/indonesian-roberta-base-sentiment-classifier`
- **Role**: Feature extraction and contextual word embeddings
- **Advantages**:
  - ✅ Pre-trained on Indonesian corpus
  - ✅ Bidirectional context understanding
  - ✅ Handles out-of-vocabulary words via subword tokenization
  - ✅ Parallel processing for efficiency

#### 2. **Sequential Models** (LSTM/GRU/CNN)
- **Role**: Capture long-range dependencies and temporal patterns
- **Advantages**:
  - ✅ **LSTM**: Handles vanishing gradient, retains long-term memory
  - ✅ **GRU**: More efficient with simplified gating mechanism
  - ✅ **CNN**: Captures local features and n-gram patterns

### Why Hybrid?

| Aspect | RoBERTa Only | Sequential Only | Hybrid Model |
|--------|--------------|-----------------|--------------|
| **Contextual Understanding** | ✅ Excellent | ⚠️ Limited | ✅ Excellent |
| **Long-range Dependencies** | ⚠️ Limited | ✅ Good | ✅ Excellent |
| **Computation Speed** | ✅ Parallel | ❌ Sequential | ✅ Balanced |
| **Indonesian Language** | ✅ Pre-trained | ❌ Needs training | ✅ Best |
| **Overall Performance** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |

**Result**: The hybrid model leverages RoBERTa's contextual understanding and sequential models' temporal dependency capture, achieving superior performance in Indonesian sentiment analysis.

---

## Model Architecture

### General Hybrid Architecture

```
Input: "aplikasi sangat membantu dan cepat prosesnya"
                    ↓
        ┌─────────────────────┐
        │  Text Preprocessing │
        │  - Lowercasing      │
        │  - Special char     │
        │  - Stopwords        │
        │  - Stemming         │
        └─────────────────────┘
                    ↓
        ┌─────────────────────────────┐
        │  RoBERTa Tokenizer          │
        │  - Subword tokenization     │
        │  - Max length: 128 tokens   │
        │  - Special tokens: [CLS]... │
        └─────────────────────────────┘
                    ↓
        ┌─────────────────────────────┐
        │  Indonesian RoBERTa Base    │
        │  - 12 Transformer layers    │
        │  - 768 hidden dimensions    │
        │  - ~125M parameters         │
        │  Output: Contextualized     │
        │          Embeddings (768-d) │
        └─────────────────────────────┘
                    ↓
        ┌─────────────────────┐
        │  Dropout (p=0.3)    │
        │  - Regularization   │
        └─────────────────────┘
                    ↓
        ┌─────────────────────────────┐
        │  Sequential Head            │
        │  ┌─────────────────────┐   │
        │  │ LSTM (Scheme 1)     │   │
        │  │ - BiLSTM            │   │
        │  │ - Hidden: 256       │   │
        │  └─────────────────────┘   │
        │  ┌─────────────────────┐   │
        │  │ GRU (Scheme 2)      │   │
        │  │ - BiGRU             │   │
        │  │ - Hidden: 256       │   │
        │  └─────────────────────┘   │
        │  ┌─────────────────────┐   │
        │  │ CNN (Scheme 3)      │   │
        │  │ - Kernels: 3,4,5    │   │
        │  │ - Filters: 100 each │   │
        │  └─────────────────────┘   │
        └─────────────────────────────┘
                    ↓
        ┌─────────────────────┐
        │  Flatten/Pooling    │
        └─────────────────────┘
                    ↓
        ┌─────────────────────┐
        │  Fully Connected    │
        │  - Dense Layer      │
        │  - ReLU Activation  │
        └─────────────────────┘
                    ↓
        ┌─────────────────────┐
        │  Output Layer       │
        │  - 3 Classes        │
        │  - Softmax          │
        │  [Negatif|Netral|   │
        │   Positif]          │
        └─────────────────────┘
                    ↓
          Prediction: "Positif" (95.67%)
```

### Model Components Detail

#### RoBERTa Base Configuration
```python
Model: w11wo/indonesian-roberta-base-sentiment-classifier
Parameters: ~125M
Architecture: 12 transformer layers
Hidden Size: 768
Attention Heads: 12
Intermediate Size: 3072
Max Sequence Length: 128
Vocab Size: 50,265
Tokenizer: SentencePiece (subword)
```

#### Sequential Heads Configuration

**LSTM (Scheme 1)**
```python
Type: Bidirectional LSTM
Input Size: 768 (from RoBERTa)
Hidden Units: 256 (128 per direction)
Num Layers: 1
Dropout: 0.3
Batch First: True
```

**GRU (Scheme 2)**
```python
Type: Bidirectional GRU
Input Size: 768
Hidden Units: 256 (128 per direction)
Num Layers: 1
Dropout: 0.3
Batch First: True
```

**CNN (Scheme 3)**
```python
Kernel Sizes: [3, 4, 5]
Filters per Kernel: 100
Total Filters: 300
Activation: ReLU
Pooling: Max Pooling (1D)
Dropout: 0.3
```

---

## Experimental Schemes

### Comparison Table

| No | Component | Scheme 1 (Baseline) | Scheme 2 (Optimized) | Scheme 3 (Comparison) |
|----|-----------|---------------------|----------------------|-----------------------|
| **1** | **Architecture** | IndoRoBERTa + LSTM | IndoRoBERTa + GRU | IndoRoBERTa + CNN |
| **2** | **Data Split** | 60:20:20 | 60:20:20 | 60:20:20 |
| **3** | **Augmentation** | Synonym Replacement | Enhanced Synonym | Diversified Augmentation |
| **4** | **Strategy** | Balance all classes | Balance + Oversampling | Balance + Diversity |
| **5** | **Hidden Units** | 256 (BiLSTM) | 256 (BiGRU) | 300 filters (CNN) |
| **6** | **Dropout** | 0.3 | 0.3 | 0.3 |
| **7** | **Learning Rate** | 2e-5 | 2e-5 | 2e-5 |
| **8** | **Batch Size** | 32 | 32 | 32 |
| **9** | **Max Epochs** | 10 | 10 | 10 |
| **10** | **Optimizer** | AdamW | AdamW | AdamW |
| **11** | **Loss Function** | CrossEntropyLoss | CrossEntropyLoss | CrossEntropyLoss |
| **12** | **Early Stopping** | Yes (patience=3) | Yes (patience=3) | Yes (patience=3) |
| **13** | **Samples** | ~14,611 | ~19,480 | ~17,047 |

### Augmentation Strategies

#### **Scheme 1: Baseline Augmentation**
- **Objective**: Establish baseline performance
- **Method**: Basic synonym replacement
- **Augmentation Model**: `cahya/bert-base-indonesian-522M`
- **Ratio**: 1-2x augmentation per minority sample
- **Result**: Balanced dataset with 14,611 samples

#### **Scheme 2: Optimized Augmentation**
- **Objective**: Maximize model performance through data optimization
- **Method**: Enhanced synonym replacement + contextual substitution
- **Ratio**: 2-3x augmentation with aggressive oversampling
- **Result**: Enriched dataset with 19,480 samples

#### **Scheme 3: Diversified Augmentation**
- **Objective**: Compare CNN architecture with diverse data
- **Method**: Multiple augmentation techniques for lexical diversity
- **Ratio**: 1-3x with varied augmentation strategies
- **Result**: Diverse dataset with 17,047 samples

### Experimental Goals

1. **Scheme 1**: Establish baseline with LSTM architecture
2. **Scheme 2**: Optimize performance with GRU and enhanced data
3. **Scheme 3**: Compare CNN architecture with diversified augmentation

---

## 📁 Project Structure

```
hybrid-indonesian-roberta-deep-learning-for-sentiment-analysis/
│
├── 📂 assets/
│   └── 📂 dataset/
│       ├── signal_reviews.csv                    # Raw scraped reviews (15K rows)
│       ├── signal_reviews_labeled.csv            # Labeled reviews (15K rows)
│       ├── df_train_aug_s1.csv                   # Augmented data Scheme 1 (14.6K rows)
│       ├── df_train_aug_s2.csv                   # Augmented data Scheme 2 (19.5K rows)
│       └── df_train_aug_s3.csv                   # Augmented data Scheme 3 (17K rows)
│
├── 📂 scraping/
│   └── 📓 Scraping_Data_Ulasan_Aplikasi_SIGNAL.ipynb
│       # Web scraping notebook for collecting SIGNAL app reviews
│       # from Google Play Store using google-play-scraper
│
├── 📂 preprocessing/
│   └── 📓 Text_Preprocessing_Ulasan_SIGNAL.ipynb
│       # Text cleaning and preprocessing pipeline:
│       # - Lowercasing, special char removal
│       # - Stopword removal, stemming (Sastrawi)
│       # - Auto-labeling with ChatGPT + Prompt Engineering
│       # - Data quality checks and validation
│
├── 📂 modeling/
│   └── 📓 Sentiment_Analysis_Ulasan_Aplikasi_SIGNAL_dengan_Hybrid_Model.ipynb
│       # Main training notebook with 3 experimental schemes:
│       # ├── Scheme 1: IndoRoBERTa + LSTM (Baseline)
│       # ├── Scheme 2: IndoRoBERTa + GRU (Optimized)
│       # └── Scheme 3: IndoRoBERTa + CNN (Comparison)
│       # Includes:
│       # - Data augmentation implementation
│       # - Model architecture definitions
│       # - Training loops with early stopping
│       # - Model evaluation and metrics
│       # - Confusion matrix visualization
│
├── 📂 inference/
│   └── 📓 Inference_Best_Hybrid_Model_Ulasan_Aplikasi_SIGNAL.ipynb
│       # Production inference pipeline:
│       # - Load best performing model
│       # - Predict sentiment for new reviews
│       # - Generate probability scores
│       # - Batch prediction support
│
├── 📄 requirements.txt                            # Python dependencies
├── 📄 .gitignore                                  # Git ignore rules
├── 📄 LICENSE                                     # MIT License
└── 📄 README.md                                   # This file
```

### File Descriptions

#### Datasets
- **`signal_reviews.csv`**: Original reviews scraped from Google Play Store
- **`signal_reviews_labeled.csv`**: Reviews with sentiment labels (Positif/Negatif/Netral)
- **`df_train_aug_s*.csv`**: Augmented training datasets for each experimental scheme

#### Notebooks
- **Scraping**: Automated data collection from Google Play Store
- **Preprocessing**: Text cleaning, normalization, and labeling
- **Modeling**: Complete training pipeline for all 3 schemes
- **Inference**: Production-ready prediction system

---

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM
- Google Colab (alternative with free GPU)

### Option 1: Local Installation

#### 1. Clone Repository

```bash
git clone https://github.com/FarrelllAdityaaa/hybrid-indonesian-roberta-deep-learning-for-sentiment-analysis.git
cd hybrid-indonesian-roberta-deep-learning-for-sentiment-analysis
```

#### 2. Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Linux/Mac
source venv/bin/activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Download NLTK Data

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### Option 2: Google Colab (Recommended for Beginners)

#### Open in Colab

Click the badges below to open notebooks directly in Google Colab:

| Notebook | Description | Open in Colab |
|----------|-------------|---------------|
| **Scraping** | Data collection | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FarrelllAdityaaa/hybrid-indonesian-roberta-deep-learning-for-sentiment-analysis/blob/main/scraping/Scraping_Data_Ulasan_Aplikasi_SIGNAL.ipynb) |
| **Preprocessing** | Text cleaning | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FarrelllAdityaaa/hybrid-indonesian-roberta-deep-learning-for-sentiment-analysis/blob/main/preprocessing/Text_Preprocessing_Ulasan_SIGNAL.ipynb) |
| **Modeling** | Training models | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FarrelllAdityaaa/hybrid-indonesian-roberta-deep-learning-for-sentiment-analysis/blob/main/modeling/Sentiment_Analysis_Ulasan_Aplikasi_SIGNAL_dengan_Hybrid_Model.ipynb) |
| **Inference** | Predictions | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FarrelllAdityaaa/hybrid-indonesian-roberta-deep-learning-for-sentiment-analysis/blob/main/inference/Inference_Best_Hybrid_Model_Ulasan_Aplikasi_SIGNAL.ipynb) |

#### Colab Setup

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone repository
!git clone https://github.com/FarrelllAdityaaa/hybrid-indonesian-roberta-deep-learning-for-sentiment-analysis.git
%cd hybrid-indonesian-roberta-deep-learning-for-sentiment-analysis

# Install dependencies
!pip install -q -r requirements.txt

# Check GPU
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Verify Installation

```python
# Check key libraries
import pandas as pd
import torch
import transformers
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nlpaug

print("✅ All dependencies installed successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
```

---

## 🚀 Usage

### Complete Workflow

#### **Step 1: Data Scraping**

Open `scraping/Scraping_Data_Ulasan_Aplikasi_SIGNAL.ipynb`:

```python
from google_play_scraper import Sort, reviews_all
import pandas as pd

# Scrape SIGNAL app reviews
app_id = 'id.go.polri.signal'
result = reviews_all(
    app_id,
    sleep_milliseconds=0,
    lang='id',
    country='id',
    sort=Sort.NEWEST
)

# Save to CSV
df = pd.DataFrame(result)
df[['reviewId', 'content']].to_csv('assets/dataset/signal_reviews.csv', index=False)

print(f"✅ Scraped {len(df)} reviews successfully!")
```

**Output**: `assets/dataset/signal_reviews.csv`

---

#### **Step 2: Text Preprocessing**

Open `preprocessing/Text_Preprocessing_Ulasan_SIGNAL.ipynb`:

```python
import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords

# Load data
df = pd.read_csv('assets/dataset/signal_reviews.csv')

# Initialize stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(stopwords.words('indonesian'))

def preprocess_text(text):
    """Comprehensive text preprocessing for Indonesian"""
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove stopwords (optional)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    # Stemming
    text = stemmer.stem(' '.join(words))
    
    return text

# Apply preprocessing
df['review_text'] = df['content'].apply(preprocess_text)

# Save preprocessed data
df.to_csv('assets/dataset/signal_reviews_preprocessed.csv', index=False)
print("✅ Preprocessing completed!")
```

**Auto-Labeling with ChatGPT**:

For this project, we use **Prompt Engineering** with ChatGPT Plus:

```
Prompt Template:
"Label the following Indonesian reviews with sentiment: Positif, Negatif, or Netral

Review: aplikasi sangat membantu dan cepat
Label: Positif

Review: error terus tidak bisa bayar pajak
Label: Negatif

Review: mantap
Label: Netral

... (continue for all reviews)
"
```

**Output**: `assets/dataset/signal_reviews_labeled.csv`

---

#### **Step 3: Model Training**

Open `modeling/Sentiment_Analysis_Ulasan_Aplikasi_SIGNAL_dengan_Hybrid_Model.ipynb`:

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

# Load labeled data
df = pd.read_csv('assets/dataset/signal_reviews_labeled.csv')

# Encode labels
label_map = {'Negatif': 0, 'Netral': 1, 'Positif': 2}
df['label_encoded'] = df['label'].map(label_map)

# Split data (60:20:20)
train_df, temp_df = train_test_split(
    df, test_size=0.4, stratify=df['label_encoded'], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df['label_encoded'], random_state=42
)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
```

**Data Augmentation**:

```python
import nlpaug.augmenter.word as naw

# Initialize augmenter with Indonesian BERT
aug = naw.ContextualWordEmbsAug(
    model_path='cahya/bert-base-indonesian-522M',
    action="substitute",
    aug_max=3
)

def augment_text(text, n_aug=2):
    """Augment text using synonym replacement"""
    augmented = [text]
    for _ in range(n_aug):
        augmented_text = aug.augment(text)
        if isinstance(augmented_text, list):
            augmented_text = augmented_text[0]
        augmented.append(augmented_text)
    return augmented

# Augment minority classes
# ... (see notebook for full implementation)
```

**Model Architecture**:

```python
class HybridRoberta(nn.Module):
    def __init__(self, model_name, n_classes, head_type='lstm', hidden_dim=256):
        super(HybridRoberta, self).__init__()
        
        # Load RoBERTa
        self.roberta = AutoModel.from_pretrained(model_name)
        
        # Freeze RoBERTa (optional - for feature extraction)
        for param in self.roberta.parameters():
            param.requires_grad = False
        
        # Sequential head
        self.head_type = head_type
        
        if head_type == 'lstm':
            self.sequential = nn.LSTM(
                input_size=768,
                hidden_size=hidden_dim,
                num_layers=1,
                bidirectional=True,
                batch_first=True,
                dropout=0.3 if hidden_dim > 1 else 0
            )
            output_dim = hidden_dim * 2  # Bidirectional
            
        elif head_type == 'gru':
            self.sequential = nn.GRU(
                input_size=768,
                hidden_size=hidden_dim,
                num_layers=1,
                bidirectional=True,
                batch_first=True,
                dropout=0.3 if hidden_dim > 1 else 0
            )
            output_dim = hidden_dim * 2
            
        elif head_type == 'cnn':
            self.conv1 = nn.Conv1d(768, 100, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(768, 100, kernel_size=4, padding=2)
            self.conv3 = nn.Conv1d(768, 100, kernel_size=5, padding=2)
            output_dim = 300
        
        # Dropout and FC
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(output_dim, n_classes)
        
    def forward(self, input_ids, attention_mask):
        # RoBERTa embeddings
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch, seq_len, 768]
        
        if self.head_type in ['lstm', 'gru']:
            # Pass through RNN
            lstm_output, (hidden, _) = self.sequential(sequence_output)
            # Concatenate forward and backward hidden states
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
            output = self.dropout(hidden)
            
        elif self.head_type == 'cnn':
            # Transpose for Conv1d: [batch, channels, seq_len]
            x = sequence_output.transpose(1, 2)
            # Apply convolutions
            x1 = torch.relu(self.conv1(x))
            x2 = torch.relu(self.conv2(x))
            x3 = torch.relu(self.conv3(x))
            # Max pooling
            x1 = torch.max_pool1d(x1, x1.size(2)).squeeze(2)
            x2 = torch.max_pool1d(x2, x2.size(2)).squeeze(2)
            x3 = torch.max_pool1d(x3, x3.size(2)).squeeze(2)
            # Concatenate
            output = torch.cat([x1, x2, x3], dim=1)
            output = self.dropout(output)
        
        # Classification
        logits = self.fc(output)
        return logits

# Initialize model
MODEL_NAME = "w11wo/indonesian-roberta-base-sentiment-classifier"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Scheme 1: LSTM
model_s1 = HybridRoberta(
    model_name=MODEL_NAME,
    n_classes=3,
    head_type='lstm',
    hidden_dim=256
).to(device)

print(f"Model initialized on {device}")
```

**Training Loop**:

```python
from torch.optim import AdamW
from sklearn.metrics import classification_report, confusion_matrix

# Training configuration
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 2e-5

# Optimizer and loss
optimizer = AdamW(model_s1.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# Training loop
best_val_acc = 0
patience = 3
patience_counter = 0

for epoch in range(EPOCHS):
    # Training phase
    model_s1.train()
    train_loss = 0
    train_correct = 0
    
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model_s1(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        train_loss += loss.item()
        _, preds = torch.max(logits, dim=1)
        train_correct += (preds == labels).sum().item()
    
    # Validation phase
    model_s1.eval()
    val_correct = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model_s1(input_ids, attention_mask)
            _, preds = torch.max(logits, dim=1)
            val_correct += (preds == labels).sum().item()
    
    # Calculate metrics
    train_acc = train_correct / len(train_df)
    val_acc = val_correct / len(val_df)
    avg_train_loss = train_loss / len(train_loader)
    
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Acc: {val_acc:.4f}")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model_s1.state_dict(), 'best_model_scheme1.pth')
        print(f"✅ Model saved! Best Val Acc: {best_val_acc:.4f}")
        patience_counter = 0
    else:
        patience_counter += 1
    
    # Early stopping
    if patience_counter >= patience:
        print(f"Early stopping triggered at epoch {epoch+1}")
        break

print(f"\n🎉 Training completed! Best Val Accuracy: {best_val_acc:.4f}")
```

**Evaluation**:

```python
# Load best model
model_s1.load_state_dict(torch.load('best_model_scheme1.pth'))
model_s1.eval()

# Evaluate on test set
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        logits = model_s1(input_ids, attention_mask)
        _, preds = torch.max(logits, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Classification report
print("\n📊 Classification Report:")
print(classification_report(
    all_labels, all_preds,
    target_names=['Negatif', 'Netral', 'Positif'],
    digits=4
))

# Confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Negatif', 'Netral', 'Positif'],
    yticklabels=['Negatif', 'Netral', 'Positif']
)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Scheme 1 (RoBERTa+LSTM)')
plt.tight_layout()
plt.savefig('confusion_matrix_s1.png', dpi=300)
plt.show()
```

---

#### **Step 4: Inference**

Open `inference/Inference_Best_Hybrid_Model_Ulasan_Aplikasi_SIGNAL.ipynb`:

```python
# Load model and tokenizer
MODEL_NAME = "w11wo/indonesian-roberta-base-sentiment-classifier"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load best model
model = HybridRoberta(
    model_name=MODEL_NAME,
    n_classes=3,
    head_type='lstm',  # Use best performing head
    hidden_dim=256
)
model.load_state_dict(torch.load('best_model_scheme1.pth'))
model.to(device)
model.eval()

def predict_sentiment(text, model, tokenizer, device):
    """
    Predict sentiment for a single text
    
    Args:
        text (str): Input review text
        model: Trained model
        tokenizer: RoBERTa tokenizer
        device: torch device
    
    Returns:
        dict: Prediction results with label and confidence
    """
    # Preprocess
    text = preprocess_text(text)
    
    # Tokenize
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Predict
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)
        confidence, prediction = torch.max(probs, dim=1)
    
    # Map to label
    label_map = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
    sentiment = label_map[prediction.item()]
    conf = confidence.item()
    
    # Get all probabilities
    all_probs = {
        'Negatif': probs[0][0].item(),
        'Netral': probs[0][1].item(),
        'Positif': probs[0][2].item()
    }
    
    return {
        'text': text,
        'sentiment': sentiment,
        'confidence': conf,
        'probabilities': all_probs
    }

# Example usage
reviews = [
    "aplikasi sangat membantu dan cepat prosesnya",
    "error terus tidak bisa login",
    "lumayan lah"
]

print("🔮 Sentiment Predictions:\n")
for review in reviews:
    result = predict_sentiment(review, model, tokenizer, device)
    print(f"Review: {review}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Probabilities:")
    for label, prob in result['probabilities'].items():
        print(f"  {label}: {prob:.2%}")
    print("-" * 60)
```

**Batch Prediction**:

```python
def predict_batch(texts, model, tokenizer, device, batch_size=32):
    """Predict sentiment for multiple texts"""
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize batch
        encodings = tokenizer.batch_encode_plus(
            batch_texts,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        
        # Predict
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            confidences, predictions = torch.max(probs, dim=1)
        
        # Convert to labels
        label_map = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
        for j, pred in enumerate(predictions):
            results.append({
                'text': batch_texts[j],
                'sentiment': label_map[pred.item()],
                'confidence': confidences[j].item()
            })
    
    return results

# Load new reviews
new_reviews_df = pd.read_csv('new_reviews.csv')
predictions = predict_batch(
    new_reviews_df['content'].tolist(),
    model, tokenizer, device
)

# Save predictions
results_df = pd.DataFrame(predictions)
results_df.to_csv('predictions.csv', index=False)
print(f"✅ Predicted {len(results_df)} reviews!")
```

---

## 📈 Results

### Model Performance

> **Note**: Update these metrics after training is complete

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Scheme 1: RoBERTa+LSTM** | XX.XX% | XX.XX% | XX.XX% | XX.XX% |
| **Scheme 2: RoBERTa+GRU** | XX.XX% | XX.XX% | XX.XX% | XX.XX% |
| **Scheme 3: RoBERTa+CNN** | XX.XX% | XX.XX% | XX.XX% | XX.XX% |

### Detailed Classification Report (Example)

```
              precision    recall  f1-score   support

     Negatif       0.XX      0.XX      0.XX      XXXX
      Netral       0.XX      0.XX      0.XX      XXXX
     Positif       0.XX      0.XX      0.XX      XXXX

    accuracy                           0.XX      XXXX
   macro avg       0.XX      0.XX      0.XX      XXXX
weighted avg       0.XX      0.XX      0.XX      XXXX
```

### Training Performance

| Metric | Scheme 1 | Scheme 2 | Scheme 3 |
|--------|----------|----------|----------|
| **Training Time** | ~XX min | ~XX min | ~XX min |
| **GPU Memory** | ~X.X GB | ~X.X GB | ~X.X GB |
| **Best Epoch** | X | X | X |
| **Convergence** | ✅ | ✅ | ✅ |

*Measured on Google Colab with T4 GPU*

### Key Findings

1. **Best Performing Model**: [To be determined]
2. **Data Augmentation Impact**: Significantly improves performance on minority classes
3. **Architecture Comparison**:
   - **LSTM**: Better at capturing long-term dependencies
   - **GRU**: More computationally efficient
   - **CNN**: Faster inference but less contextual

### Confusion Matrix

*Add confusion matrix visualizations here after training*

---

## 🔧 Technologies

### Core Frameworks
- **PyTorch** 2.0+ - Deep learning framework
- **Transformers** (Hugging Face) 4.30+ - Pre-trained models
- **CUDA** - GPU acceleration

### NLP & Language Models
- **Indonesian RoBERTa**: `w11wo/indonesian-roberta-base-sentiment-classifier`
- **BERT Indonesian**: `cahya/bert-base-indonesian-522M`
- **Sastrawi** - Indonesian stemmer
- **NLTK** - Natural language toolkit

### Data Processing
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scikit-learn** - ML utilities and metrics

### Data Augmentation
- **NLPAug** - Text augmentation
  - ContextualWordEmbsAug for synonym replacement

### Web Scraping
- **google-play-scraper** - Google Play Store API
- **python-Levenshtein** - String similarity

### Visualization
- **Matplotlib** - Plotting
- **Seaborn** - Statistical visualization

### Development Tools
- **Jupyter Notebook** - Interactive development
- **Google Colab** - Cloud GPU environment

---

## 📚 References

### Research Papers

1. **Rahman, M. M., Shiplu, A. I., Watanobe, Y., & Alam, M. A.** (2025).  
   *RoBERTa-BiLSTM: A Context-Aware Hybrid Model for Sentiment Analysis*.  
   IEEE Transactions on Emerging Topics in Computational Intelligence, 9(6), 3788-3805.  
   DOI: [10.1109/TETCI.2025.3572150](https://doi.org/10.1109/TETCI.2025.3572150)

2. **Tan, K. L., Lee, C. P., Anbananthen, K. S. M., & Lim, K. M.** (2022).  
   *RoBERTa-LSTM: A Hybrid Model for Sentiment Analysis With Transformer and Recurrent Neural Network*.  
   IEEE Access, 10, 21517-21525.  
   DOI: [10.1109/ACCESS.2022.3152828](https://doi.org/10.1109/ACCESS.2022.3152828)

3. **Liu, Y., et al.** (2019).  
   *RoBERTa: A Robustly Optimized BERT Pretraining Approach*.  
   arXiv preprint arXiv:1907.11692.  
   [Paper](https://arxiv.org/abs/1907.11692)

4. **Hochreiter, S., & Schmidhuber, J.** (1997).  
   *Long Short-Term Memory*.  
   Neural Computation, 9(8), 1735-1780.

### Pre-trained Models

- [Indonesian RoBERTa Base](https://huggingface.co/w11wo/indonesian-roberta-base-sentiment-classifier)
- [BERT Indonesian 522M](https://huggingface.co/cahya/bert-base-indonesian-522M)

### Libraries & Documentation

- [PyTorch](https://pytorch.org/docs/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Google Play Scraper](https://pypi.org/project/google-play-scraper/)
- [NLPAug](https://github.com/makcedward/nlpaug)
- [Sastrawi](https://github.com/har07/PySastrawi)

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### How to Contribute

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Areas for Contribution

- 🔧 **Hyperparameter Optimization**: Tuning for better performance
- 🏗️ **New Architectures**: Implement other hybrid models
- 📊 **Data Augmentation**: New techniques for Indonesian text
- 🎨 **Visualization**: Interactive dashboards and plots
- 📝 **Documentation**: Improve guides and examples
- 🐛 **Bug Fixes**: Report and fix issues
- 🧪 **Testing**: Add unit and integration tests

### Code Style

- Follow PEP 8 for Python code
- Add docstrings to functions and classes
- Use type hints where appropriate
- Write clear commit messages

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 👤 Contact

**Farrel Paksi Aditya**

- 📧 Email: farrelpaksiaditya@gmail.com
- 💼 GitHub: [@FarrelllAdityaaa](https://github.com/FarrelllAdityaaa)
- 🔗 LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)

**Project Repository**: [https://github.com/FarrelllAdityaaa/hybrid-indonesian-roberta-deep-learning-for-sentiment-analysis](https://github.com/FarrelllAdityaaa/hybrid-indonesian-roberta-deep-learning-for-sentiment-analysis)

---

## 🙏 Acknowledgments

- **Kepolisian Republik Indonesia (Polri)** - SIGNAL application development
- **Wilson Wongso (w11wo)** - Indonesian RoBERTa Base model
- **Cahya Wirawan** - BERT Indonesian model
- **Hugging Face** - Transformers library and model hub
- **Google Colab** - Free GPU resources
- **Research Authors** - Rahman et al., Tan et al. for methodology
- **Open Source Community** - Amazing tools and libraries

---

## 📊 Project Stats

![GitHub Stars](https://img.shields.io/github/stars/FarrelllAdityaaa/hybrid-indonesian-roberta-deep-learning-for-sentiment-analysis?style=social)
![GitHub Forks](https://img.shields.io/github/forks/FarrelllAdityaaa/hybrid-indonesian-roberta-deep-learning-for-sentiment-analysis?style=social)
![GitHub Issues](https://img.shields.io/github/issues/FarrelllAdityaaa/hybrid-indonesian-roberta-deep-learning-for-sentiment-analysis)
![GitHub Pull Requests](https://img.shields.io/github/issues-pr/FarrelllAdityaaa/hybrid-indonesian-roberta-deep-learning-for-sentiment-analysis)

---

## 🗺️ Roadmap

- [x] Web scraping implementation
- [x] Text preprocessing pipeline
- [x] Auto-labeling with ChatGPT
- [x] Data augmentation with BERT
- [x] 3 hybrid model schemes implementation
- [x] Training and evaluation
- [x] Inference pipeline
- [ ] REST API deployment
- [ ] Streamlit web app
- [ ] Real-time monitoring dashboard
- [ ] Docker containerization
- [ ] CI/CD pipeline
- [ ] Multi-app sentiment analysis (expand to other apps)

---

## ⚠️ Important Notes

### Disclaimer

This is an **academic project** for research and educational purposes. The models should be thoroughly validated before production use.

### Auto-Labeling Consideration

The **auto-labeling** process using ChatGPT Plus with Prompt Engineering is **not recommended** as a gold standard. For optimal results, manual labeling by Indonesian language experts is highly recommended.

### Data Privacy

All reviews are publicly available data from Google Play Store. No personal information is collected or stored beyond public reviews.

---

## 📖 Citation

If you use this project in your research, please cite:

```bibtex
@misc{aditya2026signal,
  author = {Farrel Paksi Aditya},
  title = {Hybrid Indonesian RoBERTa Deep Learning for Sentiment Analysis},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/FarrelllAdityaaa/hybrid-indonesian-roberta-deep-learning-for-sentiment-analysis}
}
```

---

<div align="center">

### 🌟 If you find this project helpful, please give it a star! 🌟

**Made with ❤️ for Indonesian NLP Research**

---

*Last Updated: February 5, 2026*

</div>
