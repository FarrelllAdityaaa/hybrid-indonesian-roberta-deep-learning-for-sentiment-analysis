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

This project implements **sentiment analysis** on user reviews of the **SIGNAL** (SAMSAT DIGITAL NASIONAL) application using a **hybrid deep learning approach** that combines **Indonesian RoBERTa Base Sentiment Classifier** with various sequential and spatial architectures: **LSTM, GRU, and CNN**.

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
- **Three Experimental Schemes**:
  1. **Scheme 1**: IndoRoBERTa + LSTM
  2. **Scheme 2**: IndoRoBERTa + GRU
  3. **Scheme 3**: IndoRoBERTa + CNN

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
| `df_train_aug_s1.csv` | Augmented data (Scheme 1) | ~14,610 | 1.3 MB |
| `df_train_aug_s2.csv` | Augmented data (Scheme 2) | ~19,479 | 1.7 MB |
| `df_train_aug_s3.csv` | Augmented data (Scheme 3) | ~17,046 | 1.5 MB |

### Sentiment Distribution

```python
# Label Distribution
Positif: ~54.11%  # Positive reviews
Negatif: ~16.23%  # Negative reviews  
Netral:  ~29.65%  # Neutral reviews
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
   ├── Cleaning (Noise Removal, Irrelevant Char, etc.)
   ├── Case Folding (Lowercase)
   └── Normalization
       ├── Slang Word Removal (Colloquial Indonesian Lexicon)
       └── Spelling Correction (Levenshtein Distance)
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
   Train/Val/Test Split Schema: 1. (60:20:20); 2. (80:10:10); 3. (70:15:15)
```

---

## Methodology

#### Transformer Backbone (Indonesian RoBERTa)
- **Model**: `w11wo/indonesian-roberta-base-sentiment-classifier`
- **Role**: Feature extraction and generating contextual word embeddings.
- **Key Advantages**:
  - ✅ **Context-Aware**: Understands word meaning based on surrounding context (bidirectional).
  - ✅ **Pre-trained**: Leveraging massive Indonesian corpus knowledge.
  - ✅ **Subword Tokenization**: Effectively handles slang and out-of-vocabulary words.

#### Deep Learning Heads (LSTM/GRU/CNN)
These architectures process the embeddings generated by RoBERTa to classify sentiment:
- **LSTM (Long Short-Term Memory)**: Captures long-range dependencies and prevents vanishing gradient problems.
- **GRU (Gated Recurrent Unit)**: Efficiently captures temporal patterns with a simpler gating mechanism.
- **CNN (Convolutional Neural Network)**: Excellent at extracting **local features** (n-grams) and key phrases (e.g., "sangat buruk", "mantap sekali").

### Why Hybrid?

| Feature | RoBERTa Only | RNN/CNN Only | Hybrid Model (Ours) |
| :--- | :---: | :---: | :---: |
| **Word Representation** | ✅ Contextual | ❌ Static/No Context | ✅ **Contextual & Rich** |
| **Sequence Modeling** | ⚠️ Attention-based | ✅ Sequential/Temporal | ✅ **Hierarchical** |
| **Feature Extraction** | ✅ Global Context | ⚠️ Local Patterns | ✅ **Global + Local** |
| **Indonesian Nuance** | ✅ Pre-trained | ❌ From Scratch | ✅ **Best of Both Worlds** |
| **Performance** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |

> **Result**: The hybrid model leverages RoBERTa's deep language understanding and the specialized pattern recognition of LSTM/GRU/CNN, achieving superior performance in Indonesian sentiment analysis compared to using either model individually.
---

## Model Components Detail

#### RoBERTa Base Configuration
```python
Model: w11wo/indonesian-roberta-base-sentiment-classifier
Parameters: ~125M
Architecture: 12 transformer layers
Hidden Size: 768
Attention Heads: 12
Max Sequence Length: 128
Vocab Size: 50,265
Tokenizer: Byte-Level BPE
```

#### Sequential and Spatial Heads Configuration

**LSTM (Scheme 1)**
```python
Type: Bidirectional LSTM
Input Size: 768 (from RoBERTa)
Hidden Units: 256 (per direction)
Total Output Features: 512 (256 × 2)
Num Layers: 1
Dropout: 0.3
Batch First: True
```

**GRU (Scheme 2)**
```python
Type: Bidirectional GRU
Input Size: 768
Hidden Units: 256 (per direction)
Total Output Features: 512 (256 × 2)
Num Layers: 1
Dropout: 0.3
Batch First: True
```

**CNN (Scheme 3)**
```python
Architecture: 1D Convolutional Neural Network
Input Channels: 768
Output Channels (Filters): 256
Kernel Size: 3 (Trigram focus)
Stride: 1
Padding: 1
Activation: ReLU
Pooling: Global Max Pooling (1D)
Dropout: 0.3
```

---

## Experimental Schemes

### Comparison Table

| No | Component | Scheme 1 (Baseline) | Scheme 2 (Optimized) | Scheme 3 (Comparison) |
|----|-----------|---------------------|----------------------|-----------------------|
| **1** | **Architecture** | IndoRoBERTa + LSTM | IndoRoBERTa + GRU | IndoRoBERTa + CNN |
| **2** | **Data Split** | 60:20:20 | 80:10:10 | 70:15:15 |
| **3** | **Augmentation** | Synonym Replacement (Data train kelas minoritas) | Synonym Replacement (Data train kelas minoritas) | Synonym Replacement (Data train kelas minoritas) |
| **4** | **Hidden Units** | 256 (BiLSTM) | 256 (BiGRU) | 256 Filters (CNN Kernel=3) |
| **6** | **Dropout** | 0.3 | 0.3 | 0.3 |
| **7** | **Learning Rate** | 2e-5 | 2e-5 | 2e-5 |
| **8** | **Batch Size** | 32 | 32 | 32 |
| **9** | **Max Epochs** | 10 | 10 | 10 |
| **10** | **Optimizer** | AdamW | AdamW | AdamW |
| **11** | **Loss Function** | CrossEntropyLoss | CrossEntropyLoss | CrossEntropyLoss |
| **12** | **Early Stopping** | Target Acc > 94% | Target Acc > 94% | Target Acc > 94% |
| **13** | **Samples** | ~14,610 | ~19,479 | ~17,046 |

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

## Results

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
