# Medical Question Answering System with RAG

A Retrieval-Augmented Generation (RAG) system for medical question answering, combining semantic search with large language models to provide accurate and contextually relevant medical information.

## Features

- **RAG Architecture**: Combines FAISS vector search with language models for enhanced accuracy
- **Medical Data Processing**: Preprocesses medical documents from multiple authoritative sources
- **Semantic Search**: Uses sentence transformers for similarity-based document retrieval
- **RLHF Training**: Includes Reinforcement Learning from Human Feedback implementation
- **Comparative Analysis**: Built-in comparison between RAG and non-RAG responses
- **Safety Features**: Includes medical disclaimers and professional consultation recommendations

## Project Structure

```
├── Demo.ipynb              # Interactive demonstration notebook
├── main.py                 # Main application entry point
├── preprocessing.py        # Data preprocessing and indexing
├── gido.py                # Additional utilities
├── RLHF.py                # Reinforcement Learning from Human Feedback
├── requirements.txt        # Python dependencies
├── data/
│   ├── faiss_index/       # FAISS vector index and mappings
│   ├── raw/               # Raw medical data from various sources
│   ├── preference_data*.jsonl # RLHF training data
│   ├── qwen-rlhf/         # Fine-tuned model configurations
│   └── reward-model/      # Reward model for RLHF
```

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Required packages listed in `requirements.txt`

### Installation

1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd "RAG for Medical QA"
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up the FAISS index (if not already available):
    ```bash
    python preprocessing.py
    ```

### Usage

#### Interactive Demo
Open and run `Demo.ipynb` in Jupyter Notebook for an interactive experience:
```bash
jupyter notebook Demo.ipynb
```

#### Command Line Interface
```bash
python main.py
```

#### Colab Demo
Access the online demo: [Colab Notebook](https://colab.research.google.com/drive/1oG3zecZXkD25uvWqhqYVsPNofIliIq7e?usp=sharing&fbclid=IwY2xjawLCQ81leHRuA2FlbQIxMABicmlkETFONXQ3V0Zwcmlpa3ZxQUZZAR6cBOWOUeJB8kGYltDh2pu_AcGVDnbrYm7_0XFn5xtMD92TU50dHyqWBxUAPA_aem_muTljS0rzEXv8UoaN30yEw)

## System Architecture

### RAG Pipeline

1. **Query Processing**: User input is encoded using sentence transformers
2. **Document Retrieval**: FAISS index searches for relevant medical documents
3. **Context Generation**: Retrieved documents are formatted as context
4. **Response Generation**: Language model generates answers using retrieved context
5. **Post-processing**: Adds medical disclaimers and formats output

### Models Used

- **Encoder**: `all-MiniLM-L6-v2` for semantic embeddings
- **Generator**: `Qwen/Qwen2.5-0.5B-Instruct` for response generation
- **Vector Database**: FAISS for efficient similarity search

## Data Sources

The system processes medical Q&A data from multiple authoritative sources:

- CancerGov Q&A
- GARD (Genetic and Rare Diseases)
- GHR (Genetics Home Reference)
- MedlinePlus Health Topics
- NIDDK, NINDS, CDC, NHLBI databases
- And more...

## RLHF Training

The system includes Reinforcement Learning from Human Feedback:

```bash
python RLHF.py
```

This trains the model on preference data to improve response quality and alignment with medical expertise.

## Medical Disclaimer

**Important**: This system is for educational and research purposes only. All responses include appropriate medical disclaimers recommending consultation with qualified healthcare professionals for accurate diagnosis and personalized medical advice.
