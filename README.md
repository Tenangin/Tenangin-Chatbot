# Tenangin Chatbot Development

**Tenangin Chatbot** is a Retrieval-Augmented Generation (RAG) based chatbot designed to provide information and answers to questions about mental health. This project uses a collection of data from books and blog articles to build its knowledge base and leverages it to provide relevant and informative responses.

This is a repository for developing feature Chatbot RAG in Tenangin Applications. 

## Table of Contents

1.  [Overview](https://www.google.com/search?q=%23overview)
2.  [System Architecture](https://www.google.com/search?q=%23system-architecture)
3.  [Directory Structure](https://www.google.com/search?q=%23directory-structure)
4.  [Project Workflow](https://www.google.com/search?q=%23project-workflow)
5.  [How to Run](https://www.google.com/search?q=%23how-to-run)
6.  [Evaluation](https://www.google.com/search?q=%23evaluation)
7.  [Contributors](https://www.google.com/search?q=%23contributors)

## Overview

In today's digital era, access to accurate mental health information is crucial. Tenangin Chatbot was created to provide an easily accessible Q\&A platform. Unlike general-purpose language models, this chatbot uses the RAG approach, which allows it to retrieve factual information from trusted sources (books and articles) before generating an answer. This ensures that the responses given are not only contextually relevant but also based on verified data.

## System Architecture

The system is built with a RAG architecture that consists of several main stages:

1.  **Data Collection**: Data is gathered from various sources, including a book (`Kesehatan Mental Book 1.txt`) and relevant blog articles (`scraped_articles.csv`).

2.  **Preprocessing & Chunking**: The raw text from data sources is cleaned of unnecessary elements. Then, long documents are split into smaller parts (*chunks*) to be more easily processed by the model. This process is documented in the notebooks within the `Preprocessing/` directory.

3.  **Embedding Generation**: Each text chunk is converted into a numerical vector representation (embedding) using a language model. These embeddings capture the semantic meaning of the text, allowing the system to search and compare text similarity efficiently. This process is handled in `Embedding/embedding.ipynb`.

4.  **Information Retrieval**: When a user asks a question, the query is also converted into an embedding. The system then uses a similarity search (e.g., cosine similarity) to find the text chunks from the knowledge base that are most relevant to the user's question.

5.  **Answer Generation**: The most relevant text chunks are combined with the user's original question. This combined text is then fed into a Large Language Model (LLM) as context. The LLM's task is to generate a coherent, fluent, and contextual answer based on the provided information. This inference process can be seen in `Inference/inference.py`.

6. **With Framework Langchain** : This Chabot RAG using framework **Langchain** for fast and sustainable development.

## Directory Structure

Here is an explanation of the directory structure and key files in this repository:

```
.
├── Data/
│   ├── Kesehatan Mental Book 1.txt  # Data source from a book
│   ├── scraped_articles.csv         # Data source from web articles
│   ├── link_blogs_clean.csv         # List of blog links used
│   ├── dokumen3.txt                 # Additional data document
│   └── dokumen4.txt                 # Additional data document
│
├── Preprocessing/
│   ├── assesing.ipynb               # Notebook for initial data analysis
│   ├── preprocessing.ipynb          # Main notebook for data cleaning
│   └── preprocessing_chonkie.ipynb  # Notebook for the data chunking process
│
├── Embedding/
│   └── embedding.ipynb              # Notebook to create vector embeddings
│
├── Inference/
│   ├── inference.ipynb              # Notebook for chatbot inference testing
│   └── inference.py                 # Python script to run the chatbot
│
├── evaluasi-tenobot.csv             # Chatbot performance evaluation results
└── evaluate.ipynb                   # Notebook for evaluating chatbot responses
```

## Project Workflow

1.  **Data Collection**: Gather relevant data from the `.txt` and `.csv` files located in the `Data/` directory.
2.  **Data Assessment & Cleaning**: Analyze the data quality using `assesing.ipynb` and clean it with the scripts in `preprocessing.ipynb`.
3.  **Chunking**: Split the cleaned data into smaller chunks using `preprocessing_chonkie.ipynb`.
4.  **Embedding**: Run `embedding.ipynb` to convert the text chunks into vectors and save them (likely in a vector store like FAISS or ChromaDB, though not explicitly specified).
5.  **Inference**: Execute `inference.py` to interact with the chatbot. This script will take user input, search for relevant information, and generate an answer.
6.  **Evaluation**: Use `evaluate.ipynb` to measure the quality of the chatbot's answers based on specific metrics, with the results stored in `evaluasi-tenobot.csv`.

## How to Run

To run this chatbot in your local environment, follow these steps:

1.  **Clone the Repository** (if hosted)

    ```bash
    git clone [YOUR_REPOSITORY_URL]
    cd tenangin-chatbot
    ```

2.  **Create a Virtual Environment**

    ```bash
    python -m venv venv
    source venv/bin/activate  # For Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    *Note: A `requirements.txt` file was not found. You will need to create one based on the libraries used in the notebooks (`pandas`, `transformers`, `torch`, `scikit-learn`, etc.).*

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Data Pipeline**
    Ensure you have run the notebooks in the `Preprocessing/` and `Embedding/` directories sequentially to process the data and create the necessary embedding files.

5.  **Run the Chatbot**
    Use the inference script to start the chatbot.

    ```bash
    python Inference/inference.py
    ```

    Alternatively, run the code cells inside `Inference/inference.ipynb`.

## Evaluation

The chatbot's performance is evaluated to ensure the quality of its responses. The evaluation process is carried out using `evaluate.ipynb`, which may compare the generated answers against reference answers or use RAG evaluation metrics like *faithfulness* and *answer relevancy*. The quantitative results of this evaluation are stored in the `evaluasi-tenobot.csv` file.

## Contributors

  * **Muhammad Rizki Al-Fathir** - *Informatics Engineering Student*
