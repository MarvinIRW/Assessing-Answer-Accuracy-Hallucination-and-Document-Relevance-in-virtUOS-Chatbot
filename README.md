# RAG-Based Chatbot for Osnabrück University

Welcome to the official repository for my bachelor’s thesis on a domain-specific, **Retrieval-Augmented Generation (RAG)** chatbot, built to answer questions from prospective, enrolled, and international students at Osnabrück University. This chatbot is **forked from [askUOS](https://github.com/virtUOS/askUOS)** and was extensively evaluated on dimensions such as **hallucination, accuracy, user satisfaction, coherence, clarity, fluency,** and **context quality** in both English and German.

## Thesis in Brief

The thesis investigates whether a RAG chatbot—one that retrieves and incorporates relevant university-specific documents into its answers—can produce usable answers in the domain-specific context of a university. I conducted:
- **Human evaluations** to rate responses on accuracy, hallucinations, user satisfaction, etc.
- **Automated evaluations** (e.g., BLEU, ROUGE, BERTScore, BARTScore, and an LLM-as-a-Judge approach).
- **Correlation analyses** to compare human-annotated scores with automated metrics.
  
Results indicate that while retrieval grounding produces mostly hallucination free and coherent, clear, and fluent answers, there is still variability in accuracy. The thesis underscores the complexity of evaluating and deploying domain-specific chatbots, advocating for a multi-method approach that combines automated checks and targeted human assessments.

## Repository Structure

Below is an overview of the main folders in this repository.

```
.
├── chatbot
├── code
│   ├── dataset
│   └── eval
├── data
    ├── answer_survey
    ├── chatbot_data
    ├── eval
    ├── figures
    ├── human_eval
    └── question_survey

```

### 1. `chatbot/`
- **Description**: Contains the **chatbot implementation**, forked at the 8th of January 2025 from [askUOS](https://github.com/virtUOS/askUOS).  
- **Note**: This is where you can explore the code responsible for the actual chatbot logic, RAG pipeline, and environment setup.

### 2. `code/`
- **Description**: A collection of Jupyter notebooks for **data processing, cleaning, survey allocation**, and **evaluation scripts**.
- **Subfolders**:
  - `dataset/`  
    - Notebooks that perform tasks like:
      - Cleaning questions and answers.
      - Allocating and merging data for the surveys.
      - Generating the final curated dataset used to evaluate the chatbot.
  - `eval/`  
    - Scripts and notebooks for **automatic** and **human** evaluation.
    - Further subdivided into:
      - `automatic_eval/`: Contains correlation analysis, lexical metrics, semantic metrics, and the LLM-as-a-Judge approach.
      - `human_eval/`: Contains notebooks for preparing and cleaning the human evaluation surveys, analyzing annotator agreement, and generating summary statistics.

### 3. `data/`
- **Description**: The **core data** used for this thesis, including **question-answer pairs**, **human evaluation results**, **chatbot outputs**, and **reference data**.
- **Notable Contents**:
  - `answer_survey/`:
    - Contains CSV files with **reference answers** (human-written) and data from the **answer collection surveys**.
    - `.lss` files (LimeSurvey exports) for the different survey setups.
  - `chatbot_data/`:
    - Final **chatbot responses** and metadata, including the CSV with all Q&A pairs used in the evaluation.
  - `eval/`:
    - CSV files with **evaluation results** from automatic metrics and from the LLM-as-a-Judge approach.
  - `figures/`:
    - Various **plots and figures** illustrating demographic distributions, evaluation metrics, correlation heatmaps, etc.
  - `human_eval/`:
    - Cleaned and raw data from **human annotators** (ratings on accuracy, hallucination, coherence, etc.).
  - `question_survey/`:
    - CSVs with **initial and cleaned user questions** for the chatbot.

## How to Use This Repository

1. **Chatbot Exploration**  
   - Navigate to `chatbot/` to see the RAG-based chatbot code. This is the fork of [askUOS](https://github.com/virtUOS/askUOS).

2. **Data Processing & Analysis**  
   - Look in `code/` for detailed **data cleaning** and **evaluation** notebooks. Each notebook is named to reflect its processing step or metric computation.

3. **Datasets & Results**  
   - The `data/` folder contains all relevant CSVs for questions, answers, and survey data. You can replicate or re-check the analysis using the notebooks in `code/eval/`.

4. **Figures & Visuals**  
   - In `data/figures/`, you will find the **plots** used in the thesis to illustrate metric correlations, demographic distributions, and performance comparisons.

## Contact

If you have questions or suggestions regarding this research or the repository structure, feel free to open an issue or contact me directly at mwurch@uni-osnabrueck.de.

Happy exploring, and thanks for your interest in this RAG chatbot project!