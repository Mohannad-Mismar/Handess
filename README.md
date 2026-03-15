# 🏛️ Handess: Intelligent Engineering Compliance Assistant

**Handess** is a specialized Retrieval-Augmented Generation (RAG) system designed to navigate the complex landscape of Jordanian structural engineering regulations. By transforming "Dark Data" into actionable intelligence, Handess helps engineers achieve 100% compliance on their first submission.

---

## 🚀 The Challenge: Conquering "Dark Data"

In the Jordanian AEC sector, critical regulatory data often exists in non-searchable PDF formats and scanned tables.

* 
**High Rejection Rates:** Complex, overlapping amendments lead to frequent plan rejections at the municipal level .


* 
**Hallucination Risk:** General-purpose AI models lack access to these closed-source local documents and often "guess" the legal requirements .


* 
**Operational Bottlenecks:** Manual document review is time-consuming and prone to human error.



## 🛠️ The Solution

Handess solves this by architecting a strictly grounded AI pipeline that out-performs standard LLMs in localized precision.

### Key Features:

* 
**Custom Ingestion Pipeline:** Digitizes unsearchable PDFs into structured Markdown while preserving legal hierarchies.


* 
**Advanced Semantic Retrieval:** Utilizes `multilingual-e5-large` embeddings and a **FAISS** vector store with **Maximal Marginal Relevance (MMR)** for diverse and accurate clause fetching .


* 
**Arabic-Native Intelligence:** Powered by the **Fanar LLM** for nuanced understanding of Arabic legal syntax.


* 
**Zero-Hallucination Guardrails:** Implements strict refusal mechanisms for out-of-scope queries.
  

## 🏗️ Technical Architecture

1. 
**Frontend:** Lightweight HTML/CSS/JavaScript chat interface .


2. 
**Backend:** Flask API orchestrating the RAG workflow via LangChain.


3. 
**Data Layer:** Structured Markdown/JSONL knowledge base stored in a local FAISS index .



## 🗺️ Roadmap

* [ ] **MEP Integration:** Expanding coverage to Mechanical, Electrical, and Plumbing codes.


* [ ] **Multimodal Retrieval:** Enabling the system to "see" and interpret site diagrams .


* [ ] **CAD/Revit Plugins:** Direct "Design-to-Compliance" workflows within drafting software .



## 👨‍💻 Author

**Mohannad Mismar**
* Computer Science Student | German Jordanian University 


* Erasmus+ Scholar | Hochschule Offenburg 


* [LinkedIn](https://www.linkedin.com/in/mohannad-mismar/) | [GitHub](https://github.com/Mohannad-Mismar) 
