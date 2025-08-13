import sys
from pathlib import Path
import pandas as pd

src_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(src_path))

from utils.paths import DATA_DICT

# Define the glossary as a list of dictionaries
glossary = [
    # Programming Languages, Database Languages, Data Frameworks, Analytics Tech
    {
        "Term": "Python",
        "Acronym": None,
        "Category": "Programming Languages",
    },
    {
        "Term": "R",
        "Acronym": None,
        "Category": "Programming Languages",
    },
    {
        "Term": "Java",
        "Acronym": None,
        "Category": "Programming Languages",
    },
    {
        "Term": "C++",
        "Acronym": None,
        "Category": "Programming Languages",
    },
    {
        "Term": "C#",
        "Acronym": None,
        "Category": "Programming Languages",
    },
    {
        "Term": "JavaScript",
        "Acronym": "JS",
        "Category": "Programming Languages",
    },
    {
        "Term": "TypeScript",
        "Acronym": "TS",
        "Category": "Programming Languages",
    },
    {
        "Term": "Go",
        "Acronym": "Golang",
        "Category": "Programming Languages",
    },
    {
        "Term": "Scala",
        "Acronym": None,
        "Category": "Programming Languages",
    },
    {
        "Term": "Julia",
        "Acronym": None,
        "Category": "Programming Languages",
    },
    {
        "Term": "SQL",
        "Acronym": "Structured Query Language",
        "Category": "Database Languages",
    },
    {
        "Term": "PL/SQL",
        "Acronym": None,
        "Category": "Database Languages",
    },
    {
        "Term": "T-SQL",
        "Acronym": None,
        "Category": "Database Languages",
    },
    {
        "Term": "NoSQL",
        "Acronym": None,
        "Category": "Database Languages",
    },
    {
        "Term": "MongoDB",
        "Acronym": None,
        "Category": "Database Languages",
    },
    {
        "Term": "PostgreSQL",
        "Acronym": None,
        "Category": "Database Languages",
    },
    {
        "Term": "MySQL",
        "Acronym": None,
        "Category": "Database Languages",
    },
    {
        "Term": "SQLite",
        "Acronym": None,
        "Category": "Database Languages",
    },
    {
        "Term": "Oracle Database",
        "Acronym": None,
        "Category": "Database Languages",
    },
    {
        "Term": "Redis",
        "Acronym": None,
        "Category": "Database Languages",
    },
    {
        "Term": "Elasticsearch",
        "Acronym": None,
        "Category": "Database Languages",
    },
    {
        "Term": "Apache Hive",
        "Acronym": None,
        "Category": "Database Languages",
    },
    {
        "Term": "Apache HBase",
        "Acronym": None,
        "Category": "Database Languages",
    },
    {
        "Term": "Apache Cassandra",
        "Acronym": None,
        "Category": "Database Languages",
    },
    {
        "Term": "Apache Spark",
        "Acronym": None,
        "Category": "Data Frameworks",
    },
    {
        "Term": "Apache Hadoop",
        "Acronym": None,
        "Category": "Data Frameworks",
    },
    {
        "Term": "Apache Flink",
        "Acronym": None,
        "Category": "Data Frameworks",
    },
    {
        "Term": "Apache Beam",
        "Acronym": None,
        "Category": "Data Frameworks",
    },
    {
        "Term": "Apache Airflow",
        "Acronym": None,
        "Category": "Data Frameworks",
    },
    {
        "Term": "dbt",
        "Acronym": "data build tool",
        "Category": "Data Frameworks",
    },
    {
        "Term": "Pandas",
        "Acronym": None,
        "Category": "Python Libraries",
    },
    {
        "Term": "Dask",
        "Acronym": None,
        "Category": "Data Frameworks",
    },
    {
        "Term": "Vaex",
        "Acronym": None,
        "Category": "Data Frameworks",
    },
    {
        "Term": "Modin",
        "Acronym": None,
        "Category": "Data Frameworks",
    },
    {
        "Term": "Power BI",
        "Acronym": None,
        "Category": "Analytics Tech",
    },
    {
        "Term": "Tableau",
        "Acronym": None,
        "Category": "Analytics Tech",
    },
    {
        "Term": "Looker",
        "Acronym": None,
        "Category": "Analytics Tech",
    },
    {
        "Term": "Google Data Studio",
        "Acronym": None,
        "Category": "Analytics Tech",
    },
    {
        "Term": "Qlik",
        "Acronym": None,
        "Category": "Analytics Tech",
    },
    {
        "Term": "SAP BusinessObjects",
        "Acronym": None,
        "Category": "Analytics Tech",
    },
    {
        "Term": "Alteryx",
        "Acronym": None,
        "Category": "Analytics Tech",
    },
    {
        "Term": "Sisense",
        "Acronym": None,
        "Category": "Analytics Tech",
    },
    {
        "Term": "MicroStrategy",
        "Acronym": None,
        "Category": "Analytics Tech",
    },
    {
        "Term": "SAS",
        "Acronym": "Statistical Analysis System",
        "Category": "Analytics Tech",
    },
    {
        "Term": "SPSS",
        "Acronym": "Statistical Package for the Social Sciences",
        "Category": "Analytics Tech",
    },
    {
        "Term": "MATLAB",
        "Acronym": None,
        "Category": "Analytics Tech",
    },
    {
        "Term": "Stata",
        "Acronym": None,
        "Category": "Analytics Tech",
    },
    {
        "Term": "Jupyter",
        "Acronym": None,
        "Category": "Analytics Tech",
    },
    {
        "Term": "Excel",
        "Acronym": None,
        "Category": "Analytics Tech",
    },
    {
        "Term": "Google Sheets",
        "Acronym": None,
        "Category": "Analytics Tech",
    },
    # Core AI/ML Concepts
    {
        "Term": "Artificial Intelligence",
        "Acronym": "AI",
        "Category": "Core Concepts",
    },
    {
        "Term": "Machine Learning",
        "Acronym": "ML",
        "Category": "Core Concepts",
    },
    {
        "Term": "Deep Learning",
        "Acronym": "DL",
        "Category": "Core Concepts",
    },
    {
        "Term": "Artificial General Intelligence",
        "Acronym": "AGI",
        "Category": "Core Concepts",
    },
    {
        "Term": "Reinforcement Learning",
        "Acronym": "RL",
        "Category": "Core Concepts",
    },
    {
        "Term": "Generative AI",
        "Acronym": None,
        "Category": "Core Concepts",
    },
    {
        "Term": "Explainable AI",
        "Acronym": "XAI",
        "Category": "Core Concepts",
    },
    {
        "Term": "Federated Learning",
        "Acronym": None,
        "Category": "Core Concepts",
    },
    {
        "Term": "Edge AI",
        "Acronym": None,
        "Category": "Core Concepts",
    },
    {
        "Term": "Quantum Machine Learning",
        "Acronym": "QML",
        "Category": "Core Concepts",
    },
    # Learning Paradigms
    {
        "Term": "Supervised Learning",
        "Acronym": None,
        "Category": "Learning Paradigms",
    },
    {
        "Term": "Unsupervised Learning",
        "Acronym": None,
        "Category": "Learning Paradigms",
    },
    {
        "Term": "Semi-supervised Learning",
        "Acronym": None,
        "Category": "Learning Paradigms",
    },
    {
        "Term": "Self-supervised Learning",
        "Acronym": None,
        "Category": "Learning Paradigms",
    },
    {
        "Term": "Transfer Learning",
        "Acronym": None,
        "Category": "Learning Paradigms",
    },
    {
        "Term": "Few-shot Learning",
        "Acronym": None,
        "Category": "Learning Paradigms",
    },
    {
        "Term": "Zero-shot Learning",
        "Acronym": None,
        "Category": "Learning Paradigms",
    },
    {
        "Term": "Multi-task Learning",
        "Acronym": "MTL",
        "Category": "Learning Paradigms",
    },
    {
        "Term": "Online Learning",
        "Acronym": None,
        "Category": "Learning Paradigms",
    },
    {
        "Term": "Active Learning",
        "Acronym": None,
        "Category": "Learning Paradigms",
    },
    {
        "Term": "Curriculum Learning",
        "Acronym": None,
        "Category": "Learning Paradigms",
    },
    {
        "Term": "Meta Learning",
        "Acronym": "Learning to Learn",
        "Category": "Learning Paradigms",
    },
    # Neural Network Architectures
    {
        "Term": "Neural Networks",
        "Acronym": "NN",
        "Category": "Neural Architectures",
    },
    {
        "Term": "Convolutional Neural Networks",
        "Acronym": "CNNs",
        "Category": "Neural Architectures",
    },
    {
        "Term": "Recurrent Neural Networks",
        "Acronym": "RNNs",
        "Category": "Neural Architectures",
    },
    {
        "Term": "Long Short-Term Memory",
        "Acronym": "LSTM",
        "Category": "Neural Architectures",
    },
    {
        "Term": "Gated Recurrent Unit",
        "Acronym": "GRU",
        "Category": "Neural Architectures",
    },
    {
        "Term": "Transformer Models",
        "Acronym": None,
        "Category": "Neural Architectures",
    },
    {
        "Term": "Generative Adversarial Networks",
        "Acronym": "GANs",
        "Category": "Neural Architectures",
    },
    {
        "Term": "Variational Autoencoders",
        "Acronym": "VAEs",
        "Category": "Neural Architectures",
    },
    {
        "Term": "U-Net",
        "Acronym": None,
        "Category": "Neural Architectures",
    },
    {
        "Term": "ResNet",
        "Acronym": "Residual Networks",
        "Category": "Neural Architectures",
    },
    {
        "Term": "DenseNet",
        "Acronym": "Densely Connected Networks",
        "Category": "Neural Architectures",
    },
    {
        "Term": "EfficientNet",
        "Acronym": None,
        "Category": "Neural Architectures",
    },
    {
        "Term": "Vision Transformer",
        "Acronym": "ViT",
        "Category": "Neural Architectures",
    },
    {
        "Term": "Swin Transformer",
        "Acronym": None,
        "Category": "Neural Architectures",
    },
    {
        "Term": "ConvNeXt",
        "Acronym": None,
        "Category": "Neural Architectures",
    },
    # Large Language Models
    {
        "Term": "Large Language Models",
        "Acronym": "LLMs",
        "Category": "Language Models",
    },
    {
        "Term": "GPT",
        "Acronym": "Generative Pre-trained Transformer",
        "Category": "Language Models",
    },
    {
        "Term": "BERT",
        "Acronym": "Bidirectional Encoder Representations from Transformers",
        "Category": "Language Models",
    },
    {
        "Term": "RoBERTa",
        "Acronym": "Robustly Optimized BERT Pretraining Approach",
        "Category": "Language Models",
    },
    {
        "Term": "T5",
        "Acronym": "Text-to-Text Transfer Transformer",
        "Category": "Language Models",
    },
    {
        "Term": "BART",
        "Acronym": "Bidirectional and Auto-Regressive Transformers",
        "Category": "Language Models",
    },
    {
        "Term": "LLaMA",
        "Acronym": "Large Language Model Meta AI",
        "Category": "Language Models",
    },
    {
        "Term": "Gemini",
        "Acronym": None,
        "Category": "Language Models",
    },
    {
        "Term": "Claude",
        "Acronym": None,
        "Category": "Language Models",
    },
    {
        "Term": "PaLM",
        "Acronym": "Pathways Language Model",
        "Category": "Language Models",
    },
    {
        "Term": "Codex",
        "Acronym": None,
        "Category": "Language Models",
    },
    {
        "Term": "InstructGPT",
        "Acronym": None,
        "Category": "Language Models",
    },
    {
        "Term": "ChatGPT",
        "Acronym": None,
        "Category": "Language Models",
    },
    {
        "Term": "Qwen",
        "Acronym": None,
        "Category": "Language Models",
    },
    {
        "Term": "DeepSeek",
        "Acronym": None,
        "Category": "Language Models",
    },
    {
        "Term": "Perplexity",
        "Acronym": None,
        "Category": "Language Models",
    },
    # Computer Vision
    {
        "Term": "Computer Vision",
        "Acronym": "CV",
        "Category": "Computer Vision",
    },
    {
        "Term": "Image Classification",
        "Acronym": None,
        "Category": "Computer Vision",
    },
    {
        "Term": "Object Detection",
        "Acronym": None,
        "Category": "Computer Vision",
    },
    {
        "Term": "Image Segmentation",
        "Acronym": None,
        "Category": "Computer Vision",
    },
    {
        "Term": "Instance Segmentation",
        "Acronym": None,
        "Category": "Computer Vision",
    },
    {
        "Term": "Semantic Segmentation",
        "Acronym": None,
        "Category": "Computer Vision",
    },
    {
        "Term": "Panoptic Segmentation",
        "Acronym": None,
        "Category": "Computer Vision",
    },
    {
        "Term": "Face Recognition",
        "Acronym": None,
        "Category": "Computer Vision",
    },
    {
        "Term": "Face Detection",
        "Acronym": None,
        "Category": "Computer Vision",
    },
    {
        "Term": "Optical Character Recognition",
        "Acronym": "OCR",
        "Category": "Computer Vision",
    },
    {
        "Term": "Image Generation",
        "Acronym": None,
        "Category": "Computer Vision",
    },
    {
        "Term": "Style Transfer",
        "Acronym": None,
        "Category": "Computer Vision",
    },
    {
        "Term": "Image Super-resolution",
        "Acronym": None,
        "Category": "Computer Vision",
    },
    {
        "Term": "Video Analysis",
        "Acronym": None,
        "Category": "Computer Vision",
    },
    {
        "Term": "3D Vision",
        "Acronym": None,
        "Category": "Computer Vision",
    },
    # Natural Language Processing
    {
        "Term": "Natural Language Processing",
        "Acronym": "NLP",
        "Category": "Natural Language Processing",
    },
    {
        "Term": "Text Classification",
        "Acronym": None,
        "Category": "Natural Language Processing",
    },
    {
        "Term": "Named Entity Recognition",
        "Acronym": "NER",
        "Category": "Natural Language Processing",
    },
    {
        "Term": "Sentiment Analysis",
        "Acronym": None,
        "Category": "Natural Language Processing",
    },
    {
        "Term": "Machine Translation",
        "Acronym": None,
        "Category": "Natural Language Processing",
    },
    {
        "Term": "Question Answering",
        "Acronym": "QA",
        "Category": "Natural Language Processing",
    },
    {
        "Term": "Text Summarization",
        "Acronym": None,
        "Category": "Natural Language Processing",
    },
    {
        "Term": "Text Generation",
        "Acronym": None,
        "Category": "Natural Language Processing",
    },
    {
        "Term": "Language Modeling",
        "Acronym": None,
        "Category": "Natural Language Processing",
    },
    {
        "Term": "Part-of-Speech Tagging",
        "Acronym": "POS Tagging",
        "Category": "Natural Language Processing",
    },
    {
        "Term": "Dependency Parsing",
        "Acronym": None,
        "Category": "Natural Language Processing",
    },
    {
        "Term": "Coreference Resolution",
        "Acronym": None,
        "Category": "Natural Language Processing",
    },
    {
        "Term": "Information Extraction",
        "Acronym": "IE",
        "Category": "Natural Language Processing",
    },
    {
        "Term": "Text Mining",
        "Acronym": None,
        "Category": "Natural Language Processing",
    },
    {
        "Term": "Dialogue Systems",
        "Acronym": None,
        "Category": "Natural Language Processing",
    },
    # Speech and Audio
    {
        "Term": "Speech Recognition",
        "Acronym": "ASR",
        "Category": "Speech & Audio",
    },
    {
        "Term": "Text-to-Speech",
        "Acronym": "TTS",
        "Category": "Speech & Audio",
    },
    {
        "Term": "Speech Synthesis",
        "Acronym": None,
        "Category": "Speech & Audio",
    },
    {
        "Term": "Speaker Recognition",
        "Acronym": None,
        "Category": "Speech & Audio",
    },
    {
        "Term": "Speaker Diarization",
        "Acronym": None,
        "Category": "Speech & Audio",
    },
    {
        "Term": "Audio Classification",
        "Acronym": None,
        "Category": "Speech & Audio",
    },
    {
        "Term": "Music Generation",
        "Acronym": None,
        "Category": "Speech & Audio",
    },
    {
        "Term": "Audio Enhancement",
        "Acronym": None,
        "Category": "Speech & Audio",
    },
    {
        "Term": "Voice Cloning",
        "Acronym": None,
        "Category": "Speech & Audio",
    },
    {
        "Term": "Emotion Recognition",
        "Acronym": None,
        "Category": "Speech & Audio",
    },
    {
        "Term": "Accent Recognition",
        "Acronym": None,
        "Category": "Speech & Audio",
    },
    {
        "Term": "Audio Segmentation",
        "Acronym": None,
        "Category": "Speech & Audio",
    },
    {
        "Term": "Music Information Retrieval",
        "Acronym": "MIR",
        "Category": "Speech & Audio",
    },
    {
        "Term": "Audio Fingerprinting",
        "Acronym": None,
        "Category": "Speech & Audio",
    },
    {
        "Term": "Speech Enhancement",
        "Acronym": None,
        "Category": "Speech & Audio",
    },
    # Traditional ML Algorithms
    {
        "Term": "Linear Regression",
        "Acronym": None,
        "Category": "Traditional ML",
    },
    {
        "Term": "Logistic Regression",
        "Acronym": None,
        "Category": "Traditional ML",
    },
    {
        "Term": "Decision Trees",
        "Acronym": None,
        "Category": "Traditional ML",
    },
    {
        "Term": "Random Forests",
        "Acronym": None,
        "Category": "Traditional ML",
    },
    {
        "Term": "Gradient Boosting",
        "Acronym": None,
        "Category": "Traditional ML",
    },
    {
        "Term": "XGBoost",
        "Acronym": "Extreme Gradient Boosting",
        "Category": "Traditional ML",
    },
    {
        "Term": "LightGBM",
        "Acronym": "Light Gradient Boosting Machine",
        "Category": "Traditional ML",
    },
    {
        "Term": "CatBoost",
        "Acronym": None,
        "Category": "Traditional ML",
    },
    {
        "Term": "Support Vector Machines",
        "Acronym": "SVM",
        "Category": "Traditional ML",
    },
    {
        "Term": "k-Nearest Neighbors",
        "Acronym": "k-NN",
        "Category": "Traditional ML",
    },
    {
        "Term": "Naive Bayes",
        "Acronym": None,
        "Category": "Traditional ML",
    },
    {
        "Term": "K-means Clustering",
        "Acronym": None,
        "Category": "Traditional ML",
    },
    {
        "Term": "Hierarchical Clustering",
        "Acronym": None,
        "Category": "Traditional ML",
    },
    {
        "Term": "Principal Component Analysis",
        "Acronym": "PCA",
        "Category": "Traditional ML",
    },
    {
        "Term": "Singular Value Decomposition",
        "Acronym": "SVD",
        "Category": "Traditional ML",
    },
    # Python ML Libraries
    {
        "Term": "TensorFlow",
        "Acronym": None,
        "Category": "Python Libraries",
    },
    {
        "Term": "PyTorch",
        "Acronym": None,
        "Category": "Python Libraries",
    },
    {
        "Term": "Scikit-learn",
        "Acronym": None,
        "Category": "Python Libraries",
    },
    {
        "Term": "Keras",
        "Acronym": None,
        "Category": "Python Libraries",
    },
    {
        "Term": "XGBoost",
        "Acronym": None,
        "Category": "Python Libraries",
    },
    {
        "Term": "LightGBM",
        "Acronym": None,
        "Category": "Python Libraries",
    },
    {
        "Term": "Hugging Face Transformers",
        "Acronym": None,
        "Category": "Python Libraries",
    },
    {
        "Term": "NumPy",
        "Acronym": None,
        "Category": "Python Libraries",
    },
    {
        "Term": "Pandas",
        "Acronym": None,
        "Category": "Python Libraries",
    },
    {
        "Term": "Matplotlib",
        "Acronym": None,
        "Category": "Python Libraries",
    },
    {
        "Term": "Seaborn",
        "Acronym": None,
        "Category": "Python Libraries",
    },
    {
        "Term": "Plotly",
        "Acronym": None,
        "Category": "Python Libraries",
    },
    {
        "Term": "OpenCV",
        "Acronym": None,
        "Category": "Python Libraries",
    },
    {
        "Term": "Pillow",
        "Acronym": "PIL",
        "Category": "Python Libraries",
    },
    {
        "Term": "JAX",
        "Acronym": None,
        "Category": "Python Libraries",
    },
    {
        "Term": "FastAI",
        "Acronym": None,
        "Category": "Python Libraries",
    },
    {
        "Term": "Optuna",
        "Acronym": None,
        "Category": "Python Libraries",
    },
    {
        "Term": "Ray",
        "Acronym": None,
        "Category": "Python Libraries",
    },
    {
        "Term": "Dask",
        "Acronym": None,
        "Category": "Python Libraries",
    },
    {
        "Term": "Vaex",
        "Acronym": None,
        "Category": "Python Libraries",
    },
    {
        "Term": "Modin",
        "Acronym": None,
        "Category": "Python Libraries",
    },
    # MLOps and Deployment
    {
        "Term": "MLOps",
        "Acronym": "Machine Learning Operations",
        "Category": "MLOps & Deployment",
    },
    {
        "Term": "Model Serving",
        "Acronym": None,
        "Category": "MLOps & Deployment",
    },
    {
        "Term": "Model Registry",
        "Acronym": None,
        "Category": "MLOps & Deployment",
    },
    {
        "Term": "Model Versioning",
        "Acronym": None,
        "Category": "MLOps & Deployment",
    },
    {
        "Term": "Model Monitoring",
        "Acronym": None,
        "Category": "MLOps & Deployment",
    },
    {
        "Term": "A/B Testing",
        "Acronym": None,
        "Category": "MLOps & Deployment",
    },
    {
        "Term": "Canary Deployment",
        "Acronym": None,
        "Category": "MLOps & Deployment",
    },
    {
        "Term": "Blue-Green Deployment",
        "Acronym": None,
        "Category": "MLOps & Deployment",
    },
    {
        "Term": "Feature Store",
        "Acronym": None,
        "Category": "MLOps & Deployment",
    },
    {
        "Term": "Data Pipeline",
        "Acronym": None,
        "Category": "MLOps & Deployment",
    },
    {
        "Term": "ETL",
        "Acronym": "Extract, Transform, Load",
        "Category": "MLOps & Deployment",
    },
    {
        "Term": "Data Lineage",
        "Acronym": None,
        "Category": "MLOps & Deployment",
    },
    {
        "Term": "Model Interpretability",
        "Acronym": None,
        "Category": "MLOps & Deployment",
    },
    {
        "Term": "Model Explainability",
        "Acronym": None,
        "Category": "MLOps & Deployment",
    },
    {
        "Term": "Model Governance",
        "Acronym": None,
        "Category": "MLOps & Deployment",
    },
    # Cloud and Infrastructure
    {
        "Term": "AWS SageMaker",
        "Acronym": None,
        "Category": "Cloud Platforms",
    },
    {
        "Term": "Google Cloud AI Platform",
        "Acronym": None,
        "Category": "Cloud Platforms",
    },
    {
        "Term": "Azure Machine Learning",
        "Acronym": None,
        "Category": "Cloud Platforms",
    },
    {
        "Term": "Databricks",
        "Acronym": None,
        "Category": "Cloud Platforms",
    },
    {
        "Term": "Snowflake",
        "Acronym": None,
        "Category": "Cloud Platforms",
    },
    {
        "Term": "Kubernetes",
        "Acronym": "K8s",
        "Category": "Cloud Platforms",
    },
    {
        "Term": "Docker",
        "Acronym": None,
        "Category": "Cloud Platforms",
    },
    {
        "Term": "Terraform",
        "Acronym": None,
        "Category": "Cloud Platforms",
    },
    {
        "Term": "Apache Airflow",
        "Acronym": None,
        "Category": "Cloud Platforms",
    },
    {
        "Term": "Apache Spark",
        "Acronym": None,
        "Category": "Cloud Platforms",
    },
    {
        "Term": "Apache Kafka",
        "Acronym": None,
        "Category": "Cloud Platforms",
    },
    {
        "Term": "Redis",
        "Acronym": None,
        "Category": "Cloud Platforms",
    },
    {
        "Term": "Elasticsearch",
        "Acronym": None,
        "Category": "Cloud Platforms",
    },
    {
        "Term": "MongoDB",
        "Acronym": None,
        "Category": "Cloud Platforms",
    },
    {
        "Term": "PostgreSQL",
        "Acronym": None,
        "Category": "Cloud Platforms",
    },
    # Job Roles and Careers
    {
        "Term": "Data Scientist",
        "Acronym": None,
        "Category": "Job Roles",
    },
    {
        "Term": "Machine Learning Engineer",
        "Acronym": None,
        "Category": "Job Roles",
    },
    {
        "Term": "Data Analyst",
        "Acronym": None,
        "Category": "Job Roles",
    },
    {
        "Term": "AI Researcher",
        "Acronym": None,
        "Category": "Job Roles",
    },
    {
        "Term": "Data Engineer",
        "Acronym": None,
        "Category": "Job Roles",
    },
    {
        "Term": "AI Product Manager",
        "Acronym": None,
        "Category": "Job Roles",
    },
    {
        "Term": "ML Ops Engineer",
        "Acronym": None,
        "Category": "Job Roles",
    },
    {
        "Term": "Research Scientist",
        "Acronym": None,
        "Category": "Job Roles",
    },
    {
        "Term": "Applied Scientist",
        "Acronym": None,
        "Category": "Job Roles",
    },
    {
        "Term": "AI Ethics Specialist",
        "Acronym": None,
        "Category": "Job Roles",
    },
    {
        "Term": "Prompt Engineer",
        "Acronym": None,
        "Category": "Job Roles",
    },
    {
        "Term": "Computer Vision Engineer",
        "Acronym": None,
        "Category": "Job Roles",
    },
    {
        "Term": "NLP Engineer",
        "Acronym": None,
        "Category": "Job Roles",
    },
    {
        "Term": "Robotics Engineer",
        "Acronym": None,
        "Category": "Job Roles",
    },
    {
        "Term": "AI Solutions Architect",
        "Acronym": None,
        "Category": "Job Roles",
    },
    # Emerging Technologies
    {
        "Term": "Quantum Computing",
        "Acronym": None,
        "Category": "Emerging Tech",
    },
    {
        "Term": "Edge Computing",
        "Acronym": None,
        "Category": "Emerging Tech",
    },
    {
        "Term": "Federated Learning",
        "Acronym": None,
        "Category": "Emerging Tech",
    },
    {
        "Term": "AutoML",
        "Acronym": "Automated Machine Learning",
        "Category": "Emerging Tech",
    },
    {
        "Term": "Neural Architecture Search",
        "Acronym": "NAS",
        "Category": "Emerging Tech",
    },
    {
        "Term": "Few-shot Learning",
        "Acronym": None,
        "Category": "Emerging Tech",
    },
    {
        "Term": "Zero-shot Learning",
        "Acronym": None,
        "Category": "Emerging Tech",
    },
    {
        "Term": "Meta-learning",
        "Acronym": "Learning to Learn",
        "Category": "Emerging Tech",
    },
    {
        "Term": "Continual Learning",
        "Acronym": "Lifelong Learning",
        "Category": "Emerging Tech",
    },
    {
        "Term": "Self-supervised Learning",
        "Acronym": None,
        "Category": "Emerging Tech",
    },
    {
        "Term": "Contrastive Learning",
        "Acronym": None,
        "Category": "Emerging Tech",
    },
    {
        "Term": "Adversarial Training",
        "Acronym": None,
        "Category": "Emerging Tech",
    },
    {
        "Term": "Knowledge Distillation",
        "Acronym": None,
        "Category": "Emerging Tech",
    },
    {
        "Term": "Model Compression",
        "Acronym": None,
        "Category": "Emerging Tech",
    },
    {
        "Term": "Neural Pruning",
        "Acronym": None,
        "Category": "Emerging Tech",
    },
    # Data and Preprocessing
    {
        "Term": "Features",
        "Acronym": None,
        "Category": "Data & Preprocessing",
    },
    {
        "Term": "Labels",
        "Acronym": None,
        "Category": "Data & Preprocessing",
    },
    {
        "Term": "Normalization",
        "Acronym": None,
        "Category": "Data & Preprocessing",
    },
    {
        "Term": "Standardization",
        "Acronym": None,
        "Category": "Data & Preprocessing",
    },
    {
        "Term": "One-Hot Encoding",
        "Acronym": None,
        "Category": "Data & Preprocessing",
    },
    {
        "Term": "Linear Regression",
        "Acronym": None,
        "Category": "Algorithms & Models",
    },
    {
        "Term": "Logistic Regression",
        "Acronym": None,
        "Category": "Algorithms & Models",
    },
    {
        "Term": "Decision Trees",
        "Acronym": None,
        "Category": "Algorithms & Models",
    },
    {
        "Term": "Random Forests",
        "Acronym": None,
        "Category": "Algorithms & Models",
    },
    {
        "Term": "Support Vector Machines",
        "Acronym": "SVM",
        "Category": "Algorithms & Models",
    },
    {
        "Term": "k-Nearest Neighbors",
        "Acronym": "k-NN",
        "Category": "Algorithms & Models",
    },
    {
        "Term": "Naive Bayes",
        "Acronym": None,
        "Category": "Algorithms & Models",
    },
    {
        "Term": "Neural Networks",
        "Acronym": None,
        "Category": "Algorithms & Models",
    },
    {
        "Term": "Convolutional Neural Networks",
        "Acronym": "CNNs",
        "Category": "Algorithms & Models",
    },
    {
        "Term": "Recurrent Neural Networks",
        "Acronym": "RNNs",
        "Category": "Algorithms & Models",
    },
    {
        "Term": "Long Short-Term Memory",
        "Acronym": "LSTM",
        "Category": "Algorithms & Models",
    },
    {
        "Term": "Generative Adversarial Networks",
        "Acronym": "GANs",
        "Category": "Algorithms & Models",
    },
    {
        "Term": "Transformer Models",
        "Acronym": None,
        "Category": "Algorithms & Models",
    },
    {
        "Term": "Accuracy",
        "Acronym": None,
        "Category": "Evaluation Metrics",
    },
    {
        "Term": "Precision",
        "Acronym": None,
        "Category": "Evaluation Metrics",
    },
    {
        "Term": "Recall",
        "Acronym": None,
        "Category": "Evaluation Metrics",
    },
    {
        "Term": "F1 Score",
        "Acronym": None,
        "Category": "Evaluation Metrics",
    },
    {
        "Term": "Confusion Matrix",
        "Acronym": None,
        "Category": "Evaluation Metrics",
    },
    {
        "Term": "ROC Curve",
        "Acronym": None,
        "Category": "Evaluation Metrics",
    },
    {
        "Term": "AUC",
        "Acronym": "Area Under Curve",
        "Category": "Evaluation Metrics",
    },
    {
        "Term": "TensorFlow",
        "Acronym": None,
        "Category": "Tools & Frameworks",
    },
    {
        "Term": "PyTorch",
        "Acronym": None,
        "Category": "Tools & Frameworks",
    },
    {
        "Term": "Scikit-learn",
        "Acronym": None,
        "Category": "Tools & Frameworks",
    },
    {
        "Term": "Keras",
        "Acronym": None,
        "Category": "Tools & Frameworks",
    },
    {
        "Term": "XGBoost",
        "Acronym": None,
        "Category": "Tools & Frameworks",
    },
    {
        "Term": "LightGBM",
        "Acronym": None,
        "Category": "Tools & Frameworks",
    },
    {
        "Term": "Hugging Face Transformers",
        "Acronym": None,
        "Category": "Tools & Frameworks",
    },
    {
        "Term": "Recommender Systems",
        "Acronym": None,
        "Category": "Domains & Applications",
    },
    {
        "Term": "Anomaly Detection",
        "Acronym": None,
        "Category": "Domains & Applications",
    },
    {
        "Term": "Robotic Process Automation",
        "Acronym": "RPA",
        "Category": "Domains & Applications",
    },
    {
        "Term": "Autonomous Vehicles",
        "Acronym": None,
        "Category": "Domains & Applications",
    },
    {
        "Term": "Feature Engineering",
        "Acronym": None,
        "Category": "Data & Preprocessing",
    },
    {
        "Term": "Data Augmentation",
        "Acronym": None,
        "Category": "Data & Preprocessing",
    },
]
if __name__ == "__main__":
    df_glossary = pd.DataFrame(glossary)
    df_glossary.to_csv(DATA_DICT["models"]["weak"]["kws"], index=False)
