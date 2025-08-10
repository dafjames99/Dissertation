import pandas as pd

# Define the glossary as a list of dictionaries
glossary = [
    {"Term": "Artificial Intelligence", "Acronym": "AI", "Category": "Core Concepts", "Definition": "The simulation of human intelligence in machines capable of performing tasks that typically require human cognition, such as reasoning, learning, and problem-solving."},
    {"Term": "Machine Learning", "Acronym": "ML", "Category": "Core Concepts", "Definition": "A subset of AI focused on developing algorithms that enable computers to learn from and make predictions based on data, without explicit programming."},
    {"Term": "Deep Learning", "Acronym": "DL", "Category": "Core Concepts", "Definition": "A specialized area within ML that utilizes neural networks with many layers (deep neural networks) to model complex patterns in large datasets."},
    {"Term": "Artificial General Intelligence", "Acronym": "AGI", "Category": "Core Concepts", "Definition": "A theoretical form of AI that possesses the ability to understand, learn, and apply knowledge across a wide range of tasks, exhibiting human-like cognitive abilities."},
    {"Term": "Reinforcement Learning", "Acronym": "RL", "Category": "Core Concepts", "Definition": "A type of ML where an agent learns to make decisions by performing actions in an environment to maximize cumulative reward."},

    {"Term": "Supervised Learning", "Acronym": None, "Category": "Learning Paradigms", "Definition": "ML where the model is trained on labeled data, learning to map inputs to known outputs."},
    {"Term": "Unsupervised Learning", "Acronym": None, "Category": "Learning Paradigms", "Definition": "ML where the model identifies patterns and structures in unlabeled data."},
    {"Term": "Semi-supervised Learning", "Acronym": None, "Category": "Learning Paradigms", "Definition": "A hybrid approach using a small amount of labeled data and a large amount of unlabeled data."},
    {"Term": "Self-supervised Learning", "Acronym": None, "Category": "Learning Paradigms", "Definition": "A form of unsupervised learning where the system generates its own labels from the input data."},
    {"Term": "Transfer Learning", "Acronym": None, "Category": "Learning Paradigms", "Definition": "Applying knowledge gained from solving one problem to a different but related problem."},
    {"Term": "Few-shot Learning", "Acronym": None, "Category": "Learning Paradigms", "Definition": "Training models to learn from a very small number of examples."},
    {"Term": "Zero-shot Learning", "Acronym": None, "Category": "Learning Paradigms", "Definition": "Enabling models to make predictions for tasks or classes they have not seen during training."},

    {"Term": "Linear Regression", "Acronym": None, "Category": "Algorithms & Models", "Definition": "A statistical method for modeling the relationship between a dependent variable and one or more independent variables."},
    {"Term": "Logistic Regression", "Acronym": None, "Category": "Algorithms & Models", "Definition": "A regression model used for binary classification tasks."},
    {"Term": "Decision Trees", "Acronym": None, "Category": "Algorithms & Models", "Definition": "A model that splits data into subsets based on feature values, forming a tree-like structure."},
    {"Term": "Random Forests", "Acronym": None, "Category": "Algorithms & Models", "Definition": "An ensemble method using multiple decision trees to improve classification accuracy."},
    {"Term": "Support Vector Machines", "Acronym": "SVM", "Category": "Algorithms & Models", "Definition": "A supervised learning model that finds the hyperplane that best separates classes in the feature space."},
    {"Term": "k-Nearest Neighbors", "Acronym": "k-NN", "Category": "Algorithms & Models", "Definition": "A non-parametric method used for classification and regression by comparing input data to its k-nearest neighbors."},
    {"Term": "Naive Bayes", "Acronym": None, "Category": "Algorithms & Models", "Definition": "A probabilistic classifier based on applying Bayes' theorem with strong (naive) independence assumptions."},
    {"Term": "Neural Networks", "Acronym": None, "Category": "Algorithms & Models", "Definition": "Computational models inspired by the human brain, consisting of layers of interconnected nodes (neurons)."},
    {"Term": "Convolutional Neural Networks", "Acronym": "CNNs", "Category": "Algorithms & Models", "Definition": "Specialized neural networks for processing structured grid data, such as images."},
    {"Term": "Recurrent Neural Networks", "Acronym": "RNNs", "Category": "Algorithms & Models", "Definition": "Neural networks designed for sequential data, where outputs are dependent on previous computations."},
    {"Term": "Long Short-Term Memory", "Acronym": "LSTM", "Category": "Algorithms & Models", "Definition": "A type of RNN capable of learning long-term dependencies."},
    {"Term": "Generative Adversarial Networks", "Acronym": "GANs", "Category": "Algorithms & Models", "Definition": "A framework where two neural networks contest with each other to generate new, synthetic instances of data."},
    {"Term": "Transformer Models", "Acronym": None, "Category": "Algorithms & Models", "Definition": "Models that rely on self-attention mechanisms to process sequences of data, such as in NLP tasks."},

    {"Term": "Accuracy", "Acronym": None, "Category": "Evaluation Metrics", "Definition": "The proportion of correct predictions made by the model."},
    {"Term": "Precision", "Acronym": None, "Category": "Evaluation Metrics", "Definition": "The ratio of correctly predicted positive observations to the total predicted positives."},
    {"Term": "Recall", "Acronym": None, "Category": "Evaluation Metrics", "Definition": "The ratio of correctly predicted positive observations to all actual positives."},
    {"Term": "F1 Score", "Acronym": None, "Category": "Evaluation Metrics", "Definition": "The harmonic mean of precision and recall, useful for imbalanced datasets."},
    {"Term": "Confusion Matrix", "Acronym": None, "Category": "Evaluation Metrics", "Definition": "A table used to evaluate the performance of a classification model by showing the number of true positives, true negatives, false positives, and false negatives."},
    {"Term": "ROC Curve", "Acronym": None, "Category": "Evaluation Metrics", "Definition": "A graphical representation of the true positive rate versus the false positive rate."},
    {"Term": "AUC", "Acronym": "Area Under Curve", "Category": "Evaluation Metrics", "Definition": "A metric that summarizes the ROC curve, representing the likelihood of the model distinguishing between classes."},

    {"Term": "TensorFlow", "Acronym": None, "Category": "Tools & Frameworks", "Definition": "An open-source framework for building and deploying ML models, developed by Google."},
    {"Term": "PyTorch", "Acronym": None, "Category": "Tools & Frameworks", "Definition": "An open-source ML library for Python, primarily developed by Facebook's AI Research lab."},
    {"Term": "Scikit-learn", "Acronym": None, "Category": "Tools & Frameworks", "Definition": "A Python module integrating a wide range of state-of-the-art ML algorithms for medium-scale supervised and unsupervised problems."},
    {"Term": "Keras", "Acronym": None, "Category": "Tools & Frameworks", "Definition": "An open-source software library that provides a Python interface for neural networks."},
    {"Term": "XGBoost", "Acronym": None, "Category": "Tools & Frameworks", "Definition": "An optimized gradient boosting library designed to be highly efficient, flexible, and portable."},
    {"Term": "LightGBM", "Acronym": None, "Category": "Tools & Frameworks", "Definition": "A gradient boosting framework that uses tree-based learning algorithms."},
    {"Term": "Hugging Face Transformers", "Acronym": None, "Category": "Tools & Frameworks", "Definition": "A library providing general-purpose architectures for NLP."},

    {"Term": "Data Scientist", "Acronym": None, "Category": "Job Roles", "Definition": "Professionals who analyze and interpret complex data to help organizations make informed decisions."},
    {"Term": "Machine Learning Engineer", "Acronym": None, "Category": "Job Roles", "Definition": "Engineers who design, build, and deploy ML models and systems."},
    {"Term": "Data Analyst", "Acronym": None, "Category": "Job Roles", "Definition": "Individuals who collect, process, and perform statistical analyses on data."},
    {"Term": "AI Researcher", "Acronym": None, "Category": "Job Roles", "Definition": "Researchers who focus on developing new algorithms and models in AI."},
    {"Term": "Data Engineer", "Acronym": None, "Category": "Job Roles", "Definition": "Engineers who design and construct systems and infrastructure for collecting, storing, and analyzing data."},
    {"Term": "AI Product Manager", "Acronym": None, "Category": "Job Roles", "Definition": "Professionals who oversee the development and deployment of AI-driven products."},
    {"Term": "ML Ops Engineer", "Acronym": None, "Category": "Job Roles", "Definition": "Engineers who manage the lifecycle of ML models, ensuring they are deployed and maintained effectively."},

    {"Term": "Natural Language Processing", "Acronym": "NLP", "Category": "Domains & Applications", "Definition": "A field of AI focused on the interaction between computers and human language."},
    {"Term": "Computer Vision", "Acronym": "CV", "Category": "Domains & Applications", "Definition": "An interdisciplinary field that enables computers to interpret and make decisions based on visual data."},
    {"Term": "Speech Recognition", "Acronym": None, "Category": "Domains & Applications", "Definition": "The process of converting spoken language into text."},
    {"Term": "Recommender Systems", "Acronym": None, "Category": "Domains & Applications", "Definition": "Systems that predict the rating or preference a user would give to an item."},
    {"Term": "Anomaly Detection", "Acronym": None, "Category": "Domains & Applications", "Definition": "The identification of rare items, events, or observations which differ significantly from the majority of the data."},
    {"Term": "Robotic Process Automation", "Acronym": "RPA", "Category": "Domains & Applications", "Definition": "The use of software robots to automate repetitive tasks."},
    {"Term": "Autonomous Vehicles", "Acronym": None, "Category": "Domains & Applications", "Definition": "Vehicles capable of sensing their environment and operating without human involvement."},

    {"Term": "Features", "Acronym": None, "Category": "Data & Preprocessing", "Definition": "Individual measurable properties or characteristics of a phenomenon being observed."},
    {"Term": "Labels", "Acronym": None, "Category": "Data & Preprocessing", "Definition": "The output variable that the model is trying to predict."},
    {"Term": "Normalization", "Acronym": None, "Category": "Data & Preprocessing", "Definition": "The process of scaling individual data points to have a specific statistical property, like a mean of zero and a standard deviation of one."},
    {"Term": "Standardization", "Acronym": None, "Category": "Data & Preprocessing", "Definition": "The process of rescaling the features so they have a mean of zero and a standard deviation of one."},
    {"Term": "One-Hot Encoding", "Acronym": None, "Category": "Data & Preprocessing", "Definition": "A method of representing categorical variables as binary vectors."},
    {"Term": "Feature Engineering", "Acronym": None, "Category": "Data & Preprocessing", "Definition": "The process of using domain knowledge to select, modify, or create new features from raw data."},
    {"Term": "Data Augmentation", "Acronym": None, "Category": "Data & Preprocessing", "Definition": "Techniques used to increase the diversity of data available for training models, without actually collecting new data."}
]

# Create a DataFrame
df_glossary = pd.DataFrame(glossary)

# Write to CSV
csv_path = "/mnt/data/ai_ml_glossary.csv"
df_glossary.to_csv(csv_path, index=False)

csv_path
