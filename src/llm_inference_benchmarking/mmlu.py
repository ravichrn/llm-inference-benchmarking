"""MMLU benchmark data and scoring helpers.

Extracted from modal_benchmark.py so that quality-evaluation logic can be
imported independently of the Modal infrastructure.
"""

from __future__ import annotations

from typing import Any

# 100-question MMLU subset for quality scoring — avoids runtime dataset download.
# Drawn from CS, ML, statistics, and systems domains.
# Format: (question, choices A-D, correct_letter)
_MMLU_SUBSET: list[tuple[str, list[str], str]] = [
    # --- Computer Science / Programming ---
    ("What is the output of `2 ** 10` in Python?", ["512", "1024", "2048", "256"], "B"),
    (
        "Which sorting algorithm has O(n log n) average-case complexity?",
        ["Bubble sort", "Insertion sort", "Merge sort", "Selection sort"],
        "C",
    ),
    (
        "In SQL, which clause filters rows after grouping?",
        ["WHERE", "HAVING", "GROUP BY", "ORDER BY"],
        "B",
    ),
    (
        "What does the CAP theorem state a distributed system cannot simultaneously guarantee?",
        [
            "Consistency, availability, and partition tolerance",
            "Concurrency, atomicity, and persistence",
            "Caching, archival, and persistence",
            "None of the above",
        ],
        "A",
    ),
    ("Which HTTP method is idempotent but not safe?", ["GET", "POST", "PUT", "DELETE"], "D"),
    (
        "What is the time complexity of binary search?",
        ["O(n)", "O(n log n)", "O(log n)", "O(1)"],
        "C",
    ),
    (
        "Which Python data structure provides O(1) average-case lookup by key?",
        ["List", "Tuple", "Dictionary", "Deque"],
        "C",
    ),
    (
        "What does 'attention is all you need' refer to?",
        ["A sleep study", "The Transformer architecture", "An RNN variant", "A CNN architecture"],
        "B",
    ),
    (
        "Which data format is most efficient for sparse matrices?",
        ["Dense array", "CSR (Compressed Sparse Row)", "JSON", "CSV"],
        "B",
    ),
    (
        "Which technique reduces model size by approximating weights with low-rank matrices?",
        ["Pruning", "Distillation", "LoRA / Low-rank adaptation", "Quantization"],
        "C",
    ),
    (
        "What does 'bits per weight' measure in quantized models?",
        ["Inference speed", "Precision of stored weights", "Perplexity", "GPU utilization"],
        "B",
    ),
    (
        "In a min-heap, which element is always at the root?",
        ["Maximum element", "Minimum element", "Median element", "Last inserted element"],
        "B",
    ),
    (
        "What is the space complexity of merge sort?",
        ["O(1)", "O(log n)", "O(n)", "O(n log n)"],
        "C",
    ),
    (
        "Which design pattern ensures only one instance of a class exists?",
        ["Factory", "Observer", "Singleton", "Strategy"],
        "C",
    ),
    (
        "What does ACID stand for in database transactions?",
        [
            "Atomicity, Consistency, Isolation, Durability",
            "Availability, Consistency, Integrity, Durability",
            "Atomicity, Concurrency, Isolation, Durability",
            "Availability, Concurrency, Integrity, Distribution",
        ],
        "A",
    ),
    (
        "Which Python keyword is used to define a generator function?",
        ["async", "yield", "return", "lambda"],
        "B",
    ),
    ("What is the output of `bool([])` in Python?", ["True", "False", "None", "Error"], "B"),
    (
        "Which algorithm finds the shortest path in an unweighted graph?",
        ["Dijkstra", "Bellman-Ford", "BFS", "DFS"],
        "C",
    ),
    (
        "What does REST stand for?",
        [
            "Representational State Transfer",
            "Remote Execution Service Transfer",
            "Resource Encoding Standard Transfer",
            "Relational Entity State Transfer",
        ],
        "A",
    ),
    (
        "In Git, what does `git rebase` do?",
        [
            "Merges branches creating a merge commit",
            "Reapplies commits on top of another branch",
            "Deletes a branch",
            "Reverts the last commit",
        ],
        "B",
    ),
    # --- Machine Learning / Deep Learning ---
    (
        "Which activation function outputs values strictly between 0 and 1?",
        ["ReLU", "Tanh", "Sigmoid", "Leaky ReLU"],
        "C",
    ),
    (
        "What does 'backpropagation' compute?",
        [
            "The forward pass output",
            "Gradients of the loss with respect to weights",
            "The softmax probabilities",
            "The attention scores",
        ],
        "B",
    ),
    (
        "In statistics, what does a p-value below 0.05 typically indicate?",
        [
            "The null hypothesis is true",
            "The result is statistically significant",
            "The effect size is large",
            "The sample is too small",
        ],
        "B",
    ),
    (
        "Which optimization algorithm adapts learning rates per parameter?",
        ["SGD", "Momentum", "Adam", "Perceptron"],
        "C",
    ),
    (
        "What is gradient clipping used for?",
        [
            "Speed up training",
            "Prevent exploding gradients",
            "Reduce overfitting",
            "Initialize weights",
        ],
        "B",
    ),
    (
        "Which normalization technique normalizes across the feature dimension per sample?",
        [
            "Batch normalization",
            "Layer normalization",
            "Group normalization",
            "Instance normalization",
        ],
        "B",
    ),
    (
        "What does KV cache store during autoregressive inference?",
        [
            "Model weights",
            "Key and value matrices from previous tokens",
            "Query vectors",
            "Attention masks",
        ],
        "B",
    ),
    (
        "What is perplexity a measure of?",
        [
            "Model speed",
            "How well a language model predicts a sample",
            "GPU memory usage",
            "Training loss",
        ],
        "B",
    ),
    (
        "In transformer models, what is the role of positional encoding?",
        [
            "Normalize inputs",
            "Inject sequence order information",
            "Compute attention weights",
            "Scale gradients",
        ],
        "B",
    ),
    (
        "Which loss function is standard for multi-class classification?",
        ["MSE", "MAE", "Cross-entropy", "Hinge loss"],
        "C",
    ),
    (
        "What is the vanishing gradient problem?",
        [
            "Gradients grow exponentially during backprop",
            "Gradients shrink toward zero making early layers train slowly",
            "The optimizer diverges",
            "Weights become negative",
        ],
        "B",
    ),
    (
        "Which technique randomly drops neurons during training to reduce overfitting?",
        ["Batch normalization", "Weight decay", "Dropout", "Early stopping"],
        "C",
    ),
    (
        "What does a confusion matrix diagonal represent?",
        ["False positives", "False negatives", "Correct predictions", "Total samples"],
        "C",
    ),
    (
        "Which metric is most suitable when class imbalance is severe?",
        ["Accuracy", "F1-score", "Mean Squared Error", "R-squared"],
        "B",
    ),
    (
        "What does the term 'epoch' mean in neural network training?",
        [
            "One forward pass",
            "One parameter update",
            "One full pass over the training dataset",
            "One batch of data",
        ],
        "C",
    ),
    (
        "Which regularization technique adds the L2 norm of weights to the loss?",
        ["L1 regularization", "Weight decay / L2 regularization", "Dropout", "Data augmentation"],
        "B",
    ),
    (
        "What is transfer learning?",
        [
            "Training a model from scratch on new data",
            "Using a pre-trained model and fine-tuning on a new task",
            "Copying weights between identical architectures",
            "Distilling a large model into a small one",
        ],
        "B",
    ),
    (
        "Which layer type is most commonly used for sequence modeling before transformers?",
        ["Convolutional", "LSTM / RNN", "Attention", "Dense"],
        "B",
    ),
    (
        "What does 'overfitting' mean?",
        [
            "Model performs poorly on training data",
            "Model performs well on training but poorly on unseen data",
            "Model takes too long to train",
            "Model uses too little memory",
        ],
        "B",
    ),
    (
        "Which ensemble method trains models sequentially, each correcting the previous?",
        ["Bagging", "Random Forest", "Boosting", "Stacking"],
        "C",
    ),
    # --- Systems / Networking ---
    ("Which OSI layer does TCP operate at?", ["Network", "Data Link", "Transport", "Session"], "C"),
    (
        "What does DNS stand for?",
        [
            "Dynamic Network Service",
            "Domain Name System",
            "Data Node Server",
            "Distributed Name Service",
        ],
        "B",
    ),
    (
        "What is the purpose of a load balancer?",
        [
            "Store data redundantly",
            "Distribute incoming traffic across multiple servers",
            "Encrypt network traffic",
            "Cache static assets",
        ],
        "B",
    ),
    (
        "Which consistency model guarantees all nodes see the same data at the same time?",
        ["Eventual consistency", "Strong consistency", "Causal consistency", "Read-your-writes"],
        "B",
    ),
    (
        "What does a CDN primarily optimize?",
        [
            "Database queries",
            "Compute-intensive ML inference",
            "Content delivery latency for geographically distributed users",
            "Container orchestration",
        ],
        "C",
    ),
    ("In HTTPS, which protocol handles encryption?", ["HTTP", "TLS", "TCP", "DNS"], "B"),
    (
        "What is the time complexity of accessing an element in a hash table (average case)?",
        ["O(log n)", "O(n)", "O(1)", "O(n log n)"],
        "C",
    ),
    (
        "Which scheduling algorithm minimizes average waiting time?",
        ["FIFO", "Shortest Job First (SJF)", "Round Robin", "Priority scheduling"],
        "B",
    ),
    (
        "What is virtual memory?",
        [
            "Extra physical RAM",
            "An abstraction that gives processes the illusion of a large contiguous address space",
            "Swap space only",
            "GPU memory accessible by the CPU",
        ],
        "B",
    ),
    (
        "What does a mutex prevent in concurrent programming?",
        [
            "Memory leaks",
            "Two threads accessing shared data simultaneously",
            "Deadlocks",
            "Stack overflows",
        ],
        "B",
    ),
    # --- Statistics & Math ---
    ("What is the median of {3, 1, 4, 1, 5, 9, 2, 6}?", ["3", "3.5", "4", "5"], "B"),
    (
        "Which distribution is parameterized by mean and variance?",
        ["Bernoulli", "Binomial", "Normal", "Poisson"],
        "C",
    ),
    (
        "What is the central limit theorem?",
        [
            "Sample means approach a normal distribution as sample size increases",
            "All distributions are normal",
            "Variance decreases with more data",
            "The mean always equals the median",
        ],
        "A",
    ),
    (
        "What does a high R-squared value indicate in regression?",
        [
            "Causation between variables",
            "Model explains a large proportion of variance in the target",
            "Low prediction error",
            "The model is not overfit",
        ],
        "B",
    ),
    (
        "What is Bayes' theorem used for?",
        [
            "Computing determinants",
            "Updating probability estimates given new evidence",
            "Finding eigenvalues",
            "Gradient computation",
        ],
        "B",
    ),
    (
        "Which measure of central tendency is most resistant to outliers?",
        ["Mean", "Variance", "Median", "Standard deviation"],
        "C",
    ),
    (
        "What is a Type I error?",
        [
            "Failing to reject a false null hypothesis",
            "Rejecting a true null hypothesis",
            "A sample that is too small",
            "A biased estimator",
        ],
        "B",
    ),
    (
        "What does standard deviation measure?",
        [
            "The center of a distribution",
            "The spread of data around the mean",
            "The skewness of a distribution",
            "The maximum value",
        ],
        "B",
    ),
    (
        "Which sampling method ensures every population member has an equal chance of selection?",
        [
            "Convenience sampling",
            "Cluster sampling",
            "Simple random sampling",
            "Stratified sampling",
        ],
        "C",
    ),
    (
        "In hypothesis testing, what is the null hypothesis?",
        [
            "The hypothesis we aim to prove",
            "The assumption of no effect or no difference",
            "The alternative to the research hypothesis",
            "The observed outcome",
        ],
        "B",
    ),
    # --- LLM / NLP ---
    (
        "What does 'tokenization' mean in NLP?",
        [
            "Encrypting text",
            "Splitting text into subword or word units for model input",
            "Translating text",
            "Compressing text",
        ],
        "B",
    ),
    (
        "What is the purpose of the softmax function in the output layer of a language model?",
        [
            "Normalize logits into a probability distribution over the vocabulary",
            "Clip extreme values",
            "Apply dropout",
            "Compute cross-entropy loss",
        ],
        "A",
    ),
    (
        "What is RLHF?",
        [
            "A quantization technique",
            "Reinforcement Learning from Human Feedback used to align LLMs",
            "A retrieval method",
            "A tokenization strategy",
        ],
        "B",
    ),
    (
        "What does 'temperature' control in LLM sampling?",
        [
            "GPU temperature",
            "The sharpness of the token probability distribution",
            "Context window size",
            "Model precision",
        ],
        "B",
    ),
    (
        "What is 'hallucination' in LLMs?",
        [
            "Generating tokens too slowly",
            "Producing fluent but factually incorrect content",
            "Running out of context",
            "Failing to follow system prompts",
        ],
        "B",
    ),
    (
        "What is the purpose of a system prompt in an LLM API call?",
        [
            "Set the model's persona and constraints for the conversation",
            "Specify the GPU to use",
            "Define the tokenizer vocabulary",
            "Set the learning rate",
        ],
        "A",
    ),
    (
        "What does 'context window' refer to?",
        [
            "The GPU's L2 cache",
            "The maximum number of tokens a model can process in one call",
            "The training dataset size",
            "The number of attention heads",
        ],
        "B",
    ),
    (
        "What is retrieval-augmented generation (RAG)?",
        [
            "Fine-tuning a model on a private corpus",
            "Augmenting LLM responses with documents fetched from a retrieval system",
            "Caching model outputs",
            "Quantizing a model with retrieved calibration data",
        ],
        "B",
    ),
    (
        "Which attention variant reduces memory from O(n²) to O(n) by chunking?",
        ["Multi-head attention", "Flash Attention", "Cross-attention", "Sparse attention"],
        "B",
    ),
    (
        "What does 'greedy decoding' mean?",
        [
            "Sampling from the full distribution",
            "Always selecting the highest-probability next token",
            "Using beam search with k=1",
            "Sampling with temperature=0.7",
        ],
        "B",
    ),
    # --- Cloud & MLOps ---
    (
        "What is containerization?",
        [
            "Compressing model weights",
            "Packaging an application with its dependencies into an isolated unit",
            "A networking protocol",
            "A database sharding strategy",
        ],
        "B",
    ),
    (
        "What does Kubernetes orchestrate?",
        [
            "SQL queries",
            "Container deployments across a cluster",
            "Model training runs",
            "DNS resolution",
        ],
        "B",
    ),
    (
        "What is the purpose of a CI/CD pipeline?",
        [
            "Monitor production metrics",
            "Automate building, testing, and deploying software",
            "Manage cloud billing",
            "Schedule GPU jobs",
        ],
        "B",
    ),
    (
        "Which cloud storage type is best for unstructured binary data (e.g., model weights)?",
        [
            "Relational database",
            "Object storage (e.g., S3, GCS)",
            "Key-value cache",
            "Block storage",
        ],
        "B",
    ),
    (
        "What does 'infrastructure as code' (IaC) mean?",
        [
            "Writing ML models in C++",
            "Defining and provisioning infrastructure through code files",
            "Compiling Python to machine code",
            "Storing secrets in environment variables",
        ],
        "B",
    ),
    (
        "What is model serving?",
        [
            "Training a model on new data",
            "Exposing a trained model to clients via an API",
            "Storing model checkpoints",
            "Evaluating model quality offline",
        ],
        "B",
    ),
    (
        "What is a feature store in ML pipelines?",
        [
            "A GPU memory pool",
            "A centralized repository for storing and serving ML features",
            "A hyperparameter search service",
            "A model registry",
        ],
        "B",
    ),
    (
        "What does 'A/B testing' evaluate?",
        [
            "Two different hardware configurations",
            "Whether a new model or feature improves a metric vs. a control",
            "Two database schemas",
            "Checkpoint quality",
        ],
        "B",
    ),
    (
        "What is the purpose of model quantization?",
        [
            "Improve model accuracy",
            "Reduce model size and memory footprint at the cost of some precision",
            "Increase training speed",
            "Add more parameters",
        ],
        "B",
    ),
    (
        "What is data drift?",
        [
            "Network latency increase",
            "A change in the statistical properties of model input data over time",
            "GPU memory fragmentation",
            "Gradient instability",
        ],
        "B",
    ),
    # --- Algorithms & Data Structures ---
    (
        "What is dynamic programming?",
        [
            "A runtime programming paradigm",
            "Breaking problems into overlapping subproblems and caching results",
            "A parallel computing technique",
            "A graph traversal method",
        ],
        "B",
    ),
    (
        "Which graph algorithm detects cycles in a directed graph?",
        ["BFS", "Dijkstra", "DFS with recursion stack tracking", "Prim's"],
        "C",
    ),
    (
        "What is the worst-case time complexity of quicksort?",
        ["O(n log n)", "O(n²)", "O(n)", "O(log n)"],
        "B",
    ),
    (
        "Which data structure supports O(1) push and pop from one end?",
        ["Queue", "Stack", "Linked list", "Tree"],
        "B",
    ),
    (
        "What is a balanced BST?",
        [
            "A tree where all nodes have two children",
            "A BST where heights of left and right subtrees differ by at most 1",
            "A tree with no duplicate keys",
            "A tree sorted by insertion order",
        ],
        "B",
    ),
    (
        "What does amortized O(1) mean for dynamic array append?",
        [
            "Every append is O(1)",
            "The average cost per append is O(1) over a sequence of operations",
            "The array never resizes",
            "The worst case is O(1)",
        ],
        "B",
    ),
    (
        "Which traversal visits nodes in sorted order for a BST?",
        ["Preorder", "Postorder", "Inorder", "Level-order"],
        "C",
    ),
    (
        "What is memoization?",
        [
            "A cache for function results based on inputs to avoid recomputation",
            "A technique for parallelizing loops",
            "A form of data compression",
            "Saving model checkpoints",
        ],
        "A",
    ),
    (
        "What is the purpose of a bloom filter?",
        [
            "Sort elements efficiently",
            "Test whether an element is possibly in a set with no false negatives",
            "Compress data",
            "Implement a hash map",
        ],
        "B",
    ),
    (
        "Which problem does the A* algorithm solve?",
        [
            "Minimum spanning tree",
            "Shortest path with a heuristic",
            "Maximum flow",
            "Topological sort",
        ],
        "B",
    ),
    # --- Additional CS / ML (to reach 100) ---
    (
        "What is the purpose of an embedding layer in neural networks?",
        [
            "Reduce overfitting",
            "Map discrete tokens to dense continuous vectors",
            "Normalize activations",
            "Compute attention weights",
        ],
        "B",
    ),
    (
        "Which Python library is the standard for numerical array computation?",
        ["pandas", "scikit-learn", "NumPy", "SciPy"],
        "C",
    ),
    (
        "What does 'precision' measure in a classification model?",
        [
            "Fraction of actual positives correctly identified",
            "Fraction of predicted positives that are truly positive",
            "Overall accuracy",
            "Recall divided by F1",
        ],
        "B",
    ),
    (
        "What is the primary advantage of using a GPU over a CPU for deep learning?",
        [
            "Higher clock speed",
            "Larger RAM capacity",
            "Massively parallel execution of matrix operations",
            "Lower power consumption",
        ],
        "C",
    ),
    (
        "In the context of LLMs, what is 'fine-tuning'?",
        [
            "Compressing model weights",
            "Further training a pre-trained model on a task-specific dataset",
            "Pruning unused attention heads",
            "Converting model to ONNX format",
        ],
        "B",
    ),
    (
        "What does 'beam search' do during text generation?",
        [
            "Selects tokens greedily",
            "Maintains k candidate sequences and selects the highest-scoring one",
            "Samples from the full distribution",
            "Uses a draft model to propose tokens",
        ],
        "B",
    ),
    (
        "Which metric measures the harmonic mean of precision and recall?",
        ["Accuracy", "AUC-ROC", "F1-score", "MCC"],
        "C",
    ),
    (
        "What is weight initialization used for in neural networks?",
        [
            "Reduce model size",
            "Set initial parameter values to enable stable gradient flow",
            "Define the learning rate schedule",
            "Control dropout probability",
        ],
        "B",
    ),
    (
        "Which transformer component allows the model to attend to different positions?",
        [
            "Feed-forward layer",
            "Layer normalization",
            "Multi-head self-attention",
            "Residual connection",
        ],
        "C",
    ),
    (
        "What does 'zero-shot' mean when evaluating a language model?",
        [
            "The model is evaluated with zero training examples for the task",
            "The model generates zero tokens",
            "Temperature is set to zero",
            "No system prompt is used",
        ],
        "A",
    ),
]

# Subject ranges for the 100-question MMLU subset (0-indexed, inclusive).
# Kept for reference; runtime scoring now uses _load_mmlu_questions() instead.
_MMLU_SUBJECT_RANGES: list[tuple[int, int, str]] = [
    (0, 13, "cs_programming"),
    (14, 33, "ml_deep_learning"),
    (34, 42, "systems_networking"),
    (43, 52, "statistics_math"),
    (53, 64, "llm_nlp"),
    (65, 77, "cloud_mlops"),
    (78, 87, "algorithms_ds"),
    (88, 99, "additional_cs_ml"),
]


def _load_mmlu_questions(n: int = 50) -> list[dict]:
    """Convert hardcoded _MMLU_SUBSET to dict format for scoring functions."""
    out = []
    for i, (question, choices, answer_letter) in enumerate(_MMLU_SUBSET[:n]):
        subject = "other"
        for start, end, subj in _MMLU_SUBJECT_RANGES:
            if start <= i <= end:
                subject = subj
                break
        out.append(
            {
                "question": question,
                "choices": choices,
                "answer_idx": ord(answer_letter) - ord("A"),
                "subject": subject,
            }
        )
    return out


def _measure_quality_vllm(llm: Any) -> dict[str, Any]:
    """MMLU log-prob scoring via vLLM's generate with logprobs (200 questions)."""
    from vllm import SamplingParams

    correct = 0
    details: list[dict] = []
    questions = _load_mmlu_questions()
    subject_stats: dict[str, dict[str, int]] = {}

    # Request logprobs for the first generated token — used to score each choice
    # logprobs=20 ensures A/B/C/D tokens are included even when not in model's top-5
    score_params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)

    for entry in questions:
        question, choices, answer_idx = entry["question"], entry["choices"], entry["answer_idx"]
        subject = entry["subject"]
        answer_letter = chr(ord("A") + answer_idx)
        choice_labels = [chr(ord("A") + i) for i in range(len(choices))]
        prompt = f"Question: {question}\nA) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}\nAnswer:"
        out = llm.generate([prompt], score_params, use_tqdm=False)[0]
        predicted_label = "A"
        if out.outputs and out.outputs[0].logprobs:
            first_token_logprobs = out.outputs[0].logprobs[0]
            best_lp = -1e9
            for v in first_token_logprobs.values():
                tok = v.decoded_token.strip()
                if tok in choice_labels and v.logprob > best_lp:
                    best_lp = v.logprob
                    predicted_label = tok

        is_correct = predicted_label == answer_letter
        if is_correct:
            correct += 1
        s = subject_stats.setdefault(subject, {"correct": 0, "total": 0})
        s["total"] += 1
        if is_correct:
            s["correct"] += 1
        details.append(
            {
                "question": question[:60] + "…" if len(question) > 60 else question,
                "predicted": predicted_label,
                "expected": answer_letter,
                "correct": is_correct,
                "subject": subject,
            }
        )

    accuracy = correct / len(questions)
    by_subject = {
        subj: {"accuracy": round(v["correct"] / v["total"], 4), "correct": v["correct"], "total": v["total"]}
        for subj, v in subject_stats.items()
    }
    return {
        "mmlu_accuracy": round(accuracy, 4),
        "correct": correct,
        "total": len(questions),
        "mmlu_by_subject": by_subject,
        "details": details,
    }
