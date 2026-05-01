"""Modal GPU quantization benchmark for Llama-3.1-8B-Instruct (ungated mirror).

Measures all key metrics across quantization modes (fp16, int8, nf4, nf4-dq, awq, spec-dec):
  - Latency: mean, p50, p95, p99, time-to-first-token (ms)
  - Throughput: output tokens/sec, total tokens/sec
  - Memory: peak GPU VRAM (MB)
  - Perplexity: on WikiText-2 test set (128-token stride)
  - Quality: zero-shot accuracy on a 100-question MMLU subset

Prerequisites:
  modal setup          # authenticate with your Modal account (one-time)

Optional .env overrides (Modal reads .env automatically via Secret.from_dotenv):
  HUGGING_FACE_HUB_TOKEN=<token>   # only needed if switching to a gated model
  QUANT_AWQ_MODEL=<hf_model_id>    # override default AWQ checkpoint

Run:
  modal run src/llm_inference_benchmarking/modal_benchmark.py
  modal run src/llm_inference_benchmarking/modal_benchmark.py --modes fp16,nf4,awq
  modal run src/llm_inference_benchmarking/modal_benchmark.py --output results/bench.json
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import modal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Ungated community mirror — identical weights to meta-llama/Llama-3.1-8B-Instruct,
# no HuggingFace token required.
BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct"

# Pre-quantized AWQ checkpoint (ungated). Override via QUANT_AWQ_MODEL in .env.
_DEFAULT_AWQ_MODEL = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"

# Draft model for speculative decoding — same tokenizer vocab, ~2 GB VRAM.
DRAFT_MODEL = "unsloth/Llama-3.2-1B-Instruct"

# Model revision (git commit SHA) for reproducibility.
# Override via MODEL_REVISION env var; defaults to "main" (latest).
# To pin: set to the exact HF commit sha, e.g. "abc1234".
_MODEL_REVISION = os.getenv("MODEL_REVISION", "main")

_ALL_MODES = ("fp16", "int8", "nf4", "nf4-dq", "awq", "spec-dec", "vllm")

# Prompts for latency / throughput measurement
_BENCH_PROMPTS = [
    "Summarize why retrieval-augmented generation reduces hallucination in large language models.",
    "Compare diffusion models versus autoregressive models for image generation. List pros and cons.",
    "Rewrite this search query for better semantic retrieval: papers about robust RL transfer learning.",
    "Explain the transformer attention mechanism to a software engineer with no ML background.",
    "What are the key trade-offs between model quantization and full-precision inference?",
]

# 100-question MMLU subset for quality scoring — avoids runtime dataset download.
# Drawn from CS, ML, statistics, and systems domains.
# Format: (question, choices A-D, correct_letter)
_MMLU_SUBSET: list[tuple[str, list[str], str]] = [
    # --- Computer Science / Programming ---
    ("What is the output of `2 ** 10` in Python?", ["512", "1024", "2048", "256"], "B"),
    ("Which sorting algorithm has O(n log n) average-case complexity?", ["Bubble sort", "Insertion sort", "Merge sort", "Selection sort"], "C"),
    ("In SQL, which clause filters rows after grouping?", ["WHERE", "HAVING", "GROUP BY", "ORDER BY"], "B"),
    ("What does the CAP theorem state a distributed system cannot simultaneously guarantee?", ["Consistency, availability, and partition tolerance", "Concurrency, atomicity, and persistence", "Caching, archival, and persistence", "None of the above"], "A"),
    ("Which HTTP method is idempotent but not safe?", ["GET", "POST", "PUT", "DELETE"], "D"),
    ("What is the time complexity of binary search?", ["O(n)", "O(n log n)", "O(log n)", "O(1)"], "C"),
    ("Which Python data structure provides O(1) average-case lookup by key?", ["List", "Tuple", "Dictionary", "Deque"], "C"),
    ("What does 'attention is all you need' refer to?", ["A sleep study", "The Transformer architecture", "An RNN variant", "A CNN architecture"], "B"),
    ("Which data format is most efficient for sparse matrices?", ["Dense array", "CSR (Compressed Sparse Row)", "JSON", "CSV"], "B"),
    ("Which technique reduces a model's size by approximating weight matrices with lower-rank matrices?", ["Pruning", "Distillation", "LoRA / Low-rank adaptation", "Quantization"], "C"),
    ("What does 'bits per weight' measure in quantized models?", ["Inference speed", "Precision of stored weights", "Perplexity", "GPU utilization"], "B"),
    ("In a min-heap, which element is always at the root?", ["Maximum element", "Minimum element", "Median element", "Last inserted element"], "B"),
    ("What is the space complexity of merge sort?", ["O(1)", "O(log n)", "O(n)", "O(n log n)"], "C"),
    ("Which design pattern ensures only one instance of a class exists?", ["Factory", "Observer", "Singleton", "Strategy"], "C"),
    ("What does ACID stand for in database transactions?", ["Atomicity, Consistency, Isolation, Durability", "Availability, Consistency, Integrity, Durability", "Atomicity, Concurrency, Isolation, Durability", "Availability, Concurrency, Integrity, Distribution"], "A"),
    ("Which Python keyword is used to define a generator function?", ["async", "yield", "return", "lambda"], "B"),
    ("What is the output of `bool([])` in Python?", ["True", "False", "None", "Error"], "B"),
    ("Which algorithm finds the shortest path in an unweighted graph?", ["Dijkstra", "Bellman-Ford", "BFS", "DFS"], "C"),
    ("What does REST stand for?", ["Representational State Transfer", "Remote Execution Service Transfer", "Resource Encoding Standard Transfer", "Relational Entity State Transfer"], "A"),
    ("In Git, what does `git rebase` do?", ["Merges branches creating a merge commit", "Reapplies commits on top of another branch", "Deletes a branch", "Reverts the last commit"], "B"),
    # --- Machine Learning / Deep Learning ---
    ("Which activation function outputs values strictly between 0 and 1?", ["ReLU", "Tanh", "Sigmoid", "Leaky ReLU"], "C"),
    ("What does 'backpropagation' compute?", ["The forward pass output", "Gradients of the loss with respect to weights", "The softmax probabilities", "The attention scores"], "B"),
    ("In statistics, what does a p-value below 0.05 typically indicate?", ["The null hypothesis is true", "The result is statistically significant", "The effect size is large", "The sample is too small"], "B"),
    ("Which optimization algorithm adapts learning rates per parameter?", ["SGD", "Momentum", "Adam", "Perceptron"], "C"),
    ("What is gradient clipping used for?", ["Speed up training", "Prevent exploding gradients", "Reduce overfitting", "Initialize weights"], "B"),
    ("Which normalization technique normalizes across the feature dimension per sample?", ["Batch normalization", "Layer normalization", "Group normalization", "Instance normalization"], "B"),
    ("What does KV cache store during autoregressive inference?", ["Model weights", "Key and value matrices from previous tokens", "Query vectors", "Attention masks"], "B"),
    ("What is perplexity a measure of?", ["Model speed", "How well a language model predicts a sample", "GPU memory usage", "Training loss"], "B"),
    ("In transformer models, what is the role of positional encoding?", ["Normalize inputs", "Inject sequence order information", "Compute attention weights", "Scale gradients"], "B"),
    ("Which loss function is standard for multi-class classification?", ["MSE", "MAE", "Cross-entropy", "Hinge loss"], "C"),
    ("What is the vanishing gradient problem?", ["Gradients grow exponentially during backprop", "Gradients shrink toward zero making early layers train slowly", "The optimizer diverges", "Weights become negative"], "B"),
    ("Which technique randomly drops neurons during training to reduce overfitting?", ["Batch normalization", "Weight decay", "Dropout", "Early stopping"], "C"),
    ("What does a confusion matrix diagonal represent?", ["False positives", "False negatives", "Correct predictions", "Total samples"], "C"),
    ("Which metric is most suitable when class imbalance is severe?", ["Accuracy", "F1-score", "Mean Squared Error", "R-squared"], "B"),
    ("What does the term 'epoch' mean in neural network training?", ["One forward pass", "One parameter update", "One full pass over the training dataset", "One batch of data"], "C"),
    ("Which regularization technique adds the L2 norm of weights to the loss?", ["L1 regularization", "Weight decay / L2 regularization", "Dropout", "Data augmentation"], "B"),
    ("What is transfer learning?", ["Training a model from scratch on new data", "Using a pre-trained model and fine-tuning on a new task", "Copying weights between identical architectures", "Distilling a large model into a small one"], "B"),
    ("Which layer type is most commonly used for sequence modeling before transformers?", ["Convolutional", "LSTM / RNN", "Attention", "Dense"], "B"),
    ("What does 'overfitting' mean?", ["Model performs poorly on training data", "Model performs well on training but poorly on unseen data", "Model takes too long to train", "Model uses too little memory"], "B"),
    ("Which ensemble method trains models sequentially, each correcting the previous?", ["Bagging", "Random Forest", "Boosting", "Stacking"], "C"),
    # --- Systems / Networking ---
    ("Which OSI layer does TCP operate at?", ["Network", "Data Link", "Transport", "Session"], "C"),
    ("What does DNS stand for?", ["Dynamic Network Service", "Domain Name System", "Data Node Server", "Distributed Name Service"], "B"),
    ("What is the purpose of a load balancer?", ["Store data redundantly", "Distribute incoming traffic across multiple servers", "Encrypt network traffic", "Cache static assets"], "B"),
    ("Which consistency model guarantees all nodes see the same data at the same time?", ["Eventual consistency", "Strong consistency", "Causal consistency", "Read-your-writes"], "B"),
    ("What does a CDN primarily optimize?", ["Database queries", "Compute-intensive ML inference", "Content delivery latency for geographically distributed users", "Container orchestration"], "C"),
    ("In HTTPS, which protocol handles encryption?", ["HTTP", "TLS", "TCP", "DNS"], "B"),
    ("What is the time complexity of accessing an element in a hash table (average case)?", ["O(log n)", "O(n)", "O(1)", "O(n log n)"], "C"),
    ("Which scheduling algorithm minimizes average waiting time?", ["FIFO", "Shortest Job First (SJF)", "Round Robin", "Priority scheduling"], "B"),
    ("What is virtual memory?", ["Extra physical RAM", "An abstraction that gives processes the illusion of a large contiguous address space", "Swap space only", "GPU memory accessible by the CPU"], "B"),
    ("What does a mutex prevent in concurrent programming?", ["Memory leaks", "Two threads accessing shared data simultaneously", "Deadlocks", "Stack overflows"], "B"),
    # --- Statistics & Math ---
    ("What is the median of {3, 1, 4, 1, 5, 9, 2, 6}?", ["3", "3.5", "4", "5"], "B"),
    ("Which distribution is parameterized by mean and variance?", ["Bernoulli", "Binomial", "Normal", "Poisson"], "C"),
    ("What is the central limit theorem?", ["Sample means approach a normal distribution as sample size increases", "All distributions are normal", "Variance decreases with more data", "The mean always equals the median"], "A"),
    ("What does a high R-squared value indicate in regression?", ["Causation between variables", "Model explains a large proportion of variance in the target", "Low prediction error", "The model is not overfit"], "B"),
    ("What is Bayes' theorem used for?", ["Computing determinants", "Updating probability estimates given new evidence", "Finding eigenvalues", "Gradient computation"], "B"),
    ("Which measure of central tendency is most resistant to outliers?", ["Mean", "Variance", "Median", "Standard deviation"], "C"),
    ("What is a Type I error?", ["Failing to reject a false null hypothesis", "Rejecting a true null hypothesis", "A sample that is too small", "A biased estimator"], "B"),
    ("What does standard deviation measure?", ["The center of a distribution", "The spread of data around the mean", "The skewness of a distribution", "The maximum value"], "B"),
    ("Which sampling method ensures every population member has an equal chance of selection?", ["Convenience sampling", "Cluster sampling", "Simple random sampling", "Stratified sampling"], "C"),
    ("In hypothesis testing, what is the null hypothesis?", ["The hypothesis we aim to prove", "The assumption of no effect or no difference", "The alternative to the research hypothesis", "The observed outcome"], "B"),
    # --- LLM / NLP ---
    ("What does 'tokenization' mean in NLP?", ["Encrypting text", "Splitting text into subword or word units for model input", "Translating text", "Compressing text"], "B"),
    ("What is the purpose of the softmax function in the output layer of a language model?", ["Normalize logits into a probability distribution over the vocabulary", "Clip extreme values", "Apply dropout", "Compute cross-entropy loss"], "A"),
    ("What is RLHF?", ["A quantization technique", "Reinforcement Learning from Human Feedback used to align LLMs", "A retrieval method", "A tokenization strategy"], "B"),
    ("What does 'temperature' control in LLM sampling?", ["GPU temperature", "The sharpness of the token probability distribution", "Context window size", "Model precision"], "B"),
    ("What is 'hallucination' in LLMs?", ["Generating tokens too slowly", "Producing fluent but factually incorrect content", "Running out of context", "Failing to follow system prompts"], "B"),
    ("What is the purpose of a system prompt in an LLM API call?", ["Set the model's persona and constraints for the conversation", "Specify the GPU to use", "Define the tokenizer vocabulary", "Set the learning rate"], "A"),
    ("What does 'context window' refer to?", ["The GPU's L2 cache", "The maximum number of tokens a model can process in one call", "The training dataset size", "The number of attention heads"], "B"),
    ("What is retrieval-augmented generation (RAG)?", ["Fine-tuning a model on a private corpus", "Augmenting LLM responses with documents fetched from a retrieval system", "Caching model outputs", "Quantizing a model with retrieved calibration data"], "B"),
    ("Which attention variant reduces memory from O(n²) to O(n) by chunking?", ["Multi-head attention", "Flash Attention", "Cross-attention", "Sparse attention"], "B"),
    ("What does 'greedy decoding' mean?", ["Sampling from the full distribution", "Always selecting the highest-probability next token", "Using beam search with k=1", "Sampling with temperature=0.7"], "B"),
    # --- Cloud & MLOps ---
    ("What is containerization?", ["Compressing model weights", "Packaging an application with its dependencies into an isolated unit", "A networking protocol", "A database sharding strategy"], "B"),
    ("What does Kubernetes orchestrate?", ["SQL queries", "Container deployments across a cluster", "Model training runs", "DNS resolution"], "B"),
    ("What is the purpose of a CI/CD pipeline?", ["Monitor production metrics", "Automate building, testing, and deploying software", "Manage cloud billing", "Schedule GPU jobs"], "B"),
    ("Which cloud storage type is best for unstructured binary data (e.g., model weights)?", ["Relational database", "Object storage (e.g., S3, GCS)", "Key-value cache", "Block storage"], "B"),
    ("What does 'infrastructure as code' (IaC) mean?", ["Writing ML models in C++", "Defining and provisioning infrastructure through code files", "Compiling Python to machine code", "Storing secrets in environment variables"], "B"),
    ("What is model serving?", ["Training a model on new data", "Exposing a trained model to clients via an API", "Storing model checkpoints", "Evaluating model quality offline"], "B"),
    ("What is a feature store in ML pipelines?", ["A GPU memory pool", "A centralized repository for storing and serving ML features", "A hyperparameter search service", "A model registry"], "B"),
    ("What does 'A/B testing' evaluate?", ["Two different hardware configurations", "Whether a new model or feature improves a metric vs. a control", "Two database schemas", "Checkpoint quality"], "B"),
    ("What is the purpose of model quantization?", ["Improve model accuracy", "Reduce model size and memory footprint at the cost of some precision", "Increase training speed", "Add more parameters"], "B"),
    ("What is data drift?", ["Network latency increase", "A change in the statistical properties of model input data over time", "GPU memory fragmentation", "Gradient instability"], "B"),
    # --- Algorithms & Data Structures ---
    ("What is dynamic programming?", ["A runtime programming paradigm", "Breaking problems into overlapping subproblems and caching results", "A parallel computing technique", "A graph traversal method"], "B"),
    ("Which graph algorithm detects cycles in a directed graph?", ["BFS", "Dijkstra", "DFS with recursion stack tracking", "Prim's"], "C"),
    ("What is the worst-case time complexity of quicksort?", ["O(n log n)", "O(n²)", "O(n)", "O(log n)"], "B"),
    ("Which data structure supports O(1) push and pop from one end?", ["Queue", "Stack", "Linked list", "Tree"], "B"),
    ("What is a balanced BST?", ["A tree where all nodes have two children", "A BST where heights of left and right subtrees differ by at most 1", "A tree with no duplicate keys", "A tree sorted by insertion order"], "B"),
    ("What does amortized O(1) mean for dynamic array append?", ["Every append is O(1)", "The average cost per append is O(1) over a sequence of operations", "The array never resizes", "The worst case is O(1)"], "B"),
    ("Which traversal visits nodes in sorted order for a BST?", ["Preorder", "Postorder", "Inorder", "Level-order"], "C"),
    ("What is memoization?", ["A cache for function results based on inputs to avoid recomputation", "A technique for parallelizing loops", "A form of data compression", "Saving model checkpoints"], "A"),
    ("What is the purpose of a bloom filter?", ["Sort elements efficiently", "Test whether an element is possibly in a set with no false negatives", "Compress data", "Implement a hash map"], "B"),
    ("Which problem does the A* algorithm solve?", ["Minimum spanning tree", "Shortest path with a heuristic", "Maximum flow", "Topological sort"], "B"),
    # --- Additional CS / ML (to reach 100) ---
    ("What is the purpose of an embedding layer in neural networks?", ["Reduce overfitting", "Map discrete tokens to dense continuous vectors", "Normalize activations", "Compute attention weights"], "B"),
    ("Which Python library is the standard for numerical array computation?", ["pandas", "scikit-learn", "NumPy", "SciPy"], "C"),
    ("What does 'precision' measure in a classification model?", ["Fraction of actual positives correctly identified", "Fraction of predicted positives that are truly positive", "Overall accuracy", "Recall divided by F1"], "B"),
    ("What is the primary advantage of using a GPU over a CPU for deep learning?", ["Higher clock speed", "Larger RAM capacity", "Massively parallel execution of matrix operations", "Lower power consumption"], "C"),
    ("In the context of LLMs, what is 'fine-tuning'?", ["Compressing model weights", "Further training a pre-trained model on a task-specific dataset", "Pruning unused attention heads", "Converting model to ONNX format"], "B"),
    ("What does 'beam search' do during text generation?", ["Selects tokens greedily", "Maintains k candidate sequences and selects the highest-scoring one", "Samples from the full distribution", "Uses a draft model to propose tokens"], "B"),
    ("Which metric measures the harmonic mean of precision and recall?", ["Accuracy", "AUC-ROC", "F1-score", "MCC"], "C"),
    ("What is weight initialization used for in neural networks?", ["Reduce model size", "Set initial parameter values to enable stable gradient flow", "Define the learning rate schedule", "Control dropout probability"], "B"),
    ("Which transformer component allows the model to attend to different positions?", ["Feed-forward layer", "Layer normalization", "Multi-head self-attention", "Residual connection"], "C"),
    ("What does 'zero-shot' mean when evaluating a language model?", ["The model is evaluated with zero training examples for the task", "The model generates zero tokens", "Temperature is set to zero", "No system prompt is used"], "A"),
]

# Per-mode notes surfaced in output JSON — explains known caveats so results are self-interpreting
_MODE_NOTES: dict[str, str] = {
    "int8": (
        "bitsandbytes int8 is compute-bound on A10G: dequantize-then-multiply adds "
        "overhead vs fp16's native tensor cores. int8 saves VRAM but reduces throughput. "
        "Use fp16 if VRAM allows; use nf4 for the best speed/memory trade-off."
    ),
    "awq": (
        "AWQ loaded via HuggingFace path with fuse_layers=False (autoawq 0.2.7+ breaks "
        "the transformers bridge). This disables fused INT4 matmul kernels, collapsing "
        "throughput. In production (vLLM / TGI with fused kernels), AWQ typically matches "
        "or exceeds fp16 throughput at 4-bit precision."
    ),
    "spec-dec": (
        "Speculative decoding uses a 1B draft model to propose tokens that the 8B target "
        "verifies in one parallel pass. Output is mathematically identical to target-only "
        "greedy decoding. TTFT is measured without the draft model (prefill-only baseline) "
        "so it is directly comparable to other modes. Throughput gain depends on draft "
        "acceptance rate — higher on predictable/repetitive text, lower on diverse prompts."
    ),
}

# ---------------------------------------------------------------------------
# Modal image
# ---------------------------------------------------------------------------

_image = (
    # CUDA 12.4 devel image: provides libnvJitLink.so.13 (needed by bitsandbytes>=0.43) and build tools for autoawq
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install("torch==2.5.1", "numpy<2.0")
    # autoawq inspects torch at build time — must come after torch is installed
    .pip_install("autoawq>=0.2.6")
    .pip_install(
        "transformers>=4.44.0",
        "accelerate>=0.33.0",
        "bitsandbytes>=0.43.0",
        "datasets>=2.20.0",
        "huggingface_hub",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .pip_install("hf-transfer")
    # vLLM for the "vllm" benchmark mode — installed last to avoid CUDA conflicts
    .pip_install("vllm>=0.6.0")
)

# Persistent volume for caching downloaded model weights
_model_cache = modal.Volume.from_name("llm-quant-model-cache", create_if_missing=True)
_secret_payload = {
    k: v
    for k in ("HUGGING_FACE_HUB_TOKEN", "QUANT_AWQ_MODEL")
    if (v := os.getenv(k, "").strip())
}
_modal_secrets = (
    [modal.Secret.from_dict(_secret_payload, name="llm-quant-benchmark-secrets")]
    if _secret_payload
    else []
)

app = modal.App("llm-quant-benchmark")

# ---------------------------------------------------------------------------
# Remote benchmark function
# ---------------------------------------------------------------------------


@app.function(
    gpu="A10G",
    image=_image,
    timeout=7200,
    volumes={"/model-cache": _model_cache},
    secrets=_modal_secrets,
    memory=32768,
)
def run_quant_benchmark(quant_mode: str) -> dict[str, Any]:
    """Run all metrics for one quantization mode. Executed remotely on Modal GPU."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    os.environ["TRANSFORMERS_CACHE"] = "/model-cache/hf"
    os.environ["HF_HOME"] = "/model-cache/hf"
    # Optional — only needed if switching to a gated model via .env
    hf_token: str | None = os.environ.get("HUGGING_FACE_HUB_TOKEN") or None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    if quant_mode == "vllm":
        return _run_vllm_benchmark(gpu_name, hf_token)

    model_id, bnb_config, load_kwargs = _resolve_load_config(quant_mode, hf_token)

    print(f"[{quant_mode}] Loading {model_id} …")
    t_load_start = time.perf_counter()
    load_kw = {"token": hf_token} if hf_token else {}
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, cache_dir="/model-cache/hf", revision=_MODEL_REVISION, **load_kw
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch.cuda.reset_peak_memory_stats()
    if quant_mode == "awq":
        # Use autoawq's native API — the transformers bridge calls set_submodule()
        # which is broken in autoawq 0.2.7+ with transformers 4.44+
        from awq import AutoAWQForCausalLM
        model = AutoAWQForCausalLM.from_quantized(
            model_id,
            fuse_layers=False,
            safetensors=True,
            **load_kw,
        )
        model.eval()
    else:
        pretrained_kwargs: dict[str, Any] = {
            "cache_dir": "/model-cache/hf",
            "device_map": "auto",
            **load_kw,
            **load_kwargs,
        }
        if bnb_config is not None:
            pretrained_kwargs["quantization_config"] = bnb_config
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=_MODEL_REVISION, **pretrained_kwargs)
        model.eval()
    load_time_s = time.perf_counter() - t_load_start
    model_vram_mb = torch.cuda.max_memory_allocated() / 1024**2
    print(f"[{quant_mode}] Model loaded in {load_time_s:.1f}s  ({model_vram_mb:.0f} MB VRAM)")
    _model_cache.commit()  # persist downloaded weights so next run skips the download

    # Clear max_length from the model's generation_config so it never conflicts with
    # max_new_tokens passed to model.generate() — avoids a noisy transformers warning.
    if hasattr(model, "generation_config") and getattr(model.generation_config, "max_length", None):
        model.generation_config.max_length = None

    # Load draft model for speculative decoding
    draft_model = None
    if quant_mode == "spec-dec":
        print(f"[{quant_mode}] Loading draft model {DRAFT_MODEL} …")
        draft_model = AutoModelForCausalLM.from_pretrained(
            DRAFT_MODEL,
            cache_dir="/model-cache/hf",
            device_map="auto",
            dtype=torch.float16,
            revision="main",
            **load_kw,
        )
        draft_model.eval()
        draft_vram_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"[{quant_mode}] Draft model loaded  ({draft_vram_mb - model_vram_mb:.0f} MB VRAM)")
        _model_cache.commit()

    result: dict[str, Any] = {
        "quant_mode": quant_mode,
        "model_id": model_id,
        "gpu": gpu_name,
        "load_time_s": round(load_time_s, 2),
    }

    if quant_mode == "spec-dec":
        result["draft_model_id"] = DRAFT_MODEL
        result["memory"] = {
            "model_weights_mb": round(model_vram_mb, 1),
            "draft_weights_mb": round(draft_vram_mb - model_vram_mb, 1),
            "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 1),
        }
    else:
        result["memory"] = _measure_memory(model_vram_mb)
    result["latency"] = _measure_latency(model, tokenizer, device, assistant_model=draft_model)
    result["throughput"] = _measure_throughput(model, tokenizer, device, assistant_model=draft_model)
    # Speculative decoding's assisted_generation does not support batched inputs in transformers
    if quant_mode != "spec-dec":
        result["batch_throughput"] = _measure_batch_throughput(model, tokenizer, device)
    result["perplexity"] = _measure_perplexity(model, tokenizer, device)
    result["quality"] = _measure_quality(model, tokenizer, device)
    if quant_mode in _MODE_NOTES:
        result["notes"] = _MODE_NOTES[quant_mode]

    return result


# ---------------------------------------------------------------------------
# vLLM benchmark path
# ---------------------------------------------------------------------------


def _run_vllm_benchmark(gpu_name: str, hf_token: str | None) -> dict[str, Any]:
    """Benchmark the same model via vLLM's LLM engine (fp16, continuous batching)."""
    import torch
    from vllm import LLM, SamplingParams

    os.environ["TRANSFORMERS_CACHE"] = "/model-cache/hf"
    os.environ["HF_HOME"] = "/model-cache/hf"

    load_kw: dict[str, Any] = {}
    if hf_token:
        load_kw["tokenizer_revision"] = _MODEL_REVISION

    print(f"[vllm] Loading {BASE_MODEL} via vLLM engine …")
    t_load_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    llm = LLM(
        model=BASE_MODEL,
        revision=_MODEL_REVISION,
        dtype="float16",
        gpu_memory_utilization=0.85,
        max_model_len=4096,  # cap context to fit KV cache in remaining VRAM after fp16 weights
        download_dir="/model-cache/hf",
        trust_remote_code=False,
    )
    load_time_s = time.perf_counter() - t_load_start
    model_vram_mb = torch.cuda.max_memory_allocated() / 1024**2
    reserved_mb = torch.cuda.memory_reserved() / 1024**2
    print(f"[vllm] Engine ready in {load_time_s:.1f}s  ({model_vram_mb:.0f} MB VRAM)")
    _model_cache.commit()

    # Warmup
    _WARMUP = 1
    _ITERS = 3
    warmup_params = SamplingParams(max_tokens=32, temperature=0.0)
    for _ in range(_WARMUP):
        llm.generate([_BENCH_PROMPTS[0]], warmup_params, use_tqdm=False)

    # Latency (single request, 256 output tokens)
    lat_params = SamplingParams(max_tokens=256, temperature=0.0)
    latencies_ms: list[float] = []
    ttfts_ms: list[float] = []
    for prompt in _BENCH_PROMPTS * _ITERS:
        t0 = time.perf_counter()
        outputs = llm.generate([prompt], lat_params, use_tqdm=False)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(elapsed_ms)
        # vLLM exposes per-token timing via output metrics
        out = outputs[0]
        if hasattr(out, "metrics") and out.metrics is not None and hasattr(out.metrics, "first_token_time"):
            ttft = (out.metrics.first_token_time - out.metrics.first_scheduled_time) * 1000
            ttfts_ms.append(ttft)

    latencies_ms.sort()
    lat_mean = sum(latencies_ms) / len(latencies_ms)
    lat_p50 = latencies_ms[len(latencies_ms) // 2]
    lat_p95 = latencies_ms[int(len(latencies_ms) * 0.95)]
    lat_p99 = latencies_ms[int(len(latencies_ms) * 0.99)]
    ttft_mean = sum(ttfts_ms) / len(ttfts_ms) if ttfts_ms else None

    # Throughput (512 output tokens, 2 iterations)
    thr_params = SamplingParams(max_tokens=512, temperature=0.0)
    thr_times: list[float] = []
    thr_tokens: list[int] = []
    for _ in range(2):
        t0 = time.perf_counter()
        outs = llm.generate([_BENCH_PROMPTS[0]], thr_params, use_tqdm=False)
        elapsed = time.perf_counter() - t0
        thr_times.append(elapsed)
        thr_tokens.append(sum(len(o.token_ids) for out in outs for o in out.outputs))
    output_tps = sum(thr_tokens) / sum(thr_times)

    # Batch throughput
    batch_thr: dict[str, float] = {}
    for batch_size in (1, 4, 8):
        prompts = (_BENCH_PROMPTS * batch_size)[:batch_size]
        t0 = time.perf_counter()
        outs = llm.generate(prompts, thr_params, use_tqdm=False)
        elapsed = time.perf_counter() - t0
        tokens = sum(len(o.token_ids) for out in outs for o in out.outputs)
        batch_thr[f"batch{batch_size}_output_tokens_per_sec"] = round(tokens / elapsed, 1)

    # MMLU quality via log-prob scoring using vLLM's encode API
    quality = _measure_quality_vllm(llm)

    return {
        "quant_mode": "vllm",
        "model_id": BASE_MODEL,
        "gpu": gpu_name,
        "load_time_s": round(load_time_s, 2),
        "memory": {
            "model_weights_mb": round(model_vram_mb, 1),
            "reserved_mb": round(reserved_mb, 1),
        },
        "latency": {
            "max_new_tokens": 256,
            "mean_ms": round(lat_mean, 1),
            "p50_ms": round(lat_p50, 1),
            "p95_ms": round(lat_p95, 1),
            "p99_ms": round(lat_p99, 1),
            "min_ms": round(min(latencies_ms), 1),
            "max_ms": round(max(latencies_ms), 1),
            "ttft_mean_ms": round(ttft_mean, 1) if ttft_mean is not None else None,
            "ttft_p95_ms": None,
        },
        "throughput": {
            "output_tokens_per_sec": round(output_tps, 1),
            "max_new_tokens": 512,
        },
        "batch_throughput": batch_thr,
        "perplexity": None,  # vLLM does not expose per-token NLL loss
        "quality": quality,
        "notes": (
            "vLLM fp16 with continuous batching (PagedAttention). "
            "Perplexity is not computed — vLLM does not expose per-token NLL. "
            "Compare latency/throughput directly against the fp16 HuggingFace baseline."
        ),
    }


def _measure_quality_vllm(llm: Any) -> dict[str, Any]:
    """MMLU log-prob scoring via vLLM's generate with logprobs."""
    from vllm import SamplingParams

    correct = 0
    details: list[dict] = []
    evaluated = _MMLU_SUBSET[:50]

    # Request logprobs for the first generated token — used to score each choice
    score_params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=5)

    for question, choices, answer_letter in evaluated:
        choice_labels = [chr(ord("A") + i) for i in range(len(choices))]
        # One forward pass: present all choices in the prompt, read the top-k logprobs
        # of the first generated token to find which letter the model prefers.
        prompt = (
            f"Question: {question}\n"
            f"A) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}\nAnswer:"
        )
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
        details.append({
            "question": question[:60] + "…" if len(question) > 60 else question,
            "predicted": predicted_label,
            "expected": answer_letter,
            "correct": is_correct,
        })

    accuracy = correct / len(evaluated)
    return {
        "mmlu_accuracy": round(accuracy, 4),
        "correct": correct,
        "total": len(evaluated),
        "details": details,
    }


# ---------------------------------------------------------------------------
# Metric helpers (all run inside the Modal function)
# ---------------------------------------------------------------------------


def _resolve_load_config(
    quant_mode: str, hf_token: str | None  # noqa: ARG001 (kept for signature consistency)
) -> tuple[str, Any, dict]:
    """Return (model_id, bnb_config_or_None, extra_from_pretrained_kwargs)."""
    import torch
    from transformers import BitsAndBytesConfig

    if quant_mode == "fp16":
        return BASE_MODEL, None, {"dtype": torch.float16}

    if quant_mode == "int8":
        cfg = BitsAndBytesConfig(load_in_8bit=True)
        return BASE_MODEL, cfg, {}

    if quant_mode == "nf4":
        cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        return BASE_MODEL, cfg, {}

    if quant_mode == "nf4-dq":
        cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        return BASE_MODEL, cfg, {}

    if quant_mode == "awq":
        model_id = os.environ.get("QUANT_AWQ_MODEL", _DEFAULT_AWQ_MODEL)
        return model_id, None, {}

    if quant_mode == "spec-dec":
        return BASE_MODEL, None, {"dtype": torch.float16}

    raise ValueError(f"Unknown quant_mode: {quant_mode!r}")


def _measure_memory(model_vram_mb: float) -> dict[str, float]:
    import torch

    reserved_mb = torch.cuda.memory_reserved() / 1024**2
    return {
        "model_weights_mb": round(model_vram_mb, 1),
        "reserved_mb": round(reserved_mb, 1),
    }


def _measure_latency(
    model: Any, tokenizer: Any, device: str, assistant_model: Any = None
) -> dict[str, Any]:
    """Latency over _BENCH_PROMPTS × 5 iterations, plus TTFT."""
    import torch

    WARMUP = 1
    ITERS = 3
    MAX_NEW_TOKENS = 256

    gen_kw: dict[str, Any] = {"max_new_tokens": MAX_NEW_TOKENS, "max_length": None, "do_sample": False}
    if assistant_model is not None:
        gen_kw["assistant_model"] = assistant_model

    all_ms: list[float] = []
    ttft_ms_list: list[float] = []

    for prompt in _BENCH_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Warmup
        for _ in range(WARMUP):
            with torch.no_grad():
                model.generate(**inputs, **gen_kw)

        # TTFT: generate exactly 1 token (≈ prefill + one decode step)
        ttft_kw = {**gen_kw, "max_new_tokens": 1}
        ttft_kw.pop("assistant_model", None)  # spec-dec TTFT measured without draft for fair comparison
        for _ in range(3):
            t0 = time.perf_counter()
            with torch.no_grad():
                model.generate(**inputs, **ttft_kw)
            ttft_ms_list.append((time.perf_counter() - t0) * 1000)

        # Full-generation latency
        for _ in range(ITERS):
            t0 = time.perf_counter()
            with torch.no_grad():
                model.generate(**inputs, **gen_kw)
            all_ms.append((time.perf_counter() - t0) * 1000)

    all_ms.sort()
    ttft_ms_list.sort()
    n = len(all_ms)
    return {
        "max_new_tokens": MAX_NEW_TOKENS,
        "mean_ms": round(statistics.mean(all_ms), 1),
        "p50_ms": round(all_ms[n // 2], 1),
        "p95_ms": round(all_ms[min(int(n * 0.95), n - 1)], 1),
        "p99_ms": round(all_ms[min(int(n * 0.99), n - 1)], 1),
        "min_ms": round(all_ms[0], 1),
        "max_ms": round(all_ms[-1], 1),
        "ttft_mean_ms": round(statistics.mean(ttft_ms_list), 1),
        "ttft_p95_ms": round(sorted(ttft_ms_list)[int(len(ttft_ms_list) * 0.95)], 1),
    }


def _measure_throughput(
    model: Any, tokenizer: Any, device: str, assistant_model: Any = None
) -> dict[str, float]:
    """Tokens/sec for output tokens and total tokens."""
    import torch

    MAX_NEW_TOKENS = 512
    ITERS = 2
    out_tps_list: list[float] = []
    total_tps_list: list[float] = []

    gen_kw: dict[str, Any] = {"max_new_tokens": MAX_NEW_TOKENS, "max_length": None, "do_sample": False}
    if assistant_model is not None:
        gen_kw["assistant_model"] = assistant_model

    for prompt in _BENCH_PROMPTS[:3]:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        for _ in range(ITERS):
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(**inputs, **gen_kw)
            elapsed = time.perf_counter() - t0
            output_len = out.shape[1] - input_len
            out_tps_list.append(output_len / elapsed)
            total_tps_list.append(out.shape[1] / elapsed)

    return {
        "output_tokens_per_sec": round(statistics.mean(out_tps_list), 1),
        "total_tokens_per_sec": round(statistics.mean(total_tps_list), 1),
        "max_new_tokens": MAX_NEW_TOKENS,
    }


def _measure_batch_throughput(
    model: Any, tokenizer: Any, device: str, batch_sizes: list[int] | None = None
) -> dict[str, Any]:
    """Measure output tokens/sec at multiple batch sizes using left-padded inputs."""
    import torch

    if batch_sizes is None:
        batch_sizes = [1, 4, 8]

    MAX_NEW_TOKENS = 256
    ITERS = 2
    prompt = _BENCH_BENCH_PROMPTS[0]
    gen_kw: dict[str, Any] = {"max_new_tokens": MAX_NEW_TOKENS, "max_length": None, "do_sample": False}

    tokenizer.padding_side = "left"
    results: dict[str, float] = {}

    for bs in batch_sizes:
        prompts = [prompt] * bs
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        input_len = inputs["input_ids"].shape[1]

        tps_list: list[float] = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(**inputs, **gen_kw)
            elapsed = time.perf_counter() - t0
            output_tokens = (out.shape[1] - input_len) * bs
            tps_list.append(output_tokens / elapsed)

        results[f"batch{bs}_output_tokens_per_sec"] = round(statistics.mean(tps_list), 1)

    return results


def _measure_perplexity(model: Any, tokenizer: Any, device: str) -> dict[str, float]:
    """Sliding-window perplexity on WikiText-2 test set."""
    import math

    import torch
    from datasets import load_dataset

    STRIDE = 128
    MAX_LEN = 1024  # tokens to evaluate (keep it fast)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])  # type: ignore[index]
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = min(encodings.input_ids.size(1), MAX_LEN)
    input_ids = encodings.input_ids[:, :seq_len].to(device)

    nlls: list[torch.Tensor] = []
    prev_end = 0
    for begin in range(0, seq_len, STRIDE):
        end = min(begin + tokenizer.model_max_length, seq_len)
        target_len = end - prev_end
        with torch.no_grad():
            out = model(input_ids[:, begin:end], labels=input_ids[:, begin:end])
            # scale NLL to only count the non-overlapping tokens
            nlls.append(out.loss * target_len)
        prev_end = end
        if end == seq_len:
            break

    ppl = math.exp(torch.stack(nlls).sum().item() / seq_len)
    return {
        "perplexity": round(ppl, 3),
        "eval_tokens": seq_len,
        "dataset": "wikitext-2-raw-v1/test",
    }


def _measure_quality(model: Any, tokenizer: Any, device: str) -> dict[str, Any]:
    """MMLU subset accuracy: zero-shot multiple-choice via log-prob scoring."""
    import torch

    correct = 0
    details: list[dict] = []
    evaluated = _MMLU_SUBSET[:50]

    for question, choices, answer_letter in evaluated:
        answer_idx = ord(answer_letter) - ord("A")
        log_probs: list[float] = []

        for choice in choices:
            prompt = f"Question: {question}\nAnswer: {choice}"
            enc = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**enc, labels=enc["input_ids"])
            # lower loss = higher probability
            log_probs.append(-out.loss.item())

        predicted_idx = int(max(range(len(log_probs)), key=lambda i: log_probs[i]))
        predicted_letter = chr(ord("A") + predicted_idx)
        is_correct = predicted_idx == answer_idx
        if is_correct:
            correct += 1
        details.append({
            "question": question[:60] + "…" if len(question) > 60 else question,
            "predicted": predicted_letter,
            "expected": answer_letter,
            "correct": is_correct,
        })

    accuracy = correct / len(evaluated)
    return {
        "mmlu_accuracy": round(accuracy, 4),
        "correct": correct,
        "total": len(evaluated),
        "details": details,
    }


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


# Modal GPU pricing (USD/hr) — used for cost-per-token annotation.
_GPU_COST_PER_HR: dict[str, float] = {
    "T4": 0.59,
    "A10G": 1.10,
    "A100-40GB": 3.70,
    "A100-80GB": 4.00,
    "H100": 6.45,
}


def _gpu_output_path(base: str, gpu: str) -> Path:
    """Return a GPU-specific output path, e.g. results/modal_quant_a10g.json."""
    p = Path(base)
    slug = gpu.lower().replace("-", "_")
    # Only inject GPU slug when the caller used the default name pattern
    if p.stem == "modal_quant_benchmark" or p.stem.startswith("modal_quant_"):
        return p.parent / f"modal_quant_{slug}.json"
    return p


@app.local_entrypoint()
def main(
    output: str = "results/modal_quant_benchmark.json",
    gpu: str = "A10G",
    modes: str = ",".join(_ALL_MODES),
    merge: bool = False,
) -> None:
    """Fan out benchmark across all quant modes in parallel, write JSON results.

    Args:
        output: Base path for the JSON results file. The GPU type is
                automatically injected into the filename so each GPU gets its
                own file (e.g. modal_quant_a10g.json, modal_quant_a100_40gb.json).
        gpu:    Modal GPU type (T4, A10G, A100-40GB, A100-80GB, H100).
        modes:  Comma-separated quant modes to run (default: all).
        merge:  If True and the GPU's output file already exists, keep results
                for modes not being re-run. Same-mode results are replaced;
                results from other modes are preserved. Different GPUs always
                write to separate files and never interfere.
    """
    selected = [m.strip() for m in modes.split(",") if m.strip()]
    invalid = [m for m in selected if m not in _ALL_MODES]
    if invalid:
        raise SystemExit(f"Unknown modes: {invalid}. Valid: {list(_ALL_MODES)}")

    out_path = _gpu_output_path(output, gpu)
    gpu_cost_per_hr = _GPU_COST_PER_HR.get(gpu, 1.10)

    print(f"Running quantization benchmark on {gpu} for modes: {selected}")
    print(f"Output → {out_path}  (GPU cost: ${gpu_cost_per_hr}/hr)")
    print("Results will stream in as each mode completes (parallel execution).\n")

    bench_fn = run_quant_benchmark.with_options(gpu=gpu) if gpu != "A10G" else run_quant_benchmark

    # Load existing results for this GPU if merging
    existing: dict[str, dict] = {}
    if merge and out_path.exists():
        try:
            prior = json.loads(out_path.read_text())
            existing = {r["quant_mode"]: r for r in prior.get("results", [])}
            kept = [m for m in existing if m not in selected]
            print(f"Merging: keeping existing results for {kept}, replacing {selected}\n")
        except Exception as e:
            print(f"Warning: could not read existing results for merge ({e}), starting fresh.\n")

    t_start = time.perf_counter()
    new_results: list[dict] = []

    for result in bench_fn.map(selected, order_outputs=False, return_exceptions=True):
        if isinstance(result, Exception):
            print(f"  [FAILED] {result}")
            continue
        new_results.append(result)
        mode = result["quant_mode"]
        ppl_raw = result["perplexity"]
        ppl_str = f"{ppl_raw['perplexity']:.2f}" if ppl_raw else "n/a"
        lat = result["latency"]["mean_ms"]
        tps = result["throughput"]["output_tokens_per_sec"]
        acc = result["quality"]["mmlu_accuracy"]
        mem = result["memory"]["model_weights_mb"]
        print(
            f"  [{mode:8s}] ppl={ppl_str}  lat={lat:.0f}ms  "
            f"tps={tps:.0f}  mmlu={acc:.0%}  vram={mem:.0f}MB"
        )

    total_s = time.perf_counter() - t_start

    # Merge new results over existing, then sort by canonical mode order
    merged = {**existing, **{r["quant_mode"]: r for r in new_results}}
    all_results = sorted(merged.values(), key=lambda r: _ALL_MODES.index(r["quant_mode"]))
    all_modes = [r["quant_mode"] for r in all_results]

    # Annotate each result with cost per 1k output tokens using actual GPU rate
    gpu_cost_per_sec = gpu_cost_per_hr / 3600
    for r in all_results:
        tps = r["throughput"]["output_tokens_per_sec"]
        if tps > 0:
            r["cost_per_1k_output_tokens_usd"] = round(gpu_cost_per_sec / tps * 1000, 4)

    summary = {
        "benchmark": "modal_quant_benchmark",
        "base_model": BASE_MODEL,
        "gpu": gpu,
        "gpu_cost_per_hr_usd": gpu_cost_per_hr,
        "total_wall_time_s": round(total_s, 1),
        "modes_run": all_modes,
        "results": all_results,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {len(all_results)} results → {out_path}  ({total_s:.0f}s total)")
