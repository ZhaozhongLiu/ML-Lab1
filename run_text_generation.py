import torch
import numpy as np
from hw1_utils import load_ntp_data
import string

def create_ngrams(text, n=4):
    words = text.split()
    ngrams = []
    for i in range(len(words) - n):
        ngrams.append((words[i:i+n], words[i+n]))
    return ngrams

def prepare_data(text_data, n=4):
    # Create vocabulary
    words = set()
    for text in text_data:
        words.update(text.split())
    word_to_idx = {word: i for i, word in enumerate(sorted(words))}
    
    # Create ngrams
    all_ngrams = []
    for text in text_data:
        all_ngrams.extend(create_ngrams(text, n))
    
    # Create training data
    vocab_size = len(word_to_idx)
    X = []
    Y = []
    
    for context, target in all_ngrams:
        # Create input vector
        x = torch.zeros(vocab_size * n)
        for i, word in enumerate(context):
            x[i * vocab_size + word_to_idx[word]] = 1
        X.append(x)
        
        # Create target vector
        y = torch.zeros(vocab_size)
        y[word_to_idx[target]] = 1
        Y.append(y)
    
    return torch.stack(X), torch.stack(Y), word_to_idx

def cross_entropy_gd(X, Y, embedding_dim=10, num_iter=1000, lr=0.1):
    vocab_size = Y.shape[1]
    W = torch.randn(embedding_dim, vocab_size) * 0.01
    n = X.shape[0]
    
    for _ in range(num_iter):
        # Forward pass
        scores = X @ W
        # Compute softmax
        exp_scores = torch.exp(scores)
        probs = exp_scores / exp_scores.sum(dim=1, keepdim=True)
        
        # Gradient
        grad = 1/n * X.T @ (probs - Y)
        
        # Update
        W = W - lr * grad.T
    
    return W

def generate_text(W, context, vocab, n=4, num_tokens=20):
    # Convert context to indices
    context_indices = [vocab.get(c, 0) for c in context[-n:]]
    if len(context_indices) < n:
        context_indices = [0] * (n - len(context_indices)) + context_indices
    
    generated_text = context
    
    for _ in range(num_tokens):
        # Create one-hot vectors for context
        X = torch.zeros(1, len(vocab) * n)
        for i, idx in enumerate(context_indices):
            X[0, i * len(vocab) + idx] = 1
        
        # Get scores and probabilities
        scores = X @ W
        probs = torch.exp(scores)
        probs = probs / probs.sum()
        
        # Sample next token
        next_idx = torch.multinomial(probs, 1).item()
        next_char = list(vocab.keys())[next_idx]
        
        generated_text += next_char
        context_indices = context_indices[1:] + [next_idx]
    
    return generated_text

# Load data and create vocabulary
X_train, Y_train, vocab = load_ntp_data()

# Train model
embedding_dim = 10
W = cross_entropy_gd(X_train, Y_train, embedding_dim=embedding_dim)

# Initial contexts to try
contexts = [
    "The ",
    "In t",
    "She ",
    "A lo",
    "With"
]

print("\nGenerated Text Samples:")
print("-" * 50)
samples = []
for i, context in enumerate(contexts, 1):
    generated = generate_text(W, context, vocab, n=4, num_tokens=25)
    samples.append(f"Sample {i} (context: '{context}'): {generated}")
    print(samples[-1])
    print("-" * 50)

# Save the samples to a file that will be included in the LaTeX document
with open('text_samples.txt', 'w') as f:
    f.write('\n\n'.join(samples))
