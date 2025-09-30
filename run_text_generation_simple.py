import torch
import numpy as np
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
    input_dim = X.shape[1]
    output_dim = Y.shape[1]
    W = torch.randn(input_dim, output_dim) * 0.01
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
        W = W - lr * grad
    
    return W

def generate_text(W, context, word_to_idx, idx_to_word, n=4, num_tokens=20):
    vocab_size = len(word_to_idx)
    context_words = context.strip().split()[-n:]
    if len(context_words) < n:
        context_words = [''] * (n - len(context_words)) + context_words
    
    generated_text = ' '.join(context_words)
    
    for _ in range(num_tokens):
        # Create input vector for context
        x = torch.zeros(vocab_size * n)
        for i, word in enumerate(context_words):
            if word in word_to_idx:
                x[i * vocab_size + word_to_idx[word]] = 1
        
        # Get scores and probabilities
        scores = x @ W
        probs = torch.exp(scores)
        probs = probs / probs.sum()
        
        # Sample next token
        next_idx = torch.multinomial(probs, 1).item()
        next_word = idx_to_word[next_idx]
        
        generated_text += ' ' + next_word
        context_words = context_words[1:] + [next_word]
    
    return generated_text

# Example text data (simple sentences for testing)
text_data = [
    "once there was a little girl who loved to play in the garden with her friends",
    "she had a beautiful red dress and long hair that sparkled in the sun",
    "every morning she would wake up early to watch the sunrise",
    "the birds would sing their sweet songs as she walked through the flowers",
    "one day she found a magical butterfly that granted her three wishes"
]

# Prepare training data
X_train, Y_train, word_to_idx = prepare_data(text_data, n=4)
idx_to_word = {i: word for word, i in word_to_idx.items()}

# Train model
embedding_dim = 10
W = cross_entropy_gd(X_train, Y_train, embedding_dim=embedding_dim, num_iter=1000)

# Initial contexts to try
contexts = [
    "the little girl who",
    "she had a beautiful",
    "every morning she would",
    "the birds would sing",
    "one day she found"
]

print("\nGenerated Text Samples:")
print("-" * 50)
samples = []
for i, context in enumerate(contexts, 1):
    generated = generate_text(W, context, word_to_idx, idx_to_word, n=4, num_tokens=20)
    samples.append(f"Sample {i} (context: '{context}'): {generated}")
    print(samples[-1])
    print("-" * 50)

# Save the samples to a file that will be included in the LaTeX document
with open('text_samples.txt', 'w') as f:
    f.write('\n\n'.join(samples))
