import torch
import hw1_utils as utils
import matplotlib.pyplot as plt

'''
    Important
    ========================================
    The autograder evaluates your code using FloatTensors for all computations.
    If you use DoubleTensors, your results will not match those of the autograder
    due to the higher precision.

    PyTorch constructs FloatTensors by default, so simply don't explicitly
    convert your tensors to DoubleTensors or change the default tensor.

    Be sure to modify your input matrix X in exactly the way specified. That is,
    make sure to prepend the column of ones to X and not put the column anywhere
    else, and make sure your feature-expanded matrix in Problem 3 is in the
    specified order (otherwise, your w will be ordered differently than the
    reference solution's in the autograder).
'''

# Problem 2
def linear_gd(X, Y, lrate=0.01, num_iter=1000):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels
        num_iter (int): iterations of gradient descent to perform

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    '''
    pass

def linear_normal(X, Y):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    '''
    pass

def plot_linear():
    '''
        Returns:
            Figure: the figure plotted with matplotlib
    '''
    pass

# Problem 3
def poly_gd(X, Y, lrate=0.01, num_iter=1000):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels
        lrate (float): the learning rate
        num_iter (int): number of iterations of gradient descent to perform

    Returns:
        (1 + d + d * (d + 1) / 2) x 1 FloatTensor: the parameters w
    '''
    pass

def poly_normal(X,Y):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels

    Returns:
        (1 + d + d * (d + 1) / 2) x 1 FloatTensor: the parameters w
    '''
    pass

def plot_poly():
    '''
    Returns:
        Figure: the figure plotted with matplotlib
    '''
    pass

def poly_xor():
    '''
    Returns:
        n x 1 FloatTensor: the linear model's predictions on the XOR dataset
        n x 1 FloatTensor: the polynomial model's predictions on the XOR dataset
    '''
    pass

# Problem 4
def logistic(X, Y, lrate=.01, num_iter=1000):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels (use ±1 labels)
        lrate (float): learning rate for gradient descent
        num_iter (int): number of GD iterations

    Implementation guidance:
        - Prepend a column of ones to X to form X_tilde of shape (n x (d+1)).
        - Keep vector/matrix shapes consistent: X_tilde @ w -> (n x 1), w -> ((d+1) x 1), Y -> (n x 1).
        - Use fully vectorized math; do not loop over examples.
        - Average the gradient over examples (divide by n), do not sum.
        - Use torch.sigmoid for clarity; for numerical stability of the loss, consider
          torch.special.softplus(-Y * (X_tilde @ w)) (i.e., log(1+exp(.))) or torch.logaddexp(0, -Y * (X_tilde @ w)).

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w (bias first)
    '''
    pass

def logistic_vs_ols():
    '''
    Returns:
        Figure: the figure plotted with matplotlib
    '''
    pass

# Problem 5
def cross_entropy(X, Y, k, lrate=.01, num_iter=1000):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix (no bias term)
        Y (n x 1 LongTensor/FloatTensor): integer class labels in {0, ..., k-1}
        k (int): the number of classes (vocabulary size)

    Implementation guidance:
        - Keep shapes consistent: X @ W -> (n x k), W -> (d x k), Y -> (n x 1).
        - Compute p = softmax(X @ W, dim=1).
        - Build one-hot targets T via F.one_hot(Y.view(-1), num_classes=k).float() or
          torch.zeros(n, k, device=X.device).scatter_(1, Y, 1.0).
        - Gradient on logits is (p - T) / n (average over batch, do not sum).
        - Gradient on weights: grad_W = X.t() @ grad_logits.
        - Update W with the given learning rate; keep everything vectorized and in float32.

    Returns:
        d x k FloatTensor: the parameters W
    '''
    pass

def get_ntp_weights(n, embedding_dim=10):
    '''
    Arguments:
        n (int): the context size (number of previous tokens)
        embedding_dim (int): the size of each word embedding

    Implementation guidance:
        - Use utils.load_ntp_data() to get (tokenized_data, sorted_words, word_to_idx).
        - Build N-grams: for each sentence with length >= n+1, create training samples from every
          sequence of n+1 consecutive words: first n = context, last = target.
        - Use utils.load_random_embeddings(vocab_size, embedding_dim) to get embeddings.
        - Each row of X is the concatenation of embeddings for the n context tokens;
          thus d = n * embedding_dim. Y holds the target word index.
        - Let k = vocab_size = len(sorted_words). Fit W via cross_entropy(X, Y, k) using default hyperparameters.

    Returns:
        d x k FloatTensor: the parameters W (no bias)
    '''
    pass

def generate_text(w, n, num_tokens, embedding_dim=10, context = "once upon a time"):
    '''
    Arguments:
        w (d x k FloatTensor): the parameters W (no bias)
        n (int): the context size (expected >= 1)
        num_tokens (int): the number of additional tokens to generate
        embedding_dim (int): the size of each word embedding
        context (str): the initial string provided to the model (at least n tokens)

    Implementation guidance:
        - Use utils.load_ntp_data() and utils.load_random_embeddings() to obtain the vocabulary and embeddings.
        - Tokenize context via the same mapping; if context has more than n tokens, use the last n.
        - At each step: build current feature by concatenating embeddings of the last n token indices (shape d = n*embedding_dim);
          compute logits = w.t() @ feature (shape k); pick next_id = logits.argmax(); append and repeat.

    Returns:
        String: the initial context plus generated words, space-separated
    '''
    pass
