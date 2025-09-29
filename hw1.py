import torch
import hw1_utils as utils
import matplotlib.pyplot as plt
import torch.nn.functional as F

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
    n, d = X.shape
    X_tilde = torch.cat([torch.ones(n, 1), X], dim=1)
    w = torch.zeros(d + 1, 1)
    for _ in range(num_iter):
        y_pred = X_tilde @ w
        grad = (1.0 / n) * X_tilde.t() @ (y_pred - Y)
        w = w - lrate * grad
    return w


def linear_normal(X, Y):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    '''
    n, d = X.shape
    X_tilde = torch.cat([torch.ones(n, 1), X], dim=1)
    w = torch.pinverse(X_tilde) @ Y
    return w

def plot_linear():
    '''
        Returns:
            Figure: the figure plotted with matplotlib
    '''
    X, Y = utils.load_reg_data()
    w = linear_normal(X, Y)
    n = X.shape[0]
    X_tilde = torch.cat([torch.ones(n, 1), X], dim=1)
    Y_pred = X_tilde @ w
    plt.scatter(X.squeeze(), Y.squeeze(), label='Data')
    plt.plot(X.squeeze(), Y_pred.squeeze(), color='red', label='Linear Fit')
    plt.legend()
    plt.show()
    return plt.gcf()

# Problem 3
def _poly_features(X):
    n, d = X.shape
    features = [torch.ones(n, 1), X]
    quad_terms = []
    for i in range(d):
        quad_terms.append((X[:, i] ** 2).reshape(-1, 1))
        for j in range(i + 1, d):
            quad_terms.append((X[:, i] * X[:, j]).reshape(-1, 1))
    features.extend(quad_terms)
    return torch.cat(features, dim=1)

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
    X_poly = _poly_features(X)
    n, D = X_poly.shape
    w = torch.zeros(D, 1)
    for _ in range(num_iter):
        y_pred = X_poly @ w
        grad = (1.0 / n) * X_poly.t() @ (y_pred - Y)
        w = w - lrate * grad
    return w

def poly_normal(X,Y):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels

    Returns:
        (1 + d + d * (d + 1) / 2) x 1 FloatTensor: the parameters w
    '''
    X_poly = _poly_features(X)
    w = torch.pinverse(X_poly) @ Y
    return w

def plot_poly():
    '''
    Returns:
        Figure: the figure plotted with matplotlib
    '''
    X, Y = utils.load_reg_data()
    w = poly_normal(X, Y)
    X_poly = _poly_features(X)
    Y_pred = X_poly @ w
    plt.scatter(X.squeeze(), Y.squeeze(), label='Data')
    plt.plot(X.squeeze(), Y_pred.squeeze(), color='green', label='Poly Fit')
    plt.legend()
    plt.show()
    return plt.gcf()

def poly_xor():
    '''
    Returns:
        n x 1 FloatTensor: the linear model's predictions on the XOR dataset
        n x 1 FloatTensor: the polynomial model's predictions on the XOR dataset
    '''
    X, Y = utils.load_xor_data()
    w_lin = linear_normal(X, Y.reshape(-1, 1))
    w_poly = poly_normal(X, Y.reshape(-1, 1))
    def pred_lin(Z):
        Z_tilde = torch.cat([torch.ones(Z.shape[0], 1), Z], dim=1)
        return (Z_tilde @ w_lin).reshape(-1, 1)
    def pred_poly(Z):
        Z_poly = _poly_features(Z)
        return (Z_poly @ w_poly).reshape(-1, 1)
    utils.contour_plot(-2, 2, -2, 2, pred_lin, levels=[0], cmap='coolwarm')
    utils.contour_plot(-2, 2, -2, 2, pred_poly, levels=[0], cmap='coolwarm')
    lin_pred = pred_lin(X)
    poly_pred = pred_poly(X)
    return lin_pred, poly_pred

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
    n, d = X.shape
    X_tilde = torch.cat([torch.ones(n, 1), X], dim=1)
    w = torch.zeros(d + 1, 1)
    for _ in range(num_iter):
        logits = X_tilde @ w
        grad = -(1.0 / n) * X_tilde.t() @ (Y * torch.sigmoid(-Y * logits))
        w = w - lrate * grad
    return w

def logistic_vs_ols():
    '''
    Returns:
        Figure: the figure plotted with matplotlib
    '''
    X, Y = utils.load_logistic_data()
    w_log = logistic(X, Y, lrate=0.01, num_iter=100000)
    w_ols = linear_gd(X, Y, lrate=0.01, num_iter=1000)
    plt.scatter(X[:, 0], X[:, 1], c=Y.squeeze(), cmap='bwr', label='Data')
    x1 = torch.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    def plot_boundary(w, color, label):
        w0, w1, w2 = w.squeeze()
        if abs(w2) > 1e-6:
            x2 = -(w0 + w1 * x1) / w2
            plt.plot(x1, x2, color=color, label=label)
    plot_boundary(w_log, 'green', 'Logistic Boundary')
    plot_boundary(w_ols, 'red', 'OLS Boundary')
    plt.legend()
    plt.show()
    return plt.gcf()

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
    n, d = X.shape
    W = torch.zeros(d, k)
    Y = Y.view(-1)
    T = F.one_hot(Y, num_classes=k).float()
    for _ in range(num_iter):
        logits = X @ W
        p = torch.softmax(logits, dim=1)
        grad_logits = (p - T) / n
        grad_W = X.t() @ grad_logits
        W = W - lrate * grad_W
    return W

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
    tokenized_data, sorted_words, word_to_idx = utils.load_ntp_data()
    vocab_size = len(sorted_words)
    embeddings = utils.load_random_embeddings(vocab_size, embedding_dim)
    X_list, Y_list = [], []
    for sentence in tokenized_data:
        if len(sentence) < n + 1:
            continue
        idxs = [word_to_idx[w] for w in sentence]
        for i in range(len(sentence) - n):
            context_idxs = idxs[i:i+n]
            target_idx = idxs[i+n]
            context_emb = torch.cat([embeddings[j] for j in context_idxs])
            X_list.append(context_emb)
            Y_list.append(target_idx)
    X = torch.stack(X_list)
    Y = torch.tensor(Y_list)
    W = cross_entropy(X, Y, vocab_size)
    return W

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
    tokenized_data, sorted_words, word_to_idx = utils.load_ntp_data()
    vocab_size = len(sorted_words)
    embeddings = utils.load_random_embeddings(vocab_size, embedding_dim)
    context_tokens = context.lower().split()
    idxs = [word_to_idx.get(tok, 0) for tok in context_tokens]
    if len(idxs) < n:
        idxs = [0] * (n - len(idxs)) + idxs
    else:
        idxs = idxs[-n:]
    generated = context_tokens.copy()
    for _ in range(num_tokens):
        context_emb = torch.cat([embeddings[j] for j in idxs[-n:]])
        logits = w.t() @ context_emb
        next_id = torch.argmax(logits).item()
        next_word = sorted_words[next_id]
        generated.append(next_word)
        idxs.append(next_id)
    return ' '.join(generated)
