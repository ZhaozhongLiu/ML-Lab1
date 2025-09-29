import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
import string

def load_reg_data():
    # load the regression synthetic data
    torch.manual_seed(0) # force seed so same data is generated every time

    X = torch.linspace(0, 4, 100).reshape(-1, 1)
    noise = torch.normal(0, .4, size=X.shape)
    w = 0.5
    b = 1.
    Y = w * X**2 + b + noise

    return X, Y

def load_xor_data():
    X = torch.tensor([[-1,1],[1,-1],[-1,-1],[1,1]]).float()
    Y = torch.prod(X,axis=1)

    return X, Y

def load_logistic_data():
    torch.manual_seed(0) # reset seed
    return linear_problem(torch.tensor([-1., 2.]), margin=1.5, size=200)

def contour_plot(xmin, xmax, ymin, ymax, pred_fxn, ngrid=33, *, levels=None,
                 filled=False, ax=None, show=True, contour_labels=True, cmap='coolwarm'):
    """
    Make a contour plot for a 2D score function.
    - xmin, xmax, ymin, ymax: plot bounds
    - pred_fxn: function mapping an (n x d) FloatTensor to (n x 1) or (n,) scores
    - ngrid: number of grid points per axis
    - levels: list/array of contour levels (e.g., [0] to show only the decision boundary)
    - filled: if True, use contourf; otherwise use contour lines
    - ax: optional matplotlib Axes to draw on; defaults to current axes
    - show: whether to call plt.show()
    - contour_labels: whether to label contour lines
    - cmap: matplotlib colormap name
    """
    # Build grid (xy indexing for intuitive orientation)
    xgrid = torch.linspace(xmin, xmax, ngrid)
    ygrid = torch.linspace(ymin, ymax, ngrid)
    try:
        xx, yy = torch.meshgrid(xgrid, ygrid, indexing='xy')
    except TypeError:
        xx, yy = torch.meshgrid(xgrid, ygrid)

    # Get predictions on grid points
    features = torch.dstack((xx, yy)).reshape(-1, 2)
    with torch.no_grad():
        pred = pred_fxn(features)
    if isinstance(pred, torch.Tensor):
        pred = pred.reshape(-1)
        zz = pred.reshape(xx.shape)
    else:
        # fall back to numpy-like output
        zz = torch.as_tensor(pred).reshape(xx.shape)

    # Plot
    ax = ax if ax is not None else plt.gca()
    if filled:
        C = ax.contourf(xx, yy, zz, levels=levels, cmap=cmap)
    else:
        C = ax.contour(xx, yy, zz, levels=levels, cmap=cmap)
        if contour_labels:
            ax.clabel(C)
    if show:
        plt.show()
    return ax.figure

def linear_problem(w, margin, size, bounds=[-5., 5.], trans=0.0):
    in_margin = lambda x: torch.abs(w.flatten().dot(x.flatten())) / torch.norm(w) \
                          < margin
    half_margin = lambda x: 0.6*margin < w.flatten().dot(x.flatten()) / torch.norm(w) < 0.65*margin
    X = []
    Y = []
    for i in range(size):
        x = torch.zeros(2).uniform_(bounds[0], bounds[1]) + trans
        while in_margin(x):
            x.uniform_(bounds[0], bounds[1]) + trans
        if w.flatten().dot(x.flatten()) + trans > 0:
            Y.append(torch.tensor(1.))
        else:
            Y.append(torch.tensor(-1.))
        X.append(x)
    for j in range(1):
        x_out = torch.zeros(2).uniform_(bounds[0], bounds[1]) + trans
        while not half_margin(x_out):
            x_out = torch.zeros(2).uniform_(bounds[0], bounds[1]) + trans
        X.append(x_out)
        Y.append(torch.tensor(-1.))
    X = torch.stack(X)
    Y = torch.stack(Y).reshape(-1, 1)

    return X, Y

def load_ntp_data():
    ds = load_dataset("roneneldan/TinyStories", streaming=True)
    data = [x['text'] for x in ds['train'].take(10)]
    words = set()
    tokenized_data = []
    translator = str.maketrans('', '', string.punctuation)

    for sentence in data:
        cleaned_sentence = sentence.translate(translator).lower().split()
        tokenized_data.append(cleaned_sentence)
        words.update(cleaned_sentence)
    
    sorted_words = sorted(words)
    word_to_idx = {word: i for (i, word) in enumerate(sorted_words)}

    return tokenized_data, sorted_words, word_to_idx

def load_random_embeddings(vocab_size, embedding_dim):
    torch.manual_seed(0) # force seed so same data is generated every time
    X = torch.rand(vocab_size, embedding_dim)

    return X
