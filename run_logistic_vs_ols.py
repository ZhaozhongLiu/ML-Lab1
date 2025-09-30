import torch
import matplotlib.pyplot as plt
from hw1_utils import load_logistic_data

def logistic(X, Y, num_iter=1000000, lr=0.1):
    # Add bias term
    X_bias = torch.cat([torch.ones(X.shape[0], 1), X], dim=1)
    w = torch.zeros(X_bias.shape[1], 1)
    n = X.shape[0]
    
    for _ in range(num_iter):
        # Forward pass
        z = X_bias @ w
        sigmoid = 1 / (1 + torch.exp(-z))
        
        # Gradient
        grad = -1/n * X_bias.T @ ((Y - sigmoid) * sigmoid * (1 - sigmoid))
        
        # Update
        w = w - lr * grad
    
    return w

def linear_gd(X, Y, num_iter=1000, lr=0.1):
    # Add bias term
    X_bias = torch.cat([torch.ones(X.shape[0], 1), X], dim=1)
    w = torch.zeros(X_bias.shape[1], 1)
    n = X.shape[0]
    
    for _ in range(num_iter):
        # Gradient
        grad = -1/n * X_bias.T @ (Y - X_bias @ w)
        
        # Update
        w = w - lr * grad
    
    return w

# Load data
X, Y = load_logistic_data()

# Train models
w_logistic = logistic(X, Y)
w_linear = linear_gd(X, Y)

# Create plot
plt.figure(figsize=(10, 5))

# Plot data points
plt.subplot(1, 2, 1)
plt.scatter(X[Y.flatten() == 1, 0], X[Y.flatten() == 1, 1], c='blue', label='y=1')
plt.scatter(X[Y.flatten() == -1, 0], X[Y.flatten() == -1, 1], c='red', label='y=-1')

# Plot logistic regression decision boundary
x1 = torch.linspace(-5, 5, 100)
if abs(w_logistic[2]) > 1e-6:
    x2 = -(w_logistic[0] + w_logistic[1] * x1) / w_logistic[2]
    plt.plot(x1, x2, '-g', label='Decision Boundary')
else:
    x2 = torch.linspace(-5, 5, 100)
    x1_boundary = -w_logistic[0] / w_logistic[1]
    plt.axvline(x=x1_boundary, color='g', label='Decision Boundary')

plt.title('Logistic Regression')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.grid(True)

# Plot OLS decision boundary
plt.subplot(1, 2, 2)
plt.scatter(X[Y.flatten() == 1, 0], X[Y.flatten() == 1, 1], c='blue', label='y=1')
plt.scatter(X[Y.flatten() == -1, 0], X[Y.flatten() == -1, 1], c='red', label='y=-1')

if abs(w_linear[2]) > 1e-6:
    x2 = -(w_linear[0] + w_linear[1] * x1) / w_linear[2]
    plt.plot(x1, x2, '-g', label='Decision Boundary')
else:
    x2 = torch.linspace(-5, 5, 100)
    x1_boundary = -w_linear[0] / w_linear[1]
    plt.axvline(x=x1_boundary, color='g', label='Decision Boundary')

plt.title('Ordinary Least Squares')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('4-c-output.png', dpi=300, bbox_inches='tight')
plt.close()
