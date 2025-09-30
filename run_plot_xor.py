import torch
import matplotlib.pyplot as plt
from hw1_utils import load_xor_data, contour_plot

def linear_normal(X, Y):
    # Add bias term to X
    X_bias = torch.cat([torch.ones(X.shape[0], 1), X], dim=1)
    # Normal equations solution
    w = torch.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ Y
    return w

def poly_normal(X, Y):
    # Create polynomial features [1, x1, x2, x1^2, x1x2, x2^2]
    n = X.shape[0]
    x1, x2 = X[:, 0], X[:, 1]
    X_poly = torch.stack([
        torch.ones(n),
        x1, x2,
        x1**2, x1*x2, x2**2
    ], dim=1)
    
    # Normal equations solution with regularization
    lambda_reg = 1e-6  # small regularization term
    n_features = X_poly.shape[1]
    w = torch.linalg.inv(X_poly.T @ X_poly + lambda_reg * torch.eye(n_features)) @ X_poly.T @ Y
    return w

# Load XOR data
X, Y = load_xor_data()

# Get model parameters
w_lin = linear_normal(X, Y)
w_poly = poly_normal(X, Y)

# Create prediction functions for contour plot
def pred_lin(Z):
    Z_bias = torch.cat([torch.ones(Z.shape[0], 1), Z], dim=1)
    return Z_bias @ w_lin

def pred_poly(Z):
    n = Z.shape[0]
    z1, z2 = Z[:, 0], Z[:, 1]
    Z_poly = torch.stack([
        torch.ones(n),
        z1, z2,
        z1**2, z1*z2, z2**2
    ], dim=1)
    return Z_poly @ w_poly

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot linear model
ax1.scatter(X[:, 0][Y == 1], X[:, 1][Y == 1], c='blue', label='y=1')
ax1.scatter(X[:, 0][Y == -1], X[:, 1][Y == -1], c='red', label='y=-1')
contour_plot(-2, 2, -2, 2, pred_lin, ngrid=100, levels=[-1, -0.5, 0, 0.5, 1], 
             ax=ax1, show=False, cmap='coolwarm')
ax1.set_title('Linear Model')
ax1.legend()

# Plot polynomial model
ax2.scatter(X[:, 0][Y == 1], X[:, 1][Y == 1], c='blue', label='y=1')
ax2.scatter(X[:, 0][Y == -1], X[:, 1][Y == -1], c='red', label='y=-1')
contour_plot(-2, 2, -2, 2, pred_poly, ngrid=100, levels=[-1, -0.5, 0, 0.5, 1], 
             ax=ax2, show=False, cmap='coolwarm')
ax2.set_title('Polynomial Model')
ax2.legend()

plt.tight_layout()
plt.savefig('3-e-output.png', dpi=300, bbox_inches='tight')
plt.close()
