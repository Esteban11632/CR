import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

def check_gradient_flow(model):
    """Check if gradients are dying"""
    print("\n=== Gradient Flow ===")
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_mean = param.grad.abs().mean().item()
            grad_max = param.grad.abs().max().item()
            if grad_mean < 1e-7:
                print(f"{name}: mean={grad_mean:.2e}, max={grad_max:.2e} (DYING!)")
            else:
                print(f"{name}: mean={grad_mean:.2e}, max={grad_max:.2e}")