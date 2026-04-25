'''
Plot learning curves
'''

import matplotlib.pyplot as plt
import os

def plot_learning_curves(train_loss_history, train_acc_history, test_loss_history, test_acc_history, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label='Train Loss')
    plt.plot(range(1, len(test_loss_history) + 1), test_loss_history, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'loss_curve.png'))
    plt.close()

    # Accuracy curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_acc_history) + 1), train_acc_history, label='Train Accuracy')
    plt.plot(range(1, len(test_acc_history) + 1), test_acc_history, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'accuracy_curve.png'))
    plt.close()