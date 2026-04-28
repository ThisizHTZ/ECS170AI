import matplotlib.pyplot as plt
import os

def plot_learning_curves(
    train_loss_history,
    train_acc_history,
    output_folder,
    dataset_name,
    experiment_name
):
    os.makedirs(output_folder, exist_ok=True)

    prefix = f"{dataset_name.lower()}_{experiment_name}"

    # Loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_loss_history) + 1), train_loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title(f'{dataset_name} - {experiment_name} - Loss')
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(os.path.join(output_folder, f'{prefix}_loss.png'))
    plt.close()

    # Accuracy curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_acc_history) + 1), train_acc_history)
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy')
    plt.title(f'{dataset_name} - {experiment_name} - Accuracy')
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(os.path.join(output_folder, f'{prefix}_acc.png'))
    plt.close()