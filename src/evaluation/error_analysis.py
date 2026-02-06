import json
import matplotlib.pyplot as plt
import os

def load_history(history_path):
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"History file not found: {history_path}")

    with open(history_path, "r") as f:
        history = json.load(f)

    return history

def plot_learning_curves(history):
    train_loss = history.get("loss", [])
    val_loss = history.get("val_loss", [])
    train_acc = history.get("accuracy", []) or history.get("acc", [])
    val_acc = history.get("val_accuracy", []) or history.get("val_acc", [])

    epochs = range(1, len(train_loss) + 1)

    # Plot Loss
    plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Accuracy
    if train_acc and val_acc:
        plt.figure()
        plt.plot(epochs, train_acc, label="Train Accuracy")
        plt.plot(epochs, val_acc, label="Val Accuracy")
        plt.title("Training vs Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.show()

def diagnose(history):
    train_loss = history["loss"]
    val_loss = history["val_loss"]

    print("=== Diagnosis ===")

    if train_loss[-1] < val_loss[-1] and (val_loss[-1] - train_loss[-1]) > 0.1:
        print("Likely Overfitting: Training loss much lower than validation loss.")
        print("   Consider: more data, augmentation, regularization, or early stopping.")
    elif train_loss[-1] > 0.5 and val_loss[-1] > 0.5:
        print("Likely Underfitting: Both training and validation loss are high.")
        print("   Consider: bigger model, more epochs, better features.")
    else:
        print("Model seems reasonably well-fitted.")
        print("   Training and validation curves look balanced.")

def main():
    history_path = "models/trained/history.json"  # adjust if needed
    history = load_history(history_path)

    plot_learning_curves(history)
    diagnose(history)

if __name__ == "__main__":
    main()
