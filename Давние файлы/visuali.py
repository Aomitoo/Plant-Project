import pandas as pd
import matplotlib.pyplot as plt
import os

def visualize_metrics(csv_path, output_path="training_metrics.png"):
    # Загружаем данные из CSV
    df = pd.read_csv(csv_path)
    
    # Создаем фигуру с тремя подграфиками
    plt.figure(figsize=(15, 5))
    
    # Подграфик 1: Loss
    plt.subplot(1, 3, 1)
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss', alpha=0.5)
    plt.plot(df['epoch'], df['val_loss'], label='Validation Loss', marker='o')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # Подграфик 2: Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(df['epoch'], df['val_acc'], label='Validation Accuracy', marker='o')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.ylim(0, 1)  # Устанавливаем пределы для процента (0-1)
    
    # Подграфик 3: F1-Score
    plt.subplot(1, 3, 3)
    plt.plot(df['epoch'], df['val_f1'], label='Validation F1', marker='o')
    plt.title('F1-Score Curve')
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')
    plt.grid(True)
    plt.legend()
    plt.ylim(0, 1)  # Устанавливаем пределы для процента (0-1)
    
    # Настраиваем общий вид
    plt.tight_layout()
    
    # Сохраняем график
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"График сохранён как {output_path}")

# Пример использования
if __name__ == "__main__":
    csv_path = "epoch_metrics_target.csv"  # Укажи путь к твоему файлу
    visualize_metrics(csv_path)