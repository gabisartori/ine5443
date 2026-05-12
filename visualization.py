"""
Script de Visualização e Análise
Visualiza resultados de treinamento, predições e análise de features dos modelos
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import argparse

# Configurar styles
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 12)
plt.rcParams['font.size'] = 10

class VisualizationAnalysis:
    """Análise e visualização de resultados"""
    
    def __init__(self):
        self.classes = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    
    def plot_training_history(self, history_path='training_history.json'):
        """Plota histórico de treinamento"""
        
        if not Path(history_path).exists():
            print(f"Arquivo não encontrado: {history_path}")
            return
        
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(history['loss'], label='Training Loss', linewidth=2)
        axes[0, 0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss durante Treinamento')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[0, 1].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Acurácia durante Treinamento')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Train vs Val Loss
        epochs = len(history['loss'])
        axes[1, 0].plot(range(epochs), history['loss'], 'o--', label='Training', alpha=0.7)
        axes[1, 0].plot(range(epochs), history['val_loss'], 's--', label='Validation', alpha=0.7)
        axes[1, 0].fill_between(range(epochs), history['loss'], history['val_loss'], alpha=0.2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Diferença Training-Validation Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Overfitting Analysis
        gap = [abs(t - v) for t, v in zip(history['loss'], history['val_loss'])]
        axes[1, 1].bar(range(len(gap)), gap, color='coral', alpha=0.7)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('|Training Loss - Validation Loss|')
        axes[1, 1].set_title('Potencial Overfitting')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        print("✓ Gráfico salvo: training_history.png")
        plt.show()
    
    def plot_class_distribution(self, csv_path='/mnt/hdd1tb/documentos/ufsc/semestres/9/ine5443/skin/GroundTruth.csv'):
        """Plota distribuição das classes"""
        
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"Arquivo não encontrado: {csv_path}")
            return
        
        # Contar ocorrências de cada classe
        class_counts = {}
        for class_name in self.classes:
            if class_name in df.columns:
                class_counts[class_name] = df[class_name].sum()
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Barplot
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE']
        bars = axes[0].bar(class_counts.keys(), class_counts.values(), color=colors, alpha=0.8)
        axes[0].set_xlabel('Classe de Lesão')
        axes[0].set_ylabel('Número de Imagens')
        axes[0].set_title('Distribuição das Classes no Dataset')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Adicionar valores nas barras
        for bar in bars:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        colors_pie = colors
        wedges, texts, autotexts = axes[1].pie(
            class_counts.values(),
            labels=class_counts.keys(),
            autopct='%1.1f%%',
            colors=colors_pie,
            startangle=90,
            textprops={'fontsize': 10, 'weight': 'bold'}
        )
        axes[1].set_title('Proporção das Classes')
        
        plt.tight_layout()
        plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
        print("✓ Gráfico salvo: class_distribution.png")
        plt.show()
    
    def plot_prediction_results(self, results_path='prediction_results.json'):
        """Plota resultados de predições em lote"""
        
        if not Path(results_path).exists():
            print(f"Arquivo não encontrado: {results_path}")
            return
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        # Se é um resultado único, converter para lista
        if isinstance(results, dict):
            results = [results]
        
        # Contabilizar predições
        class_positive_counts = {c: 0 for c in self.classes}
        class_probabilities = {c: [] for c in self.classes}
        
        for result in results:
            predictions = result['predictions']
            for class_name in self.classes:
                if class_name in predictions:
                    prob = predictions[class_name]['probability']
                    class_probabilities[class_name].append(prob)
                    
                    if predictions[class_name]['predicted']:
                        class_positive_counts[class_name] += 1
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Contagem de predições positivas
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE']
        bars = axes[0, 0].bar(class_positive_counts.keys(), class_positive_counts.values(), color=colors, alpha=0.8)
        axes[0, 0].set_xlabel('Classe')
        axes[0, 0].set_ylabel('Quantidade')
        axes[0, 0].set_title(f'Predições Positivas ({len(results)} imagens)')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # Box plot de probabilidades
        data_for_box = [class_probabilities[c] for c in self.classes]
        bp = axes[0, 1].boxplot(data_for_box, labels=self.classes, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[0, 1].set_xlabel('Classe')
        axes[0, 1].set_ylabel('Probabilidade')
        axes[0, 1].set_title('Distribuição de Probabilidades por Classe')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Histogramas de probabilidades
        for idx, class_name in enumerate(self.classes):
            if class_probabilities[class_name]:
                axes[1, 0].hist(class_probabilities[class_name], alpha=0.6, label=class_name, bins=20)
        axes[1, 0].set_xlabel('Probabilidade')
        axes[1, 0].set_ylabel('Frequência')
        axes[1, 0].set_title('Distribuição de Probabilidades Previstas')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Taxa de positividade
        total = len(results)
        positivity_rates = {c: class_positive_counts[c] / total * 100 for c in self.classes}
        axes[1, 1].barh(list(positivity_rates.keys()), list(positivity_rates.values()), color=colors, alpha=0.8)
        axes[1, 1].set_xlabel('Taxa de Positividade (%)')
        axes[1, 1].set_title('Frequência Relativa de Predições Positivas')
        axes[1, 1].grid(True, alpha=0.3, axis='x')
        
        for idx, (class_name, rate) in enumerate(positivity_rates.items()):
            axes[1, 1].text(rate, idx, f'{rate:.1f}%', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('prediction_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Gráfico salvo: prediction_analysis.png")
        plt.show()
    
    def visualize_samples(self, image_dir, mask_dir=None, num_samples=12):
        """Visualiza amostras de imagens com máscaras"""
        
        image_path = Path(image_dir)
        images = sorted(image_path.glob('*.jpg'))[:num_samples]
        
        if not images:
            print(f"Nenhuma imagem encontrada em: {image_dir}")
            return
        
        rows = (num_samples + 3) // 4
        fig, axes = plt.subplots(rows, 4, figsize=(16, 4*rows))
        axes = axes.flatten()
        
        for idx, img_path in enumerate(images):
            img = load_img(img_path, target_size=(224, 224))
            img_array = np.array(img)
            
            axes[idx].imshow(img_array)
            axes[idx].set_title(f"{img_path.stem}")
            axes[idx].axis('off')
            
            # Carregar máscara se disponível
            if mask_dir:
                image_id = img_path.stem
                mask_path = Path(mask_dir) / f"{image_id}_segmentation.png"
                if mask_path.exists():
                    mask = load_img(mask_path, target_size=(224, 224), color_mode='grayscale')
                    mask_array = np.array(mask)
                    
                    # Criar visualização com máscara
                    overlay = img_array.copy().astype(float)
                    overlay[mask_array > 128] = overlay[mask_array > 128] * 0.5 + np.array([255, 0, 0]) * 0.5
                    
                    # Mostrar próximo subplot com máscara
                    if idx + 1 < len(axes):
                        axes[idx + 1].imshow(overlay.astype(np.uint8))
                        axes[idx + 1].set_title(f"{img_path.stem}\n(com máscara)")
                        axes[idx + 1].axis('off')
        
        # Remover axes vazios
        for ax in axes[len(images):]:
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('sample_images.png', dpi=300, bbox_inches='tight')
        print("✓ Gráfico salvo: sample_images.png")
        plt.show()

def main():
    parser = argparse.ArgumentParser(
        description='Visualização e Análise - Classificador de Lesões de Pele'
    )
    parser.add_argument('--history', action='store_true', help='Plotar histórico de treinamento')
    parser.add_argument('--distribution', action='store_true', help='Plotar distribuição de classes')
    parser.add_argument('--predictions', action='store_true', help='Plotar análise de predições')
    parser.add_argument('--samples', action='store_true', help='Visualizar amostras de imagens')
    parser.add_argument('--all', action='store_true', help='Gerar todas as visualizações')
    parser.add_argument('--history-file', type=str, default='training_history.json', help='Caminho do arquivo de histórico')
    parser.add_argument('--predictions-file', type=str, default='prediction_results.json', help='Caminho do arquivo de predições')
    parser.add_argument('--images-dir', type=str, default='/mnt/hdd1tb/documentos/ufsc/semestres/9/ine5443/skin/images', help='Diretório de imagens')
    parser.add_argument('--masks-dir', type=str, default='/mnt/hdd1tb/documentos/ufsc/semestres/9/ine5443/skin/masks', help='Diretório de máscaras')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("Análise e Visualização - Classificador de Lesões de Pele")
    print("="*70 + "\n")
    
    analyzer = VisualizationAnalysis()
    
    if args.all or args.history:
        print("Plotando histórico de treinamento...")
        analyzer.plot_training_history(args.history_file)
    
    if args.all or args.distribution:
        print("Plotando distribuição de classes...")
        analyzer.plot_class_distribution()
    
    if args.all or args.predictions:
        print("Plotando análise de predições...")
        analyzer.plot_prediction_results(args.predictions_file)
    
    if args.all or args.samples:
        print("Visualizando amostras de imagens...")
        mask_dir = args.masks_dir if Path(args.masks_dir).exists() else None
        analyzer.visualize_samples(args.images_dir, mask_dir, num_samples=12)
    
    if not any([args.all, args.history, args.distribution, args.predictions, args.samples]):
        print("Use --help para ver opções disponíveis")

if __name__ == "__main__":
    main()
