"""
Script de Inferência - Classificação de Lesões de Pele
Usa o modelo treinado para fazer previsões em novas imagens
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import json
import argparse
from pathlib import Path

class SkinLesionInference:
    """Realiza inferência com o modelo treinado"""
    
    def __init__(self, model_path='skin_lesion_model.h5', config_path='model_config.json'):
        """
        Inicializa o modelo e configurações
        
        Args:
            model_path: Caminho para o modelo treinado
            config_path: Caminho para o arquivo de configuração
        """
        print("Carregando modelo...")
        
        # Carregar modelo
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            print(f"✓ Modelo carregado: {model_path}")
        else:
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
        
        # Carregar configuração
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            print(f"✓ Configuração carregada: {config_path}")
        else:
            # Usar configuração padrão
            self.config = {
                'classes': ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC'],
                'img_size': 224,
                'batch_size': 32
            }
            print(f"⚠ Usando configuração padrão (config_path não encontrado)")
        
        self.img_size = self.config.get('img_size', 224)
        self.classes = self.config.get('classes', ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC'])
        
        # Detectar GPU
        gpus = tf.config.list_physical_devices('GPU')
        device = "GPU" if gpus else "CPU"
        print(f"✓ Dispositivo: {device}\n")
    
    def preprocess_image(self, image_path, mask_path=None):
        """
        Pré-processa imagem e máscara
        
        Args:
            image_path: Caminho para a imagem
            mask_path: Caminho para a máscara (opcional)
            
        Returns:
            Array numpy com imagem + máscara (4 canais)
        """
        # Carregar imagem
        img = load_img(image_path, target_size=(self.img_size, self.img_size))
        img_array = img_to_array(img) / 255.0
        
        # Carregar máscara se fornecida
        if mask_path and os.path.exists(mask_path):
            mask = load_img(mask_path, target_size=(self.img_size, self.img_size), color_mode='grayscale')
            mask_array = img_to_array(mask) / 255.0
        else:
            # Usar zeros se máscara não disponível
            mask_array = np.zeros((self.img_size, self.img_size, 1))
        
        # Concatenar canais (RGB + Máscara)
        combined = np.concatenate([img_array, mask_array], axis=-1)
        
        return combined
    
    def predict(self, image_path, mask_path=None, threshold=0.5):
        """
        Realiza previsão para uma imagem
        
        Args:
            image_path: Caminho para a imagem
            mask_path: Caminho para a máscara (opcional)
            threshold: Limiar para classificação (padrão: 0.5)
            
        Returns:
            Dicionário com resultados da previsão
        """
        # Pré-processar imagem
        img_array = self.preprocess_image(image_path, mask_path)
        img_batch = np.expand_dims(img_array, axis=0)
        
        # Fazer previsão
        predictions = self.model.predict(img_batch, verbose=0)[0]
        
        # Processar resultados
        results = {}
        predictions_list = []
        
        for class_name, prob in zip(self.classes, predictions):
            is_positive = prob > threshold
            results[class_name] = {
                'probability': float(prob),
                'predicted': bool(is_positive)
            }
            predictions_list.append({
                'class': class_name,
                'probability': float(prob),
                'predicted': bool(is_positive)
            })
        
        # Ordenar por probabilidade
        predictions_list.sort(key=lambda x: x['probability'], reverse=True)
        
        return {
            'image': image_path,
            'has_mask': mask_path is not None,
            'predictions': results,
            'top_predictions': predictions_list,
            'threshold': threshold
        }
    
    def predict_batch(self, image_dir, mask_dir=None, threshold=0.5):
        """
        Realiza previsões em lote para múltiplas imagens
        
        Args:
            image_dir: Diretório com imagens JPG
            mask_dir: Diretório com máscaras PNG (opcional)
            threshold: Limiar para classificação
            
        Returns:
            Lista de resultados
        """
        results = []
        
        # Obter lista de imagens
        image_files = []
        for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG']:
            image_files.extend(Path(image_dir).glob(ext))
        
        image_files = sorted(image_files)
        
        print(f"Processando {len(image_files)} imagens...")
        
        for idx, img_path in enumerate(image_files):
            # Procurar máscara correspondente
            mask_path = None
            if mask_dir:
                image_id = img_path.stem
                mask_file = Path(mask_dir) / f"{image_id}_segmentation.png"
                if mask_file.exists():
                    mask_path = str(mask_file)
            
            # Fazer previsão
            result = self.predict(str(img_path), mask_path, threshold)
            results.append(result)
            
            if (idx + 1) % 100 == 0:
                print(f"  {idx + 1}/{len(image_files)} concluído")
        
        print(f"✓ Processamento de lote concluído!\n")
        
        return results
    
    def print_result(self, result):
        """Imprime resultado de previsão de forma formatada"""
        print(f"\n{'='*70}")
        print(f"Imagem: {Path(result['image']).name}")
        print(f"Máscara: {'Sim' if result['has_mask'] else 'Não'}")
        print(f"Limiar: {result['threshold']}")
        print(f"{'-'*70}")
        
        print(f"\nTop 5 Predições:")
        for idx, pred in enumerate(result['top_predictions'][:5]):
            bar = '█' * int(pred['probability'] * 50)
            status = '✓ SIM' if pred['predicted'] else '✗ NÃO'
            print(f"  {idx+1}. {pred['class']:8} [{bar:50}] {pred['probability']:.4f} ({status})")
        
        print(f"\nTodas as Classes:")
        for class_name in self.classes:
            prob = result['predictions'][class_name]['probability']
            predicted = result['predictions'][class_name]['predicted']
            status = '✓' if predicted else '✗'
            print(f"  {status} {class_name:8}: {prob:.4f}")
        
        print(f"{'='*70}")

def main():
    parser = argparse.ArgumentParser(
        description='Classificador de Lesões de Pele - Inferência'
    )
    parser.add_argument('--image', type=str, help='Caminho para uma imagem específica')
    parser.add_argument('--mask', type=str, help='Caminho para máscara (opcional)')
    parser.add_argument('--batch', type=str, help='Diretório com imagens para processamento em lote')
    parser.add_argument('--mask-dir', type=str, help='Diretório com máscaras (para lote)')
    parser.add_argument('--model', type=str, default='skin_lesion_model.h5', help='Caminho do modelo')
    parser.add_argument('--config', type=str, default='model_config.json', help='Caminho da configuração')
    parser.add_argument('--threshold', type=float, default=0.5, help='Limiar para classificação (0-1)')
    parser.add_argument('--output', type=str, help='Salvar resultados em arquivo JSON')
    
    args = parser.parse_args()
    
    # Inicializar inferência
    print("\n" + "="*70)
    print("Classificador de Lesões de Pele - Inferência")
    print("="*70 + "\n")
    
    inference = SkinLesionInference(args.model, args.config)
    
    # Previsão única
    if args.image:
        print(f"Processando imagem: {args.image}")
        result = inference.predict(args.image, args.mask, args.threshold)
        
        # Imprimir resultado
        print(f"\n{'='*70}")
        print(f"Resultado para: {Path(args.image).name}")
        print(f"{'='*70}")
        print(f"\nMáscara: {'Fornecida' if args.mask else 'Não fornecida'}")
        print(f"Limiar: {args.threshold}\n")
        
        print("Top 5 Predições:")
        for idx, pred in enumerate(result['top_predictions'][:5]):
            bar = '█' * int(pred['probability'] * 40)
            print(f"  {idx+1}. {pred['class']:10} {bar:40} {pred['probability']:.4f}")
        
        # Salvar resultado se solicitado
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n✓ Resultado salvo em: {args.output}")
    
    # Previsão em lote
    elif args.batch:
        results = inference.predict_batch(args.batch, args.mask_dir, args.threshold)
        
        # Salvar resultados
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✓ Resultados salvos em: {args.output}")
        else:
            # Imprimir estatísticas
            print(f"\n{'='*70}")
            print("Estatísticas dos Resultados")
            print(f"{'='*70}")
            
            # Contar predições por classe
            class_counts = {c: 0 for c in inference.classes}
            for result in results:
                for class_name in inference.classes:
                    if result['predictions'][class_name]['predicted']:
                        class_counts[class_name] += 1
            
            print("\nFrequência de Classificações Positivas:")
            for class_name in sorted(class_counts, key=lambda x: class_counts[x], reverse=True):
                count = class_counts[class_name]
                percentage = 100 * count / len(results)
                print(f"  {class_name}: {count:4d} ({percentage:5.1f}%)")
    
    else:
        print("Erro: Forneça --image ou --batch")
        parser.print_help()

if __name__ == "__main__":
    main()
