"""
Classificador de Lesões de Pele (ISIC Dataset) com TensorFlow e Aceleração NVIDIA GPU
Detecta 7 tipos de lesões: MEL, NV, BCC, AKIEC, BKL, DF, VASC
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURAÇÃO GPU NVIDIA ====================
print("=" * 80)
print("Inicializando Aceleração NVIDIA CUDA")
print("=" * 80)

# Verificar disponibilidade de GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✓ {len(gpus)} GPU(s) detectada(s):")
    for gpu in gpus:
        print(f"  - {gpu}")
    
    # Configurar GPU para usar memória conforme necessário (não alocar tudo de uma vez)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    # Configurar para usar a primeira GPU
    tf.config.set_visible_devices(gpus[0], 'GPU')
    print(f"\n✓ GPU principal configurada: {gpus[0]}")
else:
    print("✗ Nenhuma GPU detectada. Usando CPU (mais lento)")
    print("  Certifique-se de que CUDA e cuDNN estão instalados")

print(f"\n✓ TensorFlow versão: {tf.__version__}")
print("=" * 80 + "\n")

# ==================== CONFIGURAÇÕES ====================
class Config:
    """Configurações do modelo e treinamento"""
    # Caminhos
    BASE_PATH = "/mnt/hdd1tb/documentos/ufsc/semestres/9/ine5443/skin"
    IMAGES_PATH = os.path.join(BASE_PATH, "images")
    MASKS_PATH = os.path.join(BASE_PATH, "masks")
    GROUNDTRUTH_PATH = os.path.join(BASE_PATH, "GroundTruth.csv")
    
    # Classes (7 tipos de lesão)
    CLASSES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    NUM_CLASSES = len(CLASSES)
    
    # Configurações de imagem
    IMG_SIZE = 224  # Tamanho padrão para redes pré-treinadas
    CHANNELS = 3  # RGB
    BATCH_SIZE = 32
    
    # Configurações de treinamento
    EPOCHS = 50
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.1
    
    # Seed para reprodutibilidade
    RANDOM_SEED = 42

# ==================== CARREGAMENTO E PREPARAÇÃO DE DADOS ====================

class DataLoader:
    """Carrega e prepara os dados para treinamento"""
    
    def __init__(self, config):
        self.config = config
        self.data = None
        self.images = []
        self.labels = []
        
    def load_groundtruth(self):
        """Carrega o arquivo GroundTruth.csv"""
        print("Carregando GroundTruth.csv...")
        df = pd.read_csv(self.config.GROUNDTRUTH_PATH)
        print(f"✓ {len(df)} entradas carregadas")
        print(f"  Colunas: {list(df.columns)}")
        return df
    
    def load_image_with_mask(self, image_id):
        """Carrega imagem e máscara correspondente"""
        img_path = os.path.join(self.config.IMAGES_PATH, f"{image_id}.jpg")
        mask_path = os.path.join(self.config.MASKS_PATH, f"{image_id}_segmentation.png")
        
        try:
            # Carregar imagem
            img = load_img(img_path, target_size=(self.config.IMG_SIZE, self.config.IMG_SIZE))
            img_array = img_to_array(img) / 255.0  # Normalizar para [0, 1]
            
            # Carregar máscara (se existir)
            if os.path.exists(mask_path):
                mask = load_img(mask_path, target_size=(self.config.IMG_SIZE, self.config.IMG_SIZE), color_mode='grayscale')
                mask_array = img_to_array(mask) / 255.0  # Normalizar para [0, 1]
                
                # Concatenar imagem RGB com máscara (4 canais)
                combined = np.concatenate([img_array, mask_array], axis=-1)
                return combined, True
            else:
                # Se não houver máscara, usar zeros no 4º canal
                zeros = np.zeros((self.config.IMG_SIZE, self.config.IMG_SIZE, 1))
                combined = np.concatenate([img_array, zeros], axis=-1)
                return combined, False
                
        except Exception as e:
            print(f"Erro ao carregar {image_id}: {str(e)}")
            return None, False
    
    def prepare_data(self):
        """Carrega e prepara todos os dados"""
        print("\n" + "=" * 80)
        print("Preparando Dataset")
        print("=" * 80)
        
        # Carregar ground truth
        df = self.load_groundtruth()
        
        # Extrair imagens e labels
        print("\nCarregando imagens e máscaras...")
        valid_images = []
        valid_labels = []
        images_count = 0
        masks_found = 0
        
        for idx, row in df.iterrows():
            image_id = row['image']
            
            # Carregar imagem e máscara
            img_data, mask_found = self.load_image_with_mask(image_id)
            
            if img_data is not None:
                valid_images.append(img_data)
                
                # Extrair labels
                labels = row[self.config.CLASSES].values.astype(np.float32)
                valid_labels.append(labels)
                
                images_count += 1
                if mask_found:
                    masks_found += 1
                
                if (images_count) % 500 == 0:
                    print(f"  {images_count} imagens carregadas...")
        
        print(f"\n✓ Dataset preparado:")
        print(f"  - Imagens carregadas: {images_count}")
        print(f"  - Máscaras encontradas: {masks_found} ({100*masks_found/images_count:.1f}%)")
        print(f"  - Canal de máscara: {'Incluído' if masks_found > 0 else 'Vazio'}")
        
        # Converter para arrays numpy
        X = np.array(valid_images, dtype=np.float32)
        y = np.array(valid_labels, dtype=np.float32)
        
        print(f"\n✓ Forma dos dados:")
        print(f"  - X (imagens): {X.shape}")
        print(f"  - y (labels): {y.shape}")
        
        # Distribuição de classes
        print(f"\n✓ Distribuição de classes:")
        for idx, class_name in enumerate(self.config.CLASSES):
            count = np.sum(y[:, idx])
            percentage = 100 * count / len(y)
            print(f"  - {class_name}: {int(count)} ({percentage:.1f}%)")
        
        return X, y

# ==================== CONSTRUÇÃO DO MODELO ====================

class SkinLesionModel:
    """Construi o modelo CNN para classificação de lesões"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.history = None
        
    def build_model(self):
        """Cria arquitetura do modelo CNN com 4 canais de entrada"""
        print("\n" + "=" * 80)
        print("Construindo Modelo CNN")
        print("=" * 80)
        
        # Input com 4 canais (RGB + Máscara)
        inputs = layers.Input(shape=(self.config.IMG_SIZE, self.config.IMG_SIZE, 4))
        
        # Bloco 1: Extração de features
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Bloco 2
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Bloco 3
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Bloco 4
        x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Global Average Pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Fully Connected layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        # Output: 7 neurônios com sigmoid para multi-label classification
        outputs = layers.Dense(self.config.NUM_CLASSES, activation='sigmoid')(x)
        
        # Compilar modelo
        self.model = models.Model(inputs=inputs, outputs=outputs)
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
            loss='binary_crossentropy',  # Multi-label classification
            metrics=['accuracy', 
                    keras.metrics.Precision(),
                    keras.metrics.Recall(),
                    keras.metrics.AUC()]
        )
        
        print(f"\n✓ Modelo construído com sucesso!")
        print(f"  - Arquitetura: CNN com 4 blocos convolucionais")
        print(f"  - Entrada: 4 canais (RGB + Máscara)")
        print(f"  - Saída: {self.config.NUM_CLASSES} classes (sigmoid - multi-label)")
        print(f"  - Loss: Binary Crossentropy")
        print(f"  - Otimizador: Adam (lr={self.config.LEARNING_RATE})")
        
        # Resumo do modelo
        print(f"\nResumo do Modelo:")
        self.model.summary()
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val):
        """Treina o modelo"""
        print("\n" + "=" * 80)
        print("Treinamento do Modelo")
        print("=" * 80)
        print(f"Dispositivo: {'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'}")
        print(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Callbacks
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        # Treinar
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.EPOCHS,
            batch_size=self.config.BATCH_SIZE,
            callbacks=[early_stop, reduce_lr, model_checkpoint],
            verbose=1
        )
        
        print("\n✓ Treinamento concluído!")
        return self.history

# ==================== FUNÇÃO PRINCIPAL ====================

def main():
    """Executa o pipeline completo"""
    
    # Configuração
    config = Config()
    
    # Carregar dados
    loader = DataLoader(config)
    X, y = loader.prepare_data()
    
    # Dividir em train/val/test
    print("\n" + "=" * 80)
    print("Dividindo Dataset")
    print("=" * 80)
    
    # Primeiro, separar teste (10%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=config.TEST_SPLIT,
        random_state=config.RANDOM_SEED
    )
    
    # Depois, dividir validação (20% do temp = ~18% total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=config.VALIDATION_SPLIT,
        random_state=config.RANDOM_SEED
    )
    
    print(f"✓ Dataset dividido:")
    print(f"  - Treino: {X_train.shape[0]} amostras ({100*X_train.shape[0]/len(X):.1f}%)")
    print(f"  - Validação: {X_val.shape[0]} amostras ({100*X_val.shape[0]/len(X):.1f}%)")
    print(f"  - Teste: {X_test.shape[0]} amostras ({100*X_test.shape[0]/len(X):.1f}%)")
    
    # Construir modelo
    model_builder = SkinLesionModel(config)
    model_builder.build_model()
    
    # Treinar modelo
    history = model_builder.train(X_train, y_train, X_val, y_val)
    
    # Avaliar no conjunto de teste
    print("\n" + "=" * 80)
    print("Avaliação no Conjunto de Teste")
    print("=" * 80)
    
    test_loss, test_acc, test_prec, test_recall, test_auc = model_builder.model.evaluate(
        X_test, y_test, verbose=0
    )
    
    print(f"\n✓ Métricas de Teste:")
    print(f"  - Loss: {test_loss:.4f}")
    print(f"  - Accuracy: {test_acc:.4f}")
    print(f"  - Precision: {test_prec:.4f}")
    print(f"  - Recall: {test_recall:.4f}")
    print(f"  - AUC: {test_auc:.4f}")
    
    # Previsões detalhadas por classe
    print(f"\n✓ Performance por Classe:")
    y_pred = model_builder.model.predict(X_test)
    
    for idx, class_name in enumerate(config.CLASSES):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        y_true_binary = y_test[:, idx]
        y_pred_binary = (y_pred[:, idx] > 0.5).astype(int)
        
        acc = accuracy_score(y_true_binary, y_pred_binary)
        prec = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        
        print(f"  {class_name}:")
        print(f"    - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    # Salvar modelo
    print("\n" + "=" * 80)
    print("Salvando Modelo")
    print("=" * 80)
    
    model_builder.model.save('skin_lesion_model.h5')
    model_builder.model.save('skin_lesion_model')  # Formato SavedModel
    
    print(f"✓ Modelo salvo como 'skin_lesion_model.h5' e 'skin_lesion_model/'")
    
    # Salvar configurações e histórico
    config_dict = {
        'classes': config.CLASSES,
        'img_size': config.IMG_SIZE,
        'batch_size': config.BATCH_SIZE,
        'epochs': config.EPOCHS,
        'learning_rate': config.LEARNING_RATE
    }
    
    with open('model_config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"✓ Configuração salva como 'model_config.json'")
    
    # Salvar histórico
    history_dict = {
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss'],
        'accuracy': history.history['accuracy'],
        'val_accuracy': history.history['val_accuracy']
    }
    
    with open('training_history.json', 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    print(f"✓ Histórico de treinamento salvo como 'training_history.json'")
    
    print("\n" + "=" * 80)
    print("Pipeline Concluído com Sucesso!")
    print("=" * 80)

if __name__ == "__main__":
    main()
