"""
Script de Teste e Validação
Valida que o ambiente está configurado corretamente para treinamento
"""

import sys
import os

def check_python_version():
    """Verifica versão do Python"""
    print("\n" + "="*70)
    print("1. Verificando Versão do Python")
    print("="*70)
    
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("✓ Versão adequada (3.8+)\n")
        return True
    else:
        print("✗ Versão inadequada (requerido 3.8+)\n")
        return False

def check_tensorflow():
    """Verifica instalação do TensorFlow"""
    print("="*70)
    print("2. Verificando TensorFlow")
    print("="*70)
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__} instalado")
        return True
    except ImportError:
        print("✗ TensorFlow não instalado")
        print("  Instalar: pip install tensorflow[and-cuda]")
        return False

def check_gpu():
    """Verifica disponibilidade de GPU NVIDIA"""
    print("\n" + "="*70)
    print("3. Verificando NVIDIA GPU e CUDA")
    print("="*70)
    
    try:
        import tensorflow as tf
        
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print(f"✓ {len(gpus)} GPU(s) detectada(s):")
            for gpu in gpus:
                print(f"  - {gpu}")
            
            # Obter detalhes de build
            cuda_version = tf.sysconfig.get_build_info()['cuda_version']
            cudnn_version = tf.sysconfig.get_build_info()['cudnn_version']
            
            print(f"\n✓ CUDA {cuda_version}")
            print(f"✓ cuDNN {cudnn_version}")
            
            return True
        else:
            print("⚠ Nenhuma GPU detectada (CPU será usado)")
            print("  Requisitos para GPU NVIDIA:")
            print("  - NVIDIA GPU com Compute Capability ≥ 3.5")
            print("  - CUDA 11.8+")
            print("  - cuDNN 8.6+")
            return False
            
    except Exception as e:
        print(f"✗ Erro ao verificar GPU: {e}")
        return False

def check_dependencies():
    """Verifica dependências Python"""
    print("\n" + "="*70)
    print("4. Verificando Dependências Python")
    print("="*70)
    
    dependencies = {
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'sklearn': 'scikit-learn',
        'PIL': 'Pillow',
    }
    
    all_installed = True
    
    for module_name, display_name in dependencies.items():
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'desconhecida')
            print(f"✓ {display_name:20} {version}")
        except ImportError:
            print(f"✗ {display_name:20} NÃO INSTALADO")
            all_installed = False
    
    return all_installed

def check_data():
    """Verifica disponibilidade dos dados"""
    print("\n" + "="*70)
    print("5. Verificando Disponibilidade dos Dados")
    print("="*70)
    
    base_path = "/mnt/hdd1tb/documentos/ufsc/semestres/9/ine5443/skin"
    
    paths = {
        'GroundTruth.csv': os.path.join(base_path, 'GroundTruth.csv'),
        'images/': os.path.join(base_path, 'images'),
        'masks/': os.path.join(base_path, 'masks'),
    }
    
    all_exist = True
    
    for name, path in paths.items():
        if os.path.exists(path):
            if os.path.isdir(path):
                file_count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
                print(f"✓ {name:20} ({file_count} arquivos)")
            else:
                file_size = os.path.getsize(path) / (1024*1024)
                print(f"✓ {name:20} ({file_size:.2f} MB)")
        else:
            print(f"✗ {name:20} NÃO ENCONTRADO")
            all_exist = False
    
    if not all_exist:
        print(f"\n  Esperado em: {base_path}")
    
    return all_exist

def check_memory():
    """Verifica memória disponível"""
    print("\n" + "="*70)
    print("6. Verificando Memória do Sistema")
    print("="*70)
    
    try:
        import psutil
        
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        available_gb = mem.available / (1024**3)
        percent = mem.percent
        
        print(f"✓ Memória Total:      {total_gb:.1f} GB")
        print(f"✓ Memória Disponível: {available_gb:.1f} GB")
        print(f"✓ Uso:                {percent}%")
        
        if total_gb >= 16:
            print("✓ Memória adequada para treinamento")
            return True
        else:
            print("⚠ Memória limitada (recomendado 16+ GB)")
            return False
            
    except ImportError:
        print("⚠ psutil não instalado (pulando verificação)")
        print("  Instalar: pip install psutil")
        return True

def test_data_loading():
    """Testa carregamento de dados"""
    print("\n" + "="*70)
    print("7. Testando Carregamento de Dados")
    print("="*70)
    
    try:
        import pandas as pd
        import numpy as np
        from tensorflow.keras.preprocessing.image import load_img
        
        # Ler CSV
        csv_path = "/mnt/hdd1tb/documentos/ufsc/semestres/9/ine5443/skin/GroundTruth.csv"
        df = pd.read_csv(csv_path)
        print(f"✓ GroundTruth.csv carregado ({len(df)} linhas)")
        
        # Testar carregamento de imagem
        img_id = df.iloc[0]['image']
        img_path = f"/mnt/hdd1tb/documentos/ufsc/semestres/9/ine5443/skin/images/{img_id}.jpg"
        
        if os.path.exists(img_path):
            img = load_img(img_path, target_size=(224, 224))
            print(f"✓ Imagem {img_id} carregada com sucesso")
        else:
            print(f"⚠ Imagem {img_id} não encontrada")
            return False
        
        # Testar carregamento de máscara
        mask_path = f"/mnt/hdd1tb/documentos/ufsc/semestres/9/ine5443/skin/masks/{img_id}_segmentation.png"
        if os.path.exists(mask_path):
            mask = load_img(mask_path, target_size=(224, 224), color_mode='grayscale')
            print(f"✓ Máscara {img_id}_segmentation.png carregada com sucesso")
        else:
            print(f"⚠ Máscara não encontrada (será usando zeros)")
        
        return True
        
    except Exception as e:
        print(f"✗ Erro ao testar carregamento: {e}")
        return False

def test_model_creation():
    """Testa criação de modelo"""
    print("\n" + "="*70)
    print("8. Testando Criação de Modelo")
    print("="*70)
    
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
        
        # Criar modelo pequeno de teste
        inputs = layers.Input(shape=(224, 224, 4))
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(7, activation='sigmoid')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("✓ Modelo criado com sucesso")
        print(f"  - Input: (224, 224, 4)")
        print(f"  - Output: 7 classes")
        print(f"  - Total de parâmetros: {model.count_params():,}")
        
        return True
        
    except Exception as e:
        print(f"✗ Erro ao criar modelo: {e}")
        return False

def main():
    """Executa todas as verificações"""
    
    print("\n" + "="*70)
    print("VALIDAÇÃO DO AMBIENTE")
    print("Classificador de Lesões de Pele com TensorFlow + NVIDIA GPU")
    print("="*70)
    
    results = {
        'Python': check_python_version(),
        'TensorFlow': check_tensorflow(),
        'GPU': check_gpu(),
        'Dependências': check_dependencies(),
        'Dados': check_data(),
        'Memória': check_memory(),
        'Carregamento de Dados': test_data_loading(),
        'Criação de Modelo': test_model_creation(),
    }
    
    # Resumo
    print("\n" + "="*70)
    print("RESUMO DA VALIDAÇÃO")
    print("="*70)
    
    for check_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{check_name:25} {status}")
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    print(f"\n{'='*70}")
    print(f"Resultado: {passed}/{total} verificações passaram")
    print(f"{'='*70}\n")
    
    if passed == total:
        print("🎉 Ambiente pronto para treinamento!")
        print("\nPróximo passo:")
        print("  python skin_lesion_classifier.py")
        return 0
    else:
        print("⚠️ Algumas verificações falharam.")
        print("   Resolva os problemas acima antes de treinar o modelo.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
