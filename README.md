# Classificador Neural de Lesões de Pele com TensorFlow e Aceleração NVIDIA

**Projeto**: Classificação de Lesões de Pele (ISIC Dataset)  
**Framework**: TensorFlow/Keras com Aceleração GPU NVIDIA  
**Linguagem**: Python 3.8+  
**Dataset**: 7 Classes de Lesões de Pele  

---

## 📋 Conteúdo

Este projeto implementa uma rede neural convolucional completa para classificação automática de lesões de pele usando a metodologia de aprendizado profundo com aceleração por GPU NVIDIA.

### Arquivos Inclusos

| Arquivo | Descrição |
|---------|-----------|
| `skin_lesion_classifier.py` | Script principal de treinamento do modelo |
| `inference.py` | Script para fazer predições com o modelo treinado |
| `visualization.py` | Análise e visualização de resultados |
| `README.md` | Este arquivo com documentação completa |

---

## 🚀 Requisitos

### Hardware
- **GPU NVIDIA**: CUDA Compute Capability ≥ 3.5
- **RAM**: Mínimo 16GB (32GB recomendado)
- **Espaço em Disco**: ~10GB para modelo + dados

### Software
```bash
# Python 3.8 ou superior
python --version

# Dependências principais:
tensorflow>=2.10.0  # Com suporte NVIDIA CUDA
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

### Configuração NVIDIA CUDA

```bash
# Verificar instalação de CUDA
nvcc --version

# Verificar cuDNN
ldconfig -p | grep cudnn

# Usar as versões compatíveis com TensorFlow 2.10+:
# - CUDA 11.8
# - cuDNN 8.6+
```

---

## 📦 Instalação

### 1. Clonar/Baixar o Projeto
```bash
cd /mnt/hdd1tb/documentos/ufsc/semestres/9/ine5443/skin
```

### 2. Criar Ambiente Virtual (Recomendado)

**Usando venv:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

**Usando conda:**
```bash
conda create -n skin_lesion python=3.10
conda activate skin_lesion
```

### 3. Instalar Dependências

```bash
pip install --upgrade pip setuptools wheel

# Instalar TensorFlow com suporte GPU
pip install tensorflow[and-cuda]==2.13.0

# Instalar dependências adicionais
pip install pandas scikit-learn matplotlib seaborn
```

### 4. Verificar Instalação

```bash
python -c "import tensorflow as tf; print(len(tf.config.list_physical_devices('GPU')), 'GPU(s) detectada(s)')"
```

---

## 📊 Estrutura dos Dados

### Diretório Esperado
```
/mnt/hdd1tb/documentos/ufsc/semestres/9/ine5443/skin/
├── GroundTruth.csv           # Labels em 7 classes (csv)
├── images/                   # 3285 imagens JPG
│   ├── ISIC_0031116.jpg
│   ├── ISIC_0031117.jpg
│   └── ... (até ISIC_0034320.jpg)
├── masks/                    # ~2565 máscaras de segmentação PNG
│   ├── ISIC_0031756_segmentation.png
│   └── ...
└── scripts/                  # Scripts adicionais
```

### Formato GroundTruth.csv
```csv
image,MEL,NV,BCC,AKIEC,BKL,DF,VASC
ISIC_0024306,0.0,1.0,0.0,0.0,0.0,0.0,0.0
ISIC_0024307,0.0,1.0,0.0,0.0,0.0,0.0,0.0
...
```

**Classes (7 tipos de lesão):**
- **MEL** - Melanoma
- **NV** - Nevo (Nevus)
- **BCC** - Carcinoma Basocelular
- **AKIEC** - Queratose Actínica
- **BKL** - Queratose Benigna
- **DF** - Dermatofibroma
- **VASC** - Lesão Vascular

---

## 🏃 Uso

### 1. Treinamento do Modelo

```bash
python skin_lesion_classifier.py
```

**O que acontece:**
1. ✓ Detecta GPU NVIDIA e configura CUDA
2. ✓ Carrega GroundTruth.csv
3. ✓ Carrega 3285 imagens + máscaras
4. ✓ Divide em train/val/test (70%/10%/10% + 10% test)
5. ✓ Constrói modelo CNN com 4 canais de entrada (RGB + máscara)
6. ✓ Treina por até 50 epochs com early stopping
7. ✓ Avalia no conjunto de teste
8. ✓ Salva modelo em `skin_lesion_model.h5` e `skin_lesion_model/`

**Saída Esperada:**
```
================================================================================
Inicializando Aceleração NVIDIA CUDA
================================================================================
✓ 1 GPU(s) detectada(s):
  - PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')

✓ TensorFlow versão: 2.13.0

================================================================================
Preparando Dataset
================================================================================

Carregando GroundTruth.csv...
✓ 2693 entradas carregadas
  Colunas: ['image', 'MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

Carregando imagens e máscaras...
✓ Dataset preparado:
  - Imagens carregadas: 2525
  - Máscaras encontradas: 2254 (89.3%)
  - Canal de máscara: Incluído

✓ Forma dos dados:
  - X (imagens): (2525, 224, 224, 4)
  - y (labels): (2525, 7)

[... Detalhes de treinamento ...]

Treinamento concluído!
```

### 2. Fazer Predições

#### Predição em Imagem Única

```bash
python inference.py --image /caminho/para/imagem.jpg --mask /caminho/para/mascara.png
```

Exemplo com imagens do dataset:
```bash
python inference.py --image /mnt/hdd1tb/documentos/ufsc/semestres/9/ine5443/skin/images/ISIC_0031116.jpg
```

**Saída:**
```
Resultado para: ISIC_0031116.jpg

Máscara: Não fornecida
Limiar: 0.5

Top 5 Predições:
  1. NV     ██████████████████████████████████████ 0.9234
  2. BKL    ████████████████████ 0.4521
  3. MEL    ██████████ 0.2145
  4. AKIEC  ███ 0.0892
  5. DF     ██ 0.0634
```

#### Predição em Lote

```bash
python inference.py --batch /mnt/hdd1tb/documentos/ufsc/semestres/9/ine5443/skin/images \
                    --mask-dir /mnt/hdd1tb/documentos/ufsc/semestres/9/ine5443/skin/masks \
                    --output predictions_results.json
```

**Opções:**
- `--threshold`: Limiar de classificação (padrão: 0.5)
- `--output`: Salvar resultados em JSON

### 3. Visualização e Análise

```bash
# Plotar histórico de treinamento
python visualization.py --history

# Plotar distribuição de classes
python visualization.py --distribution

# Analisar predições
python visualization.py --predictions

# Visualizar amostras de imagens
python visualization.py --samples

# Gerar todas as visualizações
python visualization.py --all
```

**Saídas geradas:**
- `training_history.png` - Gráficos de loss e acurácia
- `class_distribution.png` - Distribuição das 7 classes
- `prediction_analysis.png` - Análise das predições
- `sample_images.png` - Amostra de imagens com máscaras

---

## 🧠 Arquitetura do Modelo

### Entrada
- **Resolução**: 224×224 pixels
- **Canais**: 4 (RGB + máscara de segmentação)
- **Normalização**: 0-1 (dividido por 255)

### Arquitetura CNN
```
Input (224×224×4)
  ↓
[Conv2D(32) + BatchNorm + ReLU] × 2 → MaxPool + Dropout(0.25)
[Conv2D(64) + BatchNorm + ReLU] × 2 → MaxPool + Dropout(0.25)
[Conv2D(128) + BatchNorm + ReLU] × 2 → MaxPool + Dropout(0.25)
[Conv2D(256) + BatchNorm + ReLU] × 2 → MaxPool + Dropout(0.25)
  ↓
Global Average Pooling
  ↓
Dense(512) + BatchNorm + Dropout(0.5)
Dense(256) + BatchNorm + Dropout(0.5)
Dense(7, activation='sigmoid')  ← 7 classes
  ↓
Output: 7 probabilidades (uma por classe)
```

### Características
- **Type**: Fully Convolutional Neural Network (FCN)
- **Loss**: Binary Crossentropy (multi-label classification)
- **Optimizer**: Adam (lr=0.001)
- **Regularization**: Dropout + BatchNormalization
- **Callbacks**: Early Stopping + Learning Rate Reduction + Model Checkpoint

### Multi-Label vs Multi-Class
Este problema é **multi-label** (múltiplas saídas podem ser 1):
- Cada imagem pode representar múltiplos tipos de lesão
- Output: 7 neurônios com **sigmoid** (não softmax)
- Loss: **binary_crossentropy** (não categorical_crossentropy)

---

## 📈 Métricas de Avaliação

O modelo é avaliado usando:

1. **Acurácia**: % de predições corretas
2. **Precision**: De todos os positivos preditos, quantos são reais
3. **Recall**: De todos os positivos reais, quantos foram encontrados
4. **F1-Score**: Média harmônica entre precision e recall
5. **AUC-ROC**: Área sob a curva ROC (mede poder discriminativo)
6. **Curva Loss**: Monitora overfitting (val_loss vs train_loss)

---

## 🔧 Configurações Ajustáveis

Edite as configurações em `skin_lesion_classifier.py`:

```python
class Config:
    IMG_SIZE = 224          # Tamanho das imagens
    BATCH_SIZE = 32         # Tamanho do lote
    EPOCHS = 50             # Máximo de épocas
    LEARNING_RATE = 0.001   # Taxa de aprendizado
    VALIDATION_SPLIT = 0.2  # Proporção validação
    TEST_SPLIT = 0.1        # Proporção teste
```

### Recomendações de Ajuste

**Para RTX 3090/4090 (24GB VRAM):**
```python
BATCH_SIZE = 64
IMG_SIZE = 256
EPOCHS = 100
```

**Para GPU com 4-8GB VRAM:**
```python
BATCH_SIZE = 16
IMG_SIZE = 192
EPOCHS = 30
```

**Para melhor acurácia (mais lento):**
```python
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
EPOCHS = 100
```

---

## 💾 Dados de Saída

Após treinamento, os seguintes arquivos são gerados:

| Arquivo | Descrição |
|---------|-----------|
| `best_model.h5` | Melhor modelo durante treinamento |
| `skin_lesion_model.h5` | Modelo final (formato H5) |
| `skin_lesion_model/` | Modelo final (formato SavedModel) |
| `model_config.json` | Configurações do modelo |
| `training_history.json` | Histórico de loss/accuracy |
| `predictions_results.json` | Resultados de predições (lote) |

---

## 🐛 Troubleshooting

### GPU Não Detectada
```python
python -c "import tensorflow as tf; print(tf.sysconfig.get_build_info()['cuda_version'])"
python -c "import tensorflow as tf; print(tf.sysconfig.get_build_info()['cudnn_version'])"
```

**Solução**: Reinstalar TensorFlow com suporte CUDA:
```bash
pip install --upgrade tensorflow[and-cuda]
```

### Memory Error
Reduzir `BATCH_SIZE` e `IMG_SIZE` em `Config`.

### Modelo Muito Lento
- Verificar se está usando GPU: `nvidia-smi`
- Aumentar `BATCH_SIZE`
- Reduzir `IMG_SIZE`

### Acurácia Baixa
- Aumentar `EPOCHS`
- Reduzir `LEARNING_RATE`
- Adicionar mais augmentação de dados

---

## 📚 Referências

### Dataset (ISIC)
- **Datasource**: International Skin Imaging Collaboration
- **Classes**: 7 tipos de lesões de pele
- **Imagens**: 3285 imagens de alta resolução
- **Máscaras**: 2565 máscaras de segmentação manual

### Tecnologias Utilizadas
- TensorFlow/Keras - Deep Learning
- NVIDIA CUDA - GPU Acceleration
- scikit-learn - Métricas e Validação
- matplotlib/seaborn - Visualização

### Artigos Relacionados
- He, K., et al. (2015). Deep Residual Learning for Image Recognition
- Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks
- Zhao, H., et al. (2017). Pyramid Scene Parsing Network

---

## 👤 Autor

**Projeto**: INE5443 - UFSC  
**Data**: 2024

---

## 📝 Licença

Este projeto segue a licença associada ao dataset ISIC.

---

## 🤝 Suporte

Para dúvidas ou problemas:
1. Verifique a seção Troubleshooting
2. Valide a instalação de CUDA com `nvidia-smi`
3. Confirme que os dados estão no diretório correto
4. Tente com `BATCH_SIZE` menor em caso de memory errors

---

**Última Atualização**: 2024  
**Versão TensorFlow**: 2.13.0+  
**Versão Python**: 3.8+
