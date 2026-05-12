# 🚀 Guia de Setup - Classificador de Lesões de Pele

Instruções passo a passo para configurar e executar o classificador neural de lesões de pele com TensorFlow e aceleração NVIDIA GPU.

---

## 📋 Índice

1. [Verificação Pré-Requisitos](#verificação-pré-requisitos)
2. [Instalação Rápida (Automática)](#instalação-rápida-automática)
3. [Instalação Manual](#instalação-manual)
4. [Validação da Instalação](#validação-da-instalação)
5. [Primeiro Treinamento](#primeiro-treinamento)
6. [Solução de Problemas](#solução-de-problemas)

---

## Verificação Pré-Requisitos

### 1. Verificar Python

```bash
python --version
# ou
python3 --version
```

**Requerido**: Python 3.8 ou superior

### 2. Verificar NVIDIA GPU

```bash
nvidia-smi
```

**Esperado**: Saída mostrando GPU NVIDIA e CUDA Compute Capability

```
Fri Jan 10 14:30:00 2025
+-------------------------------+
| NVIDIA-SMI 470.00 Driver Version: 470.00 |
| GPU  Name        Compute Capability        |
|   0  NVIDIA RTX 3090      8.6             |
+-------------------------------+
CUDA Version: 11.8
```

### 3. Verificar Espaço em Disco

```bash
# Verificar espaço livre
df -h

# Ir para diretório do projeto
cd /mnt/hdd1tb/documentos/ufsc/semestres/9/ine5443/skin
ls -la
```

**Requerido**: 
- ~10GB para modelo + ambiente virtual
- 500MB para dados

### 4. Verificar Dados

```bash
ls -lh
# Esperado:
# - GroundTruth.csv (CSV com labels)
# - images/ (pasta com ~3285 imagens JPG)
# - masks/ (pasta com ~2565 máscaras PNG)
```

---

## Instalação Rápida (Automática)

Para usuários em **Linux/Mac**, use o script de instalação automática:

```bash
cd /mnt/hdd1tb/documentos/ufsc/semestres/9/ine5443/skin

# Dar permissão de execução
chmod +x install.sh

# Executar script (instala tudo automaticamente)
./install.sh
```

Isso automáticamente:
✓ Cria ambiente virtual  
✓ Instala pip, setuptools, wheel  
✓ Instala TensorFlow com CUDA  
✓ Instala todas as dependências  
✓ Valida o ambiente  

**Fim após ~30-40 minutos** (dependendo da conexão)

---

## Instalação Manual

### Passo 1: Preparar Ambiente

```bash
# Navegar para diretório do projeto
cd /mnt/hdd1tb/documentos/ufsc/semestres/9/ine5443/skin

# Criar ambiente virtual
python3 -m venv venv

# Ativar ambiente virtual
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

**Verificação**: Você deve ver `(venv)` no prompt bash

### Passo 2: Atualizar Ferramentas Padrão

```bash
pip install --upgrade pip setuptools wheel
```

**Esperado**: Mensagem mostrando versões atualizadas de pip, setuptools, wheel

### Passo 3: Instalar TensorFlow com CUDA

```bash
# Instalação completa com suporte NVIDIA GPU
pip install tensorflow[and-cuda]==2.13.0
```

**Esperado**: Demora ~5-15 minutos (download ~2.5GB)

```
Successfully installed tensorflow-2.13.0 ...
```

### Passo 4: Instalar Dependências

```bash
# Instalar de requirements.txt
pip install -r requirements.txt
```

**Esperado**: Instalação de numpy, pandas, matplotlib, scikit-learn, etc.

### Passo 5: Validar Ambiente

```bash
python validate_environment.py
```

**Esperado**: 8/8 verificações passando ✓

```
RESUMO DA VALIDAÇÃO
================================================================================
Python                    ✓ PASS
TensorFlow                ✓ PASS
GPU                       ✓ PASS
Dependências              ✓ PASS
Dados                     ✓ PASS
Memória                   ✓ PASS
Carregamento de Dados     ✓ PASS
Criação de Modelo         ✓ PASS

================================================================================
Resultado: 8/8 verificações passaram
================================================================================

🎉 Ambiente pronto para treinamento!
```

---

## Validação da Instalação

### Quick Check - GPU

```bash
python -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print(f'{len(gpus)} GPU(s) detected')"
```

**Esperado**:
```
1 GPU(s) detected
```

### Full Validation

```bash
python validate_environment.py
```

### Teste de Carregamento de Dados

```python
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img

# Testar leitura CSV
df = pd.read_csv('GroundTruth.csv')
print(f"✓ CSV carregado: {len(df)} imagens")

# Testar carregamento de imagem
img = load_img('images/ISIC_0031116.jpg', target_size=(224, 224))
print(f"✓ Imagem carregada: {img.size}")
```

---

## Primeiro Treinamento

### Executar Treinamento Completo

```bash
# Ativar ambiente virtual (se não estiver ativado)
source venv/bin/activate

# Executar script principal
python skin_lesion_classifier.py
```

### O Que Esperar

```
================================================================================
Inicializando Aceleração NVIDIA CUDA
================================================================================
✓ 1 GPU(s) detectada(s):
  - PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')

================================================================================
Preparando Dataset
================================================================================

Carregando GroundTruth.csv...
✓ 2693 entradas carregadas
✓ 2525 imagens carregadas

[... Outputs de treinamento ...]

✓ Treinamento concluído!
```

### Duração Esperada

| GPU | Batch | Epochs | Tempo |
|-----|-------|--------|-------|
| RTX 3090 | 64 | 50 | 15-20 min |
| RTX 3080 | 32 | 50 | 25-30 min |
| RTX 2080 | 32 | 50 | 40-50 min |
| V100 | 64 | 50 | 10-15 min |
| CPU | 16 | 10 | 2-3 horas |

### Arquivos Gerados

Após conclusão, você terá:

```
skin_lesion_model.h5              # Modelo final (HDF5)
skin_lesion_model/                # Modelo (SavedModel format)
best_model.h5                     # Melhor peso durante treino
model_config.json                 # Configurações
training_history.json             # Loss/Accuracy por época
```

---

## Usando o Modelo Treinado

### Fazer Predição em Uma Imagem

```bash
source venv/bin/activate

python inference.py --image ./images/ISIC_0031116.jpg
```

### Fazer Predição em Lote

```bash
python inference.py --batch ./images \
                    --mask-dir ./masks \
                    --output predictions.json
```

### Visualizar Resultados

```bash
python visualization.py --history     # Gráficos de treinamento
python visualization.py --distribution # Dist. de classes
python visualization.py --predictions  # Análise de predições
python visualization.py --samples      # Amostras de imagens
python visualization.py --all          # Tudo junto
```

---

## Solução de Problemas

### Problema: GPU Não Detectada

**Síntoma**: `validate_environment.py` mostra "0 GPU(s) detected"

**Solução 1**: Verificar driver NVIDIA
```bash
nvidia-smi  # Deve mostrar a GPU
```

**Solução 2**: Reinstalar TensorFlow com CUDA
```bash
pip uninstall tensorflow
pip install tensorflow[and-cuda]==2.13.0
```

**Solução 3**: Verificar compatibilidade CUDA
```bash
python -c "import tensorflow as tf; print(tf.sysconfig.get_build_info()['cuda_version'])"
```

### Problema: Memory Error (`OOM`)

**Síntoma**: `tensorflow.python.framework.errors_impl.ResourceExhaustedError`

**Solução 1**: Reduzir batch size (em `skin_lesion_classifier.py`)
```python
BATCH_SIZE = 16  # Reduzir de 32
```

**Solução 2**: Reduzir tamanho da imagem
```python
IMG_SIZE = 192  # Reduzir de 224
```

### Problema: Modelo Muito Lento

**Síntoma**: Treinamento leva >30 minutos por época

**Solução 1**: Verificar se está usando GPU
```bash
nvidia-smi watch -n 1  # Ver uso de GPU em tempo real
```

**Solução 2**: Aumentar batch size
```python
BATCH_SIZE = 64  # Aumentar para aproveitar GPU
```

### Problema: Imagens Não Encontradas

**Síntoma**: `FileNotFoundError: No such file or directory`

**Solução**:
```bash
# Verificar estrutura de diretórios
ls -la images/ | head -5
ls -la masks/ | head -5
head -5 GroundTruth.csv

# Verificar permissões
chmod -R 755 images masks *.csv
```

### Problema: Pip Lento/Conexão Instável

**Solução 1**: Usar mirror de PyPI mais rápido
```bash
pip install -i https://pypi.tsinghua.edu.cn/simple -r requirements.txt
# ou
pip install -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt
```

**Solução 2**: Instalar sem cache
```bash
pip install --no-cache-dir -r requirements.txt
```

---

## Configurações Recomendadas por GPU

### NVIDIA RTX 4090 (24GB)
```python
BATCH_SIZE = 128
IMG_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 0.0005
```

### NVIDIA RTX 3090 (24GB)
```python
BATCH_SIZE = 64
IMG_SIZE = 224
EPOCHS = 50
LEARNING_RATE = 0.001
```

### NVIDIA RTX 3080 (10GB)
```python
BATCH_SIZE = 32
IMG_SIZE = 224
EPOCHS = 50
LEARNING_RATE = 0.001
```

### NVIDIA RTX 2080 (8GB)
```python
BATCH_SIZE = 16
IMG_SIZE = 192
EPOCHS = 30
LEARNING_RATE = 0.001
```

### CPU (sem GPU)
```python
BATCH_SIZE = 8
IMG_SIZE = 192
EPOCHS = 10
LEARNING_RATE = 0.001
```

---

## Próximos Passos

1. ✓ Completar instalação
2. ✓ Validar ambiente (`validate_environment.py`)
3. ✓ Executar primeiro treinamento (`skin_lesion_classifier.py`)
4. ✓ Fazer predições (`inference.py`)
5. ✓ Visualizar resultados (`visualization.py`)
6. ✓ Ajustar hiperparâmetros conforme necessário
7. ✓ Salvar modelo final e fazer deploy

---

## Recursos Adicionais

- [TensorFlow GPU Setup](https://www.tensorflow.org/install/gpu)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [ISIC Dataset](https://www.isic-archive.com/)
- [Keras Documentation](https://keras.io/)

---

## Suporte

Para dúvidas:

1. Verifique a seção "Solução de Problemas"
2. Execute `validate_environment.py`
3. Confirme que dados estão no local correto
4. Tente com configurações reduzidas (batch size menor)

---

**Última Atualização**: 2024  
**Versão TensorFlow**: 2.13.0  
**Python**: 3.8+  
**CUDA**: 11.8
