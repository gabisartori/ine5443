#!/bin/bash

# ============================================================================
# Script de Instalação - Classificador de Lesões de Pele
# Configuração automatizada do ambiente e dependências
# ============================================================================

set -e  # Sair em caso de erro

echo ""
echo "================================================================================"
echo "Instalação - Classificador de Lesões de Pele com TensorFlow + NVIDIA GPU"
echo "================================================================================"
echo ""

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Função para imprimir com cores
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

# Step 1: Verificar Python
echo "Step 1: Verificando Python..."
if ! command -v python3 &> /dev/null; then
    print_error "Python3 não encontrado. Instale Python 3.8 ou superior."
    exit 1
fi
py_version=$(python3 --version)
print_status "Encontrado: $py_version"

# Step 2: Verificar pip
echo ""
echo "Step 2: Verificando pip..."
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 não encontrado"
    exit 1
fi
print_status "pip3 encontrado"

# Step 3: Criar ambiente virtual
echo ""
echo "Step 3: Criando ambiente virtual..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_status "Ambiente virtual criado"
else
    print_warning "Ambiente virtual já existe"
fi

# Step 4: Ativar ambiente virtual
echo ""
echo "Step 4: Ativando ambiente virtual..."
source venv/bin/activate
print_status "Ambiente ativado"

# Step 5: Fazer upgrade de pip
echo ""
echo "Step 5: Atualizando pip, setuptools e wheel..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
print_status "pip, setuptools, wheel atualizados"

# Step 6: Instalar dependências
echo ""
echo "Step 6: Instalando dependências..."
echo "  Isto pode levar alguns minutos, especialmente TensorFlow+CUDA..."

if pip install -r requirements.txt; then
    print_status "Dependências instaladas com sucesso"
else
    print_error "Erro ao instalar dependências"
    exit 1
fi

# Step 7: Validar instalação
echo ""
echo "Step 7: Validando instalação..."
python3 validate_environment.py

echo ""
echo "================================================================================"
echo "Instalação Concluída!"
echo "================================================================================"
echo ""
echo "Para executar o treinamento:"
echo "  1. source venv/bin/activate"
echo "  2. python skin_lesion_classifier.py"
echo ""
echo "Para fazer predições:"
echo "  1. source venv/bin/activate"
echo "  2. python inference.py --image <caminho_da_imagem>"
echo ""
echo "Para visualizar resultados:"
echo "  1. source venv/bin/activate"
echo "  2. python visualization.py --all"
echo ""
