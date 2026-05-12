# ============================================================================
# Makefile - Classificador de Lesões de Pele
# Conveniência para executar comandos comuns
# ============================================================================

.PHONY: help install install-gpu validate train predict visualize clean

# Cores
BLUE=\033[0;34m
GREEN=\033[0;32m
NC=\033[0m # No Color

help:
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║     Classificador de Lesões de Pele - Makefile Auxiliar              ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(GREEN)Comandos disponíveis:$(NC)"
	@echo ""
	@echo "  setup               - Executar setup automático (install + validate)"
	@echo "  install             - Criar venv e instalar dependências"
	@echo "  validate            - Validar ambiente e dependências"
	@echo "  train               - Treinar modelo neural"
	@echo "  train-quick         - Treinar com configuração reduzida (teste rápido)"
	@echo "  predict             - Fazer predição em imagem específica"
	@echo "  predict-batch       - Fazer predição em lote"
	@echo "  visualize           - Gerar visualizações (todos os gráficos)"
	@echo "  visualize-history   - Gráficos de histórico de treinamento"
	@echo "  visualize-dist      - Distribuição de classes"
	@echo "  clean               - Remover arquivos gerados"
	@echo "  clean-all           - Remover tudo + venv"
	@echo "  show-config         - Mostrar configurações atuais"
	@echo ""

setup: install validate
	@echo "$(GREEN)✓ Setup automático concluído!$(NC)"
	@echo ""
	@echo "$(BLUE)Próximo passo:$(NC)"
	@echo "  source venv/bin/activate"
	@echo "  make train"
	@echo ""

install:
	@if [ ! -d "venv" ]; then \
		echo "$(BLUE)Criando ambiente virtual...$(NC)"; \
		python3 -m venv venv; \
	fi
	@echo "$(BLUE)Ativando ambiente virtual...$(NC)"
	@. venv/bin/activate && \
		pip install --upgrade pip setuptools wheel && \
		pip install -r requirements.txt
	@echo "$(GREEN)✓ Instalação concluída!$(NC)"

validate:
	@echo "$(BLUE)Validando ambiente...$(NC)"
	@. venv/bin/activate && python validate_environment.py

train:
	@echo "$(BLUE)Iniciando treinamento do modelo...$(NC)"
	@. venv/bin/activate && python skin_lesion_classifier.py

train-quick:
	@echo "$(BLUE)Treinamento rápido (teste)...$(NC)"
	@echo "BATCH_SIZE = 16" > temp_config.py
	@echo "IMG_SIZE = 192" >> temp_config.py
	@echo "EPOCHS = 5" >> temp_config.py
	@. venv/bin/activate && python skin_lesion_classifier.py
	@rm -f temp_config.py

predict:
	@echo "$(BLUE)Uso: make predict IMAGE=/caminho/para/imagem.jpg [MASK=/caminho/para/mascara.png]$(NC)"
	@echo ""
	@echo "Exemplo:"
	@echo "  make predict IMAGE=./images/ISIC_0031116.jpg"
	@echo ""
	@if [ -z "$(IMAGE)" ]; then \
		echo "$(RED)Erro: IMAGE não foi fornecido$(NC)"; \
		exit 1; \
	fi
	@. venv/bin/activate && python inference.py --image $(IMAGE) $(if $(MASK),--mask $(MASK),)

predict-batch:
	@echo "$(BLUE)Predição em lote...$(NC)"
	@. venv/bin/activate && \
		python inference.py \
			--batch ./images \
			--mask-dir ./masks \
			--output predictions_results.json
	@echo "$(GREEN)✓ Predições salvas em predictions_results.json$(NC)"

visualize:
	@echo "$(BLUE)Gerando todas as visualizações...$(NC)"
	@. venv/bin/activate && python visualization.py --all
	@echo "$(GREEN)✓ Visualizações geradas:$(NC)"
	@echo "  - training_history.png"
	@echo "  - class_distribution.png"
	@echo "  - prediction_analysis.png"
	@echo "  - sample_images.png"

visualize-history:
	@echo "$(BLUE)Gerando gráficos de histórico...$(NC)"
	@. venv/bin/activate && python visualization.py --history

visualize-dist:
	@echo "$(BLUE)Gerando distribuição de classes...$(NC)"
	@. venv/bin/activate && python visualization.py --distribution

clean:
	@echo "$(BLUE)Removendo arquivos gerados...$(NC)"
	@rm -f *.png *.json *.h5
	@rm -rf __pycache__ .pytest_cache
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✓ Limpeza concluída!$(NC)"

clean-all: clean
	@echo "$(BLUE)Removendo ambiente virtual...$(NC)"
	@rm -rf venv
	@echo "$(GREEN)✓ Ambiente completamente removido!$(NC)"

show-config:
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║                    Configurações do Projeto                            ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(GREEN)Dados:$(NC)"
	@echo "  Base Path:      /mnt/hdd1tb/documentos/ufsc/semestres/9/ine5443/skin"
	@echo "  Images:         $(shell ls -1 ./images/*.jpg 2>/dev/null | wc -l) arquivos"
	@echo "  Masks:          $(shell ls -1 ./masks/*.png 2>/dev/null | wc -l) arquivos"
	@echo "  GroundTruth:    $(shell wc -l < GroundTruth.csv) linhas"
	@echo ""
	@echo "$(GREEN)Classes de Lesão:$(NC)"
	@echo "  MEL - Melanoma"
	@echo "  NV  - Nevo (Nevus)"
	@echo "  BCC - Carcinoma Basocelular"
	@echo "  AKIEC - Queratose Actínica"
	@echo "  BKL - Queratose Benigna"
	@echo "  DF  - Dermatofibroma"
	@echo "  VASC - Lesão Vascular"
	@echo ""
	@echo "$(GREEN)Arquivos do Projeto:$(NC)"
	@echo "  skin_lesion_classifier.py - Treinamento principal"
	@echo "  inference.py              - Predição/Inferência"
	@echo "  visualization.py          - Análise e Visualização"
	@echo "  validate_environment.py   - Validação"
	@echo ""

# ============================================================================
# ALIASES ÚTEIS
# ============================================================================

.DEFAULT_GOAL := help

# Aliases para comandos comuns
all: setup train visualize
t: train
v: validate
p: predict-batch
vis: visualize
c: clean

.PHONY: all t v p vis c
