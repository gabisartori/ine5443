# Avaliação de Lesões de Pele com Critério ABC

Projeto para avaliação de imagens dermatológicas usando um critério heurístico ABC:

- A: Asymmetry (assimetria)
- B: Border irregularity (irregularidade de borda)
- C: Color variegation (variação de cor)

O repositório atual está focado principalmente no pipeline de avaliação ABC (geração de score e comparação contra GroundTruth).

## Arquivos do Repositório

| Arquivo | Descrição |
|---------|-----------|
| `GroundTruth.csv` | Rótulos reais com classes MEL, NV, BCC, AKIEC, BKL, DF, VASC |
| `test.csv` | Exemplo de saída de avaliação ABC (score e suspicious) |
| `inference.py` | Script de inferência com modelo TensorFlow (mantido para uso com modelo treinado) |
| `scripts/evaluate_abc.py` | Calcula A/B/C e gera CSV de resultados |
| `scripts/evaluate_test_accuracy.py` | Compara previsões binárias com o GroundTruth |
| `images/` | Imagens do dataset |
| `masks/` | Máscaras de segmentação |

## Requisitos

Python 3.8+ e dependências:

```bash
pip install numpy opencv-python
```

## Estrutura de Dados Esperada

```text
.
├── GroundTruth.csv
├── images/
├── masks/
└── scripts/
    ├── evaluate_abc.py
    └── evaluate_test_accuracy.py
```

## Fluxo Principal (ABC)

1. Gerar previsões ABC a partir de imagens e máscaras.
2. Comparar previsões com GroundTruth.
3. Ajustar classes positivas e limiar conforme objetivo clínico.

### 1) Gerar CSV com score ABC

```bash
python scripts/evaluate_abc.py \
  --images images \
  --masks masks \
  --output test.csv
```

Colunas geradas:

- image
- asymmetry
- border_irregularity
- color_variegation
- score
- suspicious

### 2) Avaliar contra GroundTruth (MEL + BCC como positivo)

```bash
python scripts/evaluate_test_accuracy.py \
  --ground-truth GroundTruth.csv \
  --predictions test.csv \
  --positive-classes MEL,BCC
```

Por padrão, `--positive-classes` considera `MEL,BCC`, ou seja, uma imagem é positiva se qualquer uma dessas classes for positiva no GroundTruth.

### 3) Ajustar definição de positivo (opcional)

Exemplo incluindo AKIEC:

```bash
python scripts/evaluate_test_accuracy.py \
  --ground-truth GroundTruth.csv \
  --predictions test.csv \
  --positive-classes MEL,BCC,AKIEC
```

### 4) Ajustar threshold de classificação (opcional)

Quando a coluna usada for numérica (ex.: `score`), `--threshold` controla o corte binário:

```bash
python scripts/evaluate_test_accuracy.py \
  --ground-truth GroundTruth.csv \
  --predictions test.csv \
  --positive-classes MEL,BCC \
  --threshold 0.5
```

## Observações Importantes

- O script de acurácia procura por `suspicious` primeiro e usa `score` como fallback.
- IDs de imagem são comparados por `stem` (ex.: `ISIC_XXXX`), então extensões diferentes (`.jpg`, `.png`) não quebram o matching.
- O critério ABC aqui é heurístico e não substitui diagnóstico médico.

## Próximos Passos Recomendados

1. Gerar visualizações de casos extremos (maior e menor score).
2. Avaliar sensibilidade/especificidade variando `--threshold`.
3. Validar máscaras problemáticas antes de comparar modelos.

## Licença

Consulte os arquivos de licença em `images/` e a licença do dataset ISIC.
