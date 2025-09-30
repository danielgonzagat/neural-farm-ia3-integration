# 📊 Dados Completos do Neural Farm

## ⚠️ Arquivos Grandes (11GB total)

Os dados completos estão no servidor devido ao tamanho:

### Arquivos Principais:
- `metrics.jsonl` - 9.3GB (17,302,485 linhas de métricas)
- `neural_farm.db` - 1.4GB (database SQLite)
- `checkpoint_step_100.json` - 735 bytes

### 📦 Amostra Incluída:
- `neural_farm_sample_1000.jsonl` - Primeiras e últimas 1000 linhas (para visualização)
- `checkpoint_step_100.json` - Checkpoint completo

## 🔧 Como Obter os Dados Completos

### Opção 1: Download direto do servidor
```bash
# Contate o mantenedor para acesso ao servidor
rsync -avz --progress root@SERVER:/root/neural_farm_prod/ ./
```

### Opção 2: Processar a amostra
```bash
python3 -c "
import json
with open('neural_farm_sample_1000.jsonl') as f:
    for line in f:
        data = json.loads(line)
        print(f'Step {data[\"step\"]}: fitness={data[\"brain\"][\"input\"][\"avg_fitness\"]:.2f}')
"
```

## 📈 Estatísticas

- **Total de métricas:** 17,302,485 linhas
- **Tamanho comprimido:** ~1.8GB (gzip)
- **Tamanho descomprimido:** 9.3GB
- **Período:** 2+ dias contínuos
- **CPU time:** 49 dias acumulados
- **Gerações:** 42
