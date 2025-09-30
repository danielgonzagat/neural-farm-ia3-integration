# 📊 Neural Farm Metrics - 17.5 MILHÕES DE LINHAS

## ✅ DADOS COMPLETOS DISPONÍVEIS!

O arquivo `metrics.jsonl` (9.5GB, 17,559,712 linhas) está **100% preservado** no GitHub Release!

### 📥 Como Baixar e Recompor

```bash
# 1. Baixar todos os chunks
gh release download v1.0-data --repo danielgonzagat/neural-farm-ia3-integration --pattern "metrics_chunk_*.gz"

# 2. Recompor o arquivo original
cat metrics_chunk_00.gz metrics_chunk_01.gz metrics_chunk_02.gz \
    metrics_chunk_03.gz metrics_chunk_04.gz metrics_chunk_05.gz \
    metrics_chunk_06.gz metrics_chunk_07.gz metrics_chunk_08.gz \
    metrics_chunk_09.gz metrics_chunk_10.gz metrics_chunk_11.gz \
    metrics_chunk_12.gz metrics_chunk_13.gz metrics_chunk_14.gz \
    metrics_chunk_15.gz metrics_chunk_16.gz metrics_chunk_17.gz \
    metrics_chunk_18.gz metrics_chunk_19.gz | gunzip > metrics.jsonl

# 3. Verificar
wc -l metrics.jsonl  # Deve mostrar: 17559712
du -h metrics.jsonl  # Deve mostrar: 9.5G
```

### 📦 Chunks Disponíveis

20 partes totais (500MB cada descomprimido):
- `metrics_chunk_00.gz` até `metrics_chunk_19.gz`
- **Total comprimido:** ~1.2GB
- **Total descomprimido:** 9.5GB

### 📊 Conteúdo

Cada linha contém métricas de um step da evolução:

```json
{
  "step": 42,
  "timestamp": "2025-09-30T15:55:04",
  "mean_out": 0.00017663965991232544,
  "brain": {
    "input": {
      "population": 32,
      "generation": 41,
      "total_births": 387,
      "total_deaths": 355,
      "avg_fitness": 26.59375,
      "max_fitness": 41.0
    },
    "hidden": {...},
    "output": {...}
  }
}
```

### 🔥 Estatísticas

- **Total de linhas:** 17,559,712
- **Período:** 49 dias de CPU time
- **Gerações:** 42
- **População:** 30-34 neurônios por camada
- **Fitness:** 0 → 42 (crescimento contínuo)
- **Nascimentos:** 361-387 por geração
- **Mortes:** 327-355 por geração

### 📈 Análise Rápida

```python
import json

with open('metrics.jsonl') as f:
    for i, line in enumerate(f):
        if i % 1000000 == 0:  # A cada 1 milhão
            data = json.loads(line)
            step = data['step']
            fitness = data['brain']['input']['avg_fitness']
            print(f"Step {step}: fitness={fitness:.2f}")
```

### 🎯 Valor Científico

Este dataset é **ÚNICO** porque:
- ✅ 49 dias de evolução contínua
- ✅ 17.5M datapoints de métricas
- ✅ Evidência de seleção darwiniana
- ✅ Crescimento de fitness documentado
- ✅ Impossível de reproduzir (tempo de CPU)

---

**🔗 Download:** https://github.com/danielgonzagat/neural-farm-ia3-integration/releases/tag/v1.0-data
