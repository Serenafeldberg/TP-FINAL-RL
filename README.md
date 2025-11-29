# PPO Implementation for Flappy Bird 🐦

Implementación completa de **Proximal Policy Optimization (PPO)** desde cero aplicada a Flappy Bird, con experimentación sistemática en 21 configuraciones y análisis comparativo contra Stable Baselines 3.

**Autores**: Santiago Carrillo, Serena Feldberg, Agustina Videla Rivero  
**Institución**: Universidad de San Andrés  
**Curso**: Aprendizaje por Refuerzo - 2025

---

## 🎯 Descripción del Proyecto

Este proyecto implementa PPO (Proximal Policy Optimization) completamente desde cero, sin utilizar librerías de alto nivel como Stable Baselines 3, e incluye:

1. **Implementación de PPO con todas las optimizaciones estándar**:
   - Generalized Advantage Estimation (GAE)
   - Value Function Clipping
   - Normalización de observaciones (algoritmo de Welford)
   - Learning Rate Annealing
   - Entropy Regularization
   - Mini-batches y múltiples épocas

2. **Validación rigurosa** en probe environments (CartPole-v1, Acrobot-v1) comparando contra Stable Baselines 3

3. **Experimentación sistemática en Flappy Bird** con 21 configuraciones en 2 fases:
   - **Fase 1**: Exploración de 15 configuraciones base (1M timesteps c/u)
   - **Fase 2**: Optimización dirigida con 6 configuraciones extendidas (6-20M timesteps)

4. **Análisis de métricas duales**:
   - Training reward (lo que PPO optimiza)
   - Evaluation score (pipes atravesados, objetivo real del juego)

---

## ✨ Características Principales

### Implementación Técnica
- ✅ Actor-Crítico con arquitectura MLP configurable
- ✅ GAE-λ para reducción de varianza
- ✅ PPO-Clip para estabilidad de actualizaciones
- ✅ Value clipping para sincronización actor-crítico
- ✅ Normalización running con algoritmo de Welford (corregido)
- ✅ Learning rate decay lineal
- ✅ Entropy bonus adaptable

### Debugging y Validación
- 🐛 **Bug crítico identificado y corregido**: Learning rate decay (2000× más agresivo)
- 🐛 **Bug crítico identificado y corregido**: Algoritmo de Welford (varianza creciendo sin control)
- ✅ Validación completa en CartPole y Acrobot vs SB3

### Hallazgos Experimentales
- 🔴 **Hallazgo crítico**: Config 4 degrada -19.3% con 8× más entrenamiento (1M → 8M timesteps)
- 📊 Discrepancia training reward vs evaluation score (Config 4: rank 9 → rank 1)
- 🎯 Mejor configuración: **Config 4 con 219.40 pipes promedio** (1M timesteps)

---

## 📁 Estructura del Proyecto

```
TP-FINAL-RL/
├── src/                          # Código fuente principal
│   ├── ppoAgent/                # Implementación de PPO
│   │   ├── actorCritic.py      # Arquitectura Actor-Crítico
│   │   ├── memory.py           # Rollout Buffer y GAE
│   │   ├── ppo.py              # Algoritmo PPO-Clip
│   │   └── utils.py            # Normalización (Welford)
│   ├── envs/                    # Wrappers de ambientes
│   ├── mainTrainFlappy.py      # Script principal de entrenamiento
│   └── evaluate.py             # Evaluación determinística
│
├── scripts/                     # Scripts auxiliares organizados
│   ├── analysis/               # Análisis de resultados
│   ├── plotting/               # Generación de gráficos
│   └── training/               # Scripts de entrenamiento
│
├── data/                        # Datos y resultados
│   ├── probe_envs/             # CartPole y Acrobot
│   └── flappy/                 # CSVs de evaluación
│
├── plots/                       # Visualizaciones
│   ├── flappy_analysis/        # Gráficos de Flappy Bird
│   └── probe_envs/             # Gráficos de validación
│
├── savedModels/                 # Modelos entrenados (21 configs)
├── docs/                        # Documentación
├── tests_unit/                  # Tests unitarios
└── README.md                    # Este archivo
```

**Nota**: Cada subcarpeta contiene su propio README.md con detalles específicos.

---

## 🔧 Instalación

### Requisitos previos
- Python 3.8 o superior
- pip o uv (gestor de paquetes)

### Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## 🚀 Uso

### 1. Entrenar Custom PPO en Flappy Bird

```bash
cd src
python mainTrainFlappy.py --config_name "MiConfig" \
                          --hidden_size 512 \
                          --lr 1e-4 \
                          --n_epochs 4 \
                          --batch_size 64 \
                          --entropy_coef 0.01 \
                          --total_timesteps 1000000
```

### 2. Evaluar un modelo entrenado

```bash
cd src
python evaluate.py --model_path "../savedModels/config_4_Red_Grande/ppo_flappy_Red_Grande_final.pth" \
                   --n_episodes 10
```

### 3. Analizar resultados experimentales

```bash
# Ranking por training reward
python scripts/analysis/analyze_flappy_results.py

# Evaluación determinística de todos los modelos
python scripts/analysis/evaluate_top_models.py

# Generar todas las figuras del informe
python scripts/plotting/generate_report_figures.py
```

---

## 🏆 Resultados Destacados

### Validación en Probe Environments

| Ambiente | Custom PPO | SB3 PPO | Threshold | Status |
|----------|------------|---------|-----------|--------|
| **CartPole-v1** | 475.59 ± 73.26 (ep. 605) | 500.00 (ep. 408) | 475 | ✅ Resuelto |
| **Acrobot-v1** | -99.08 ± 26.97 (ep. 1342) | -85.26 ± 44.03 (ep. 192) | -100 | ✅ Resuelto |

---

### Flappy Bird: Top-5 Configuraciones por Evaluation Score 🎯

| Rank | Config ID | Name | Mean Score (Pipes) | Timesteps |
|------|-----------|------|-------------------|-----------|
| **1** 🏆 | **4** | **Red_Grande** | **219.40** | **1M** |
| 2 | 20 | Red_Grande | 177.10 | 8M |
| 3 | 17 | Red_Grande_Menos_Epocas_LR_Suave | 162.00 | 6M |
| 4 | 12 | Red_Grande_Menos_Epocas_Batch_Grande | 147.10 | 1M |
| 5 | 18 | Red_Gigante_Una_Epoca | 139.10 | 6M |

---

### 🔴 Hallazgo Crítico: Degradación con Más Entrenamiento

**Config 4** (1M timesteps): **219.40 pipes** (Rank 1)  
**Config 20** (8M timesteps): **177.10 pipes** (Rank 2)  
**Degradación**: **-42.3 pipes (-19.3%)**

**Implicación**: Más entrenamiento NO garantiza mejor performance. Early stopping basado en evaluation score es esencial.

---

### Discrepancia Training vs Evaluation

| Config | Rank Training | Rank Evaluation | Discrepancia |
|--------|---------------|-----------------|--------------|
| Config 4 | 9 ⬇️ | 1 ⬆️ | **+8 posiciones** |
| Config 14 | 1 ⬆️ | 3 ⬇️ | -2 posiciones |

**Conclusión**: Confiar únicamente en training reward habría descartado la mejor configuración.

---

## 🐛 Bugs Críticos Identificados y Corregidos

### Bug 1: Learning Rate Decay
**Problema**: Pasaba número de updates en lugar de timesteps acumulados  
**Impacto**: LR decay 2000× más agresivo, convergencia lenta  
**Corrección**: Mantener contador acumulativo de timesteps reales

### Bug 2: Algoritmo de Welford (Normalización)
**Problema**: Acumulaba sobre varianza sin mantener correctamente M₂  
**Impacto**: CartPole 200 → 446 reward; en Flappy habría impedido aprendizaje  
**Corrección**: Implementar correctamente transformación var ↔ M₂

---

## 🎓 Lecciones Aprendidas

1. **Validación incremental es esencial**: Los bugs se detectaron en probe environments antes de contaminar Flappy Bird
2. **Evaluación dual crítica**: Training reward ≠ objetivo real del juego
3. **Convergencia temprana**: La mejor config (219 pipes) se logró con 1M timesteps, no con 8M
4. **Batch size importa**: Batch pequeño (64) + epochs altos (4) extrae mejor señal de eventos raros

---

## 🔮 Trabajo Futuro

- [ ] Implementar vectorized environments (SubprocVecEnv)
- [ ] Early stopping automático basado en evaluation score
- [ ] Experimentar con reward shaping optimizado
- [ ] Probar arquitecturas recurrentes (LSTM/GRU)
- [ ] Curriculum learning
- [ ] Meta-optimización con Bayesian optimization

---

## 👥 Contacto

- **Santiago Carrillo**: scarrillo@udesa.edu.ar
- **Serena Feldberg**: sfeldberg@udesa.edu.ar
- **Agustina Videla Rivero**: avidelarivero@udesa.edu.ar

**Universidad de San Andrés** - Buenos Aires, Argentina - 2025
