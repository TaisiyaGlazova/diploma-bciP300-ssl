# Stage 5 — Fine-Tuning SSL Encoder for P300 Classification

## Overview

Архивный эксперимент fine-tuning, выполненный для подачи тезисов на конференцию.
Содержит промежуточные результаты и сравнение стратегий.
Не является финальным экспериментом дипломной работы.

На данном этапе проводится downstream-оценка SSL-предобученного энкодера в задаче бинарной классификации P300:

**Target vs Non-Target**

Цель эксперимента — проверить гипотезу:

> Улучшает ли self-supervised предобучение качество классификации P300 по сравнению с обучением модели с нуля.

Сравниваются два подхода:

* **Scratch baseline** — обучение модели с нуля
* **SSL + Fine-tuning** — использование предобученного encoder

Эксперименты выполняются на **BCI Competition III Dataset II**.

---

## Model

### Encoder

Используется encoder из архитектуры **UNet1D**, предобученный на SSL-задаче.

* Вход: `(B, 14, 208)`
* Выход (bottleneck): `(B, 512, 13)`

---

### Classification Head

Минимальная классификационная голова:

```
Global Average Pooling (по времени)
→ Linear (512 → 2)
```

Итого:

```
(B, 512, 13)
→ (B, 512)
→ (B, 2)
```

Выбор простой головы позволяет оценивать качество именно представлений encoder.

---

## Dataset

**BCI Competition III Dataset II**

* `X`: `(N, 14, 208)`
* `y`: `(N,)`

Используются 14 каналов:
Fz, Cz, Pz, Oz, P3, P4, PO7, PO8, F3, F4, C3, C4, CP3, CP4

---

## Experimental Protocol

### Subject Split

* **subjA** — Dev (выбор стратегии)
* **subjB** — Test (финальная оценка)

---

### Calibration Protocol

Для каждого субъекта:

* 70% → `calibration_pool`
* 30% → `test_rest`

Формируются вложенные подвыборки:

```
p ∈ {10, 20, 40, 60, 100}
calib_10 ⊂ calib_20 ⊂ ... ⊂ calib_100
```

---

### Normalization

* считается отдельно для каждого субъекта
* отдельно для каждого уровня `p`
* используется только `Calib_p`

Применяется к:

```
Calib_p + Test_rest
```

Без утечек данных.

---

## Training

* Loss: **Weighted CrossEntropy** (учёт дисбаланса 1:5)
* Optimizer: **AdamW**

---

## Fine-Tuning Strategies

Стратегии подбираются на Dev-субъекте.

### 1. Full Fine-Tuning

Обучаются все параметры:

* encoder → trainable
* head → trainable

---

### 2. Low-LR Encoder

Разные learning rate:

* encoder: `1e-5`
* head: `1e-4`

---

### 3. Partial Fine-Tuning

Обучается только верх encoder:

* inc → frozen
* down1 → frozen
* down2 → frozen
* down3 → frozen
* down4 → trainable
* head → trainable

---

### 4. Warmup Fine-Tuning (основная стратегия)

Двухэтапное обучение:

**Stage 1 — Head Warmup**

* encoder → frozen
* head → trainable

**Stage 2 — Joint Fine-Tuning**

* encoder → trainable
* head → trainable
* lr:

  * encoder: `1e-5`
  * head: `1e-4`

---

## Metrics

* Основная: **ROC-AUC**
* Дополнительная: **F1-score**

Сохраняется лучший результат по ROC-AUC.

---

## Evaluation

Сравнение проводится для:

```
p ∈ {10, 20, 40, 60, 100}
```

Метрика:

```
ΔAUC = AUC_SSL − AUC_scratch
```

---

## Outputs

Для каждого эксперимента сохраняются:

### Tables

* `results.csv`

  * subject
  * p
  * scratch_auc
  * ssl_auc
  * delta_auc

---

### Figures

* AUC vs calibration size
* ΔAUC vs p
* Dev vs Test сравнение

---

## Reproducibility

* seed = 42
* эксперименты выполняются отдельно по субъектам

---

## Conclusion

Данный этап позволяет оценить влияние SSL-предобучения на снижение объёма индивидуальной калибровки в задаче P300 BCI.
