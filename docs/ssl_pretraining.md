# SSL Pretraining (Masked Reconstruction) for P300 EEG

## Overview

В рамках работы реализован этап self-supervised learning (SSL) для ЭЭГ-сигналов в задаче P300 Brain-Computer Interface (BCI).

Цель — предобучить encoder на неразмеченных ЭЭГ-данных с помощью задачи masked reconstruction и использовать его далее в downstream-заче классификации P300 при ограниченном объёме калибровочных данных.

---

## Objective

Обучить модель извлекать информативные представления ЭЭГ-сигнала без использования разметки.

Предобученный encoder используется далее для:

* fine-tuning (P300 classification)

---

## Dataset

**Источник:** BigP3BCI (PhysioNet)
**Использованные исследования:** B, J, M, P, Q, R, S1, S2

### Preprocessing

* Каналы (14): Fz, Cz, Pz, Oz, P3, P4, PO7, PO8, F3, F4, C3, C4, CP3, CP4
* Фильтрация: 0.1–20 Hz (FIR)
* Частота дискретизации: 256 Hz
* Эпохи: 0–800 ms
* Padding: до 208 отсчётов
* Формат: `(N, 14, 208)`, float32

---

## SSL Subset

Для ускорения обучения использована подвыборка ~100k эпох.

* Отбор: на уровне субъектов
* Split: train / validation (без утечек между субъектами)

Данные сохранены в `.npy` формате (\data\processed\bigp3bci\ssl\split\):

* `X_train.npy`
* `X_val.npy`

---

## Model

**Architecture:** UNet1D_Light

### Encoder

* inc → down1 → down2 → down3 → down4
* Вход: `(B, 14, 208)`
* Bottleneck: `(B, 512, 13)`

### Decoder

Используется только на этапе SSL и далее отбрасывается.

---

## Pretext Task

**Masked Time-Block Reconstruction**

* mask_ratio ≈ 0.5
* contiguous masking (block_size ≈ 16)
* маска применяется ко всем каналам

### Loss

**Masked L1 loss**

* считается только на замаскированных участках
* незамаскированные точки не участвуют в loss

---

## Training

* Optimizer: AdamW

  * lr = 3e-4
  * weight_decay = 1e-4
* Scheduler: CosineAnnealingLR
* Mixed precision (AMP)

### Training setup

* batch_size = 64
* max_epochs = 200

### Early stopping

* patience = 15
* min_delta = 1e-4

---

## Results

* обучение проходило стабильно
* train и validation loss сходятся
* наблюдается плато функции потерь

Обучение остановлено по early stopping на **105-й эпохе**.
Лучшая **90 эпоха**

### Reconstruction quality

* корректное восстановление masked областей
* возможны искажения вне masked областей (ожидаемо)

---

## Saved Artifacts

Результаты сохранены в `outputs/ssl/`:

### Checkpoints

* `encoder_best.pt` — основной encoder (для downstream)
* `encoder_last.pt`
* `checkpoint_best_full.pt` — encoder + decoder
* `checkpoint_last_full.pt`
* `encoder_epoch_*.pt` — промежуточные модели (каждые 10 эпох)

### Logs

* `loss_history.json`

### Figures

* loss curves
* reconstruction examples

---

## Notes

Модель оптимизируется только на masked участках, поэтому:

* точная реконструкция вне masked областей не гарантируется
* это не влияет на downstream-качество

---

## Next Step

Предобученный encoder используется для:

* SSL + Fine-Tuning
* сравнения со scratch baseline

---

## Conclusion

Этап SSL-предобучения успешно завершён.

Модель обучилась извлекать информативные представления ЭЭГ-сигнала, что позволяет перейти к исследованию влияния SSL на сокращение объёма индивидуальной калибровки в задаче P300 BCI.
