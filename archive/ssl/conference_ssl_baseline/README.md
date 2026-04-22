# Stage 4: SSL Pretraining (BigP3BCI) для конференции

## Overview
На данном этапе реализовано self-supervised предобучение encoder для задачи P300 BCI с целью последующего использования в downstream экспериментах (Scratch vs SSL + Fine-Tuning).

Предобучение выполняется через masked time-block reconstruction без использования меток. 

!! Предобучение не закончено: только 10 эпох !!

Encoder после SSL используется как инициализация для downstream классификации Target vs Non-Target.
________________________________________
## Dataset
Источник
BigP3BCI (PhysioNet v1.0.0)
Использованы исследования:
B, J, M, P, Q, R, S1, S2
•	169 субъектов
•	3302 EDF
________________________________________
## Preprocessing (фиксировано)
•	14 каналов:
Fz, Cz, Pz, Oz,
P3, P4, PO7, PO8,
F3, F4, C3, C4,
CP3, CP4
•	Частота дискретизации: 256 Hz
•	Фильтрация: 0.1–20 Hz
•	Epoch: 0–800 ms post-stimulus
•	Padding до 208 отсчётов
•	Формат данных:
(N, 14, 208)
Нормализация
Channel-wise z-score по SSL-dataset.
Масштабирование
Перед подачей в модель сигнал переводится в микровольты:
x = x * 1e6
Это необходимо для стабильности BatchNorm и корректной амплитуды реконструкции.
________________________________________
## Storage
Данные сохранены в формате Zarr:
X: (493290, 14, 208) float32
Для ускорения обучения использовался subset 100k эпох.
________________________________________
## SSL Objective
Masking strategy
•	Time-block masking
•	mask_ratio ≈ 0.5
•	contiguous временные блоки
•	одинаковая маска для всех каналов внутри эпохи
•	masking генерируется на лету в training loop
Loss
Masked L1 reconstruction loss:
L = |x_hat - x|  (только по masked точкам)
Loss считается исключительно на замаскированных временных позициях.
________________________________________
## Model
Backbone: UNet1D_Light
Encoder architecture (зафиксирована)
Блоки:
inc
down1
down2
down3
down4
Input:
(B, 14, 208)
Bottleneck:
(B, 512, 13)
Decoder используется только на этапе SSL.
Encoder полностью совместим с downstream UNet1DEncoder.
________________________________________
## Training Configuration
•	Optimizer: AdamW
•	Learning rate: 3e-4
•	Weight decay: 1e-4
•	Scheduler: CosineAnnealingLR
•	Mixed precision (AMP): enabled
•	Batch size: 64
•	Epochs: 10
•	Subset size: 100,000 epochs
________________________________________
## Training Outcome
•	Обучение стабильное
•	Reconstruction адекватна по масштабу
•	Нет глобального положительного смещения decoder
•	Bottleneck shape корректен: (B, 512, 13)
________________________________________
## Output
Сохранён чекпоинт:
unet_ssl_final.pt
Содержит веса всей модели (encoder + decoder).
Для downstream используется только encoder.

