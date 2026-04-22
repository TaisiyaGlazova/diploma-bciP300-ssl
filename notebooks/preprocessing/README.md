# Preprocessing

В этой папке находятся ноутбуки для подготовки и анализа датасетов, используемых в работе.

## Задачи этапа

* анализ структуры датасетов
* формирование SSL- и downstream-выборок
* приведение данных к единому формату
* построение разбиений для экспериментов

---

## Датасеты

Используются:

* **BigP3BCI (PhysioNet)** — основной датасет
* **BCI Competition III Dataset II** — дополнительный датасет

---

## Основные этапы обработки

### 1. Анализ данных

* подсчёт количества записей на субъект
* формирование индексов записей

---

### 2. Preprocessing EEG

* выбор 14 каналов
* полосовая фильтрация (0.1–20 Hz)
* ресемплинг до 256 Hz
* эпохирование (0–800 ms)
* padding до фиксированной длины (208)

Формат данных:

```text
(N, 14, 208)
```

---

### 3. Формирование выборок

#### SSL

* формируется большая неразмеченная выборка
* используется для pretraining

#### Downstream

* формируются:

  * train (dev)
  * benchmark (test)

---

### 4. Calibration protocol

Для каждого субъекта:

* 70% → calibration pool
* 30% → test_rest

Формируются вложенные подвыборки:

```text
p ∈ {10, 20, 40, 60, 100}
calib_10 ⊂ calib_20 ⊂ ... ⊂ calib_100
```

---

### 5. Нормализация

* рассчитывается отдельно для каждого субъекта
* отдельно для каждого p
* используется только Calib_p

Применяется к:

```text
Calib_p + Test_rest
```

Без утечек данных.

---

## Структура ноутбуков

### Анализ данных

* `bigp3bci_dataset_analysis.ipynb`
  — исследование структуры датасета BigP3BCI (EDA)

* `bigp3bci_downstream_epoch_count_check.ipynb`
  — проверка количества целевых эпох в calibration-подвыборках
  (sanity-check для downstream экспериментов)

---

### Основные ноутбуки preprocessing

* `01_bigp3bci_ssl_preprocessing.ipynb`
  — формирование SSL-выборки

* `02_bigp3bci_downstream_preprocessing.ipynb`
  — подготовка downstream-выборок

* `03_bigp3bci_calibration_splits.ipynb`
  — построение calibration splits и статистик

* `04_bcicomp3_preprocessing.ipynb`
  — preprocessing BCI Competition III Dataset II

---

## Результаты

Подготовленные данные сохраняются в:

```text
data/processed/
```

включая:

* epochs
* splits
* stats
* SSL dataset

---

## Замечания

* все преобразования выполняются воспроизводимо
* разбиения выполняются на уровне субъектов
* утечки между train и test отсутствуют
