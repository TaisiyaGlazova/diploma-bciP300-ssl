# Data

В проекте используются несколько EEG-датасетов для задачи классификации P300.

Данные разделены на:
- `raw/` — исходные датасеты
- `processed/` — подготовленные выборки для обучения и оценки моделей

---

## Raw данные (`data/raw/`)

### BigP3BCI
Источник: https://physionet.org/content/bigp3bci/1.0.0/ 
Содержит исходные EDF-записи EEG.

Дополнительно:
- `metadata/recordings_per_subject.csv` — количество записей на каждого субъекта

### BCI Competition III Dataset II
Источник: https://www.bbci.de/competition/iii/download/index.html?agree=yes&submit=Submit 
Классический датасет для задачи P300.

Содержит:
- `.mat` файлы с EEG
- документацию и её перевод

---

## Processed данные (`data/processed/`)

### BigP3BCI

#### SSL-выборка (`ssl/`)
Используется для self-supervised обучения.

- `ssl_bigp3bci.zarr` — основной датасет
- `build_config.json` — параметры сборки
- `build_log.csv` — лог формирования выборки
- `records_index.csv` — индекс использованных записей
- `preprocessing_summary.txt` — описание пайплайна
- `splits/` — разбиения на val/test

#### Downstream-выборки (`downstream/`)

Используются для обучения и оценки моделей.

- `train/` — обучающая выборка
- `benchmark/` — независимая тестовая выборка

Каждая содержит:
- `epochs/` — EEG эпохи (формат: `(N, C, L)`)
- `splits/` — разбиения на calibration/test
- `stats/` — статистики нормализации

Дополнительно:
- `preprocessing_summary.txt` — параметры препроцессинга
- `records_index_full.csv` — индекс записей

---

### BCI Competition III Dataset II (`bcicomp3/`)

Подготовленные данные для downstream-задачи.

Содержит:
- `epochs/` — EEG эпохи
- `splits/` — разбиения по субъектам
- `stats/` — статистики нормализации
- `preprocessing_summary_v1.txt` — описание препроцессинга

---

## Препроцессинг

Для всех датасетов применяется единый пайплайн:

- выбор каналов (14 каналов)
- band-pass фильтрация (0.1–20 Hz)
- эпохирование (0–800 мс)
- padding до 208 отсчётов
- нормализация (z-score по каналам)
- формирование разбиений (calibration / test)

Для downstream-задачи:
- нормализация считается отдельно для каждого субъекта на его calibration subset

---

## Важно

- Данные BigP3BCI не включены в репозиторий из-за большого объёма
- Пути к этим данным указываются локально в ноутбуках/скриптах
- Для воспроизведения требуется загрузка датасетов отдельно