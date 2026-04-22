# Downstream notebooks

Основные ноутбуки Stage 5:

- `dev_tuning_small.ipynb` — быстрый dev-тест на малом числе субъектов
- `dev_tuning_full.ipynb` — полный dev-tuning
- `full_finetuning_test.ipynb` — финальная оценка на benchmark
- `full_finetuning_subjB.ipynb` — дополнительная проверка на BCI Competition III Dataset II (subjB)

Общий код вынесен в:
- `src/downstream/stage5_utils.py`
- `src/downstream/model_unet.py`