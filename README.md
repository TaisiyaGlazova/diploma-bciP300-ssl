# Применение SSL-методов  машинного обучения  для классификации стимулов в ИМК-P300

Этот репозиторий содержит код для ВКР по применению SSL-методов к ЭЭГ-записям в ИМК-Р300 для классификации целевых и нецелевых стимулов.

## Структура репозитория

```text
build_ssl.ipynb - ноутбук с исследованием данных датасета BigP300BCI и сборкой SSL-выборки (большой нерабочей и маленькой рабочей)

SSL_reconstruction_50k_colab.ipynb - ноутбук с полным пайплайном SSL модели в reconstruction-задаче для маленькой выборки (50к)

linear_eval_sslreconstrucrion_50k.ipynb - ноутбук с Linear evaluation для SSL модели в reconstruction-задаче для маленькой выборки (50к)

SSL_masking-reconstruction_50k_colab.ipynb - - ноутбук с полным пайплайном SSL модели в masking-reconstruction-задаче для маленькой выборки (50к)

linear_eval_ssl20epoch_maskingreconstrucrion_50k.ipynb - ноутбук с Linear evaluation для SSL Недообученной модели (20 эпох) в masking-reconstruction-задаче для маленькой выборки (50к)

linear_eval_ssl200epoch_maskingreconstrucrion_50k.ipynb- ноутбук с Linear evaluation для SSL Дообученной модели (200 эпох) в masking-reconstruction-задаче для маленькой выборки (50к)



models:
unet_ssl.py - UNet модель
