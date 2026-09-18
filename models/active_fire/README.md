# Модуль AF — активное горение (YOLO11n)

Первая из двух моделей команды. Определяет пиксели природного горения по одному
снимку VIIRS и отдаёт бинарную маску `256×256 uint8` с классами `0/1`.

## Содержимое сущности

```text
models/active_fire/
├── predict.py        # адаптер инференса: load_model() / predict_af()
├── run_inference.py  # CLI: прогон по тестовым чипам -> results/active_fire_yolo.csv
├── artifacts/
│   ├── best.pt            # веса, YOLO11n detection
│   ├── results_yolo.csv   # журнал обучения, 100 эпох
│   └── manifest.json      # архитектура, хэши, заявленные и проверенные метрики
├── notebooks/baseline_yolo.ipynb   # исходный notebook сокомандника
├── docs/STRUCTURE_YOLO.md          # описание решения от автора модели
└── results/active_fire_yolo.csv    # готовый сабмишн модуля, 180 строк
```

## Контракт

```python
from pathlib import Path
from models.active_fire import WEIGHTS, load_model, predict_af

model = load_model(WEIGHTS, device="cpu")
mask = predict_af(Path("…/AF_xxx_VIIRS_I1-I5.tif"), model)
# numpy.ndarray, uint8, (256, 256), классы 0/1
```

`predict_af` принимает либо сам GeoTIFF, либо каталог чипа с ровно одним
`*_VIIRS_I1-I5.tif`. Результат проверяется `validate_af_mask` до возврата.

## Как устроен инференс

1. Каналы `I3`, `I4`, `I5` и `valid` читаются по **описаниям** GeoTIFF, а не по номерам.
2. Фиксированная нормировка как в notebook: `I4` → `[270, 367] K`, `I4-I5` → `[0, 60] K`,
   `I3` → `[0, 0.6]`; каждый канал масштабируется в `uint8`.
3. Невалидные пиксели (`valid < 0.5`) обнуляются во всех трёх каналах.
4. Картинка BGR увеличивается nearest-neighbour ×4 до `1024×1024` — так обучалась модель.
5. YOLO предсказывает с порогом-полом `conf=0.001`, боксы фильтруются порогом `0.5`
   и растеризуются обратно в сетку `256×256`.

## Прогон на тестовых данных

```powershell
.venv-af/Scripts/python.exe -m models.active_fire.run_inference `
  --data-dir C:/path/to/test `
  --sample C:/path/to/test/sample_submission.csv `
  --device cpu
```

По умолчанию берутся веса `models/active_fire/artifacts/best.pt`, результат пишется
в `models/active_fire/results/active_fire_yolo.csv`. Ожидаемая структура данных —
`<data-dir>/af/viirs/<chip_id>_VIIRS_I1-I5.tif`. Зависимости: `requirements-af.txt`
(отдельное окружение `.venv-af`, потому что Torch и Ultralytics не нужны веб-сервису).

Проверить файл, не запуская модель:

```powershell
.venv/Scripts/python.exe -m scripts.collect_results `
  --check models/active_fire/results/active_fire_yolo.csv
```

## Что проверено, а что нет

Проверено: веса загружаются Ultralytics (класс `fire`), полный `predict_af` на
синтетическом 8-канальном VIIRS GeoTIFF возвращает валидную маску `256×256 uint8`.
Сабмишн модуля: 180 уникальных строк, 120 непустых масок, 9246 положительных
пикселей, ошибок классов, границ и RLE нет — это покрыто тестом
`tests/test_active_fire.py::test_supplied_af_submission_is_valid`.

Не проверено: заявленные `F1 = 0.4176`, `precision = 0.2836`, `recall = 0.792`
независимо не воспроизводились — notebook сохранён без выполненных ячеек,
train/validation GeoTIFF не переданы. В `manifest.json` они помечены как
`declared_*`, а `training_reproduced: false`.

Ограничение по правилам кейса: конкурсные тестовые AF-маски нельзя показывать на
карте — координаты и даты теста скрыты. Для демонстрации в веб-приложении модель
запускается на отдельном разрешённом геопривязанном VIIRS-наборе, и полученная
`.npy` маска загружается в сервис вместе с метаданными.
