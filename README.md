# ML_CV_ImageClassification
## Обучение модели для распознавания маффинов и чихуахуа

Результаты определения точности на тестовой выборке в зависимости от параметров

тренировочные, валидационные и тестовые данные лежат в папке '/dataroot'

Все данные разделены на две соответствующие папки

### Размеры данных:
ТРЕНИРОВОЧНЫЙ ДАТАСЕТ: кол-во изображений -  282\
ВАЛИДАЦИОННЫЙ ДАТАСЕТ: кол-во изображений -  192\
ТЕСТОВЫЙ ДАТАСЕТ: кол-во изображений -  433

## Подбор гипперпараметров

### Изменение количества эпох:

1.1\
Batch_size = 32\
Epoch_num = 1\
Torch.optim.SGD(lr=0.001, momentum=0.9)\
Точность предсказания 57.96%\
1.2\
Batch_size = 32\
Epoch_num = 2\
Torch.optim.SGD(lr=0.001, momentum=0.9)\
Точность предсказания 97.92%\
1.3\
Batch_size = 32\
Epoch_num = 4\
Torch.optim.SGD(lr=0.001, momentum=0.9)\
Точность предсказания 98.61%\
1.4\
Batch_size = 32\
Epoch_num = 10\
Torch.optim.SGD(lr=0.001, momentum=0.9)\
Точность предсказания 98.85%\
#### ВЫВОД: Оптимальное количество эпох для данного датасета - 4

## Изменение batch_size:
2.1\
Batch_size = 2\
Epoch_num = 2\
Torch.optim.SGD(lr=0.001, momentum=0.9)\
Точность предсказания 90.07%\
2.2\
Batch_size = 8\
Epoch_num = 2\
Torch.optim.SGD(lr=0.001, momentum=0.9)\
Точность предсказания 98.38%\
2.3\
Batch_size = 16\
Epoch_num = 2\
Torch.optim.SGD(lr=0.001, momentum=0.9)\
Точность предсказания 99.08%\
2.4\
Batch_size = 32\
Epoch_num = 2\
Torch.optim.SGD(lr=0.001, momentum=0.9)\
Точность предсказания 97.92%\
2.5\
Batch_size = 64\
Epoch_num = 2\
Torch.optim.SGD(lr=0.001, momentum=0.9)\
Точность предсказания 88.68%
#### ВЫВОД: рекомендуемый размер batch_size – (8 – 32)

## Изменение параметров оптимизатора SGD:
3.1\
Batch_size = 16\
Epoch_num = 2\
Torch.optim.SGD(lr=0.1, momentum=0.9)\
Точность предсказания 52.19%\
3.2\
Batch_size = 16\
Epoch_num = 2\
Torch.optim.SGD(lr=0.01, momentum=0.9)\
Точность предсказания 81.52%\
3.3\
Batch_size = 16\
Epoch_num = 2\
Torch.optim.SGD(lr=0.001, momentum=0.9)\
Точность предсказания 98.90%\
3.4\
Batch_size = 16\
Epoch_num = 2\
Torch.optim.SGD(lr=0.0001, momentum=0.9)\
Точность предсказания 93.30%\
3.5\
Batch_size = 16\
Epoch_num = 2\
Torch.optim.SGD(lr=0.001, momentum=0)\
Точность предсказания 89.15%\
3.6\
Batch_size = 16\
Epoch_num = 2\
Torch.optim.SGD(lr=0.001, momentum=0.5)\
Точность предсказания 95.15%\
3.7\
Batch_size = 16\
Epoch_num = 2\
Torch.optim.SGD(lr=0.001, momentum=0.8)\
Точность предсказания 97.23%\
3.8\
Batch_size = 16\
Epoch_num = 2\
Torch.optim.SGD(lr=0.001, momentum=1.0)\
Точность предсказания 98.38%\
#### ВЫВОД: рекомендуемый параметры оптимизатора SGD – (lr=0.001, momentum=1.0)\

## ИТОГ:
#### Вывод с оптимальными гиперпараметрами:

Batch_size = 16\
Epoch_num = 4\
Torch.optim.SGD(lr=0.001, momentum=1.0)\
Точность предсказания 97.07%


























