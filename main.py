import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets
from torchvision import datasets, transforms, models
import os
from tqdm import tqdm

# предобработка данных
train_transforms = transforms.Compose([
    # transforms.RandomResizedCrop(224),  # Случайное изменение размера и обрезка до 224x224 пикселей
    # transforms.RandomHorizontalFlip(),  # Случайное горизонтальное отражение изображения

    transforms.Resize(256),  # Изменение размера до 256x256 пикселей
    transforms.CenterCrop(224),  # Обрезка до 224x224 пикселей по центру»

    transforms.ToTensor(),  # Преобразование в тензор (многомерный массив)
])

val_transforms = transforms.Compose([
    transforms.Resize(256),  # Изменение размера до 256x256 пикселей
    transforms.CenterCrop(224),  # Обрезка до 224x224 пикселей по центру»
    transforms.ToTensor(),  # Преобразование в тензор
])

data_dir = 'dataroot'   # Каталог с подкаталогами
train_dir = 'train'     # каталог с тренировочными данными
val_dir = 'val'         # каталог с валидационными данными

# Создание ImageFolder датасета для обучения и валидации с использованием заданных трансформаций
train_dataset = torchvision.datasets.ImageFolder(os.path.join(data_dir, train_dir), train_transforms)
val_dataset = torchvision.datasets.ImageFolder(os.path.join(data_dir, val_dir), val_transforms)

batch_size = 32         # Размер пакета данных для обучения


# Создание DataLoader для загрузки данных с пакетами, перемешиванием и указанием числа рабочих процессов
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)


# количество батчей с изображениями в загрузчике, общее количество изображений для обучения
print('ТРЕНИРОВОЧНЫЙ ДАТАСЕТ: Кол-во батчей - ', len(train_dataloader), ', кол-во изображений - ', len(train_dataset))
print('ВАЛИДАЦИОННЫЙ ДАТАСЕТ: Кол-во батчей - ', len(val_dataloader), ', кол-во изображений - ', len(val_dataset))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Определение устройства (GPU или CPU)

# ========== ОБУЧЕНИЕ МОДЕЛИ  ===========

# Функция для обучения модели, принимает:
    # model - предобученную модель
    # loss - функция потерь
    # optimiaer - функция оптимизатора
    # num_epochs - количество эпох
def train_model(model, loss, optimizer, num_epochs):
    for epoch in range(num_epochs):
        print('Epoch {}/{}:'.format(epoch + 1, num_epochs), flush=True)

        for phase in ['train', 'val']:
            if phase == 'train':
                dataloader = train_dataloader
                model.train()  # установка модели в тренировоный режим
            else:
                dataloader = val_dataloader
                model.eval()   # установка модели в валидационный режим

            running_loss = 0.
            running_acc = 0.

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad() # стохастический градиентный спуск для обновления весов модели

                # forward and backward
                with torch.set_grad_enabled(phase == 'train'):
                    preds = model(inputs)
                    loss_value = loss(preds, labels)
                    preds_class = preds.argmax(dim=1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss_value.backward()
                        optimizer.step()

                # статистика
                running_loss += loss_value.item()
                running_acc += (preds_class == labels.data).float().mean()

            epoch_loss = running_loss / len(dataloader)
            epoch_acc = running_acc / len(dataloader)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc), flush=True)

    return model
#
# Загрузка предварительно обученной модели ResNet18
model = models.resnet18(pretrained=True)
#
# Получение количества признаков в последнем полносвязном слое
num_ftrs = model.fc.in_features
#
# Замена последнего полносвязного слоя на слой с 2 выходами (2 класса: мафин и чихуа)
model.fc = nn.Linear(num_ftrs, 2)
#
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Определение устройства (GPU или CPU)
model = model.to(device)  # Перемещение модели на выбранное устройство
#
# Определение функции потерь (cross-entropy) и оптимизатора (SGD)
loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.8)
# lr - скорость обучения
# momentum - momentum factor
#
# # Обучение модели
model = train_model(model, loss, optimizer, num_epochs=4)

# ========== СОХРАНЕНИЕ МОДЕЛИ  ===========
torch.save(model.state_dict(), 'mafin_chihua_classification.pth')

# ========== ОПРЕДЕЛЕНИЕ ТОЧНОСТИ ПРЕДСКАЗАНИЯ НА ТЕСТВОЙ ВЫБОРКЕ  ===========

test_transforms = transforms.Compose([
    transforms.Resize(256),  # Изменение размера до 256x256 пикселей
    transforms.CenterCrop(224),  # Обрезка до 224x224 пикселей по центру»
    transforms.ToTensor(),  # Преобразование в тензор
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_dir = 'test'     # каталог с тестовыми данными

# Создание ImageFolder датасета для тетсирования с использованием заданных трансформаций
test_dataset = torchvision.datasets.ImageFolder(os.path.join(data_dir, test_dir), val_transforms)

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# количество батчей с изображениями в тестовой выборке в загрузчике, общее количество изображений в тестовой выборке
print('ТЕСТОВЫЙ ДАТАСЕТ: Кол-во батчей - ', len(test_dataloader), ', кол-во изображений - ', len(test_dataset))



import sklearn.metrics as metrics
import numpy as np

def asd(test_dataloader):

    test_predictions = []
    test_true_labels = []
    model.eval()
    for inputs, labels in tqdm(test_dataloader):

        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            preds = model(inputs)
        test_predictions.append(
            torch.nn.functional.softmax(preds, dim=1)[:, 1].data.cpu().numpy())
        test_true_labels.append(labels.data.cpu().numpy())

    test_predictions = np.concatenate(test_predictions)
    test_true_labels = np.concatenate(test_true_labels)
    test_pred_labels = np.where(test_predictions > 0.5, 1, 0)

    # accuracy
    return metrics.accuracy_score(test_true_labels, test_pred_labels)

print('Точность предсказания: {:.2f}%'.format(asd(test_dataloader)*100))
