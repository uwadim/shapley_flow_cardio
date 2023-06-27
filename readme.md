# Риск ранней послеоперационной летальности у пациентов, перенесших оперативное лечение по поводу патологии аортального клапана

## Установка
1. Проверяем/устанавливаем graphviz в систему
https://software.opensuse.org/package/graphviz

2. Создаем новое окружение (опционально), активируем его
``` bash
conda create -n shapley_flow_explain python~=3.11
conda activate shapley_flow_explain
```
3. Клонируем репозитарий Shapley Flow
``` bash
git clone https://github.com/nathanwang000/Shapley-Flow.git
```
4. Доустанавливаем необходимиые пакеты.

_У меня не сработала установка graphviz и pygraphviz из conda-forge, поэтому graphviz поставил из pip_
``` bash
python -m pip install graphviz
conda install -c conda-forge py-xgboost-gpu pandas tqdm joblib dill scikit-learn pygraphviz
```
5. Устанавливаем Hydra
```bash
python -m pip install hydra-core --upgrade
```
6. Для проверки запускаем тестовые файлы, убеждаемся, что они выполняются без ошибок.
``` bash
python xgtest.py

cd Shapley-Flow
python flow.py
```
## Структура проекта

- \data - данные проекта
- \Shapley-Flow - склонированный проект (используем flow.py)
- \src исходный код проекта

_Из папки Shapley-Flow скопировали flow.py и доработали_

Для информации о версиях пакетов привожу файлы
 - conda_list.txt
 - pip_freeze.txt
