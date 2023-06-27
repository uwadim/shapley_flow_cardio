# Обработка данных
from typing import Optional

import xgboost
from omegaconf import DictConfig
import pandas as pd
from scipy.odr import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.flow import CausalLinks, create_xgboost_f, build_feature_graph, Graph, GraphExplainer


def prepare_data(cfg: DictConfig) -> tuple:
    """Подготавливает данные для XGBoost, разбивая на train и test

    Parameters
    ----------
    cfg : DictConfig
        Объект конфигурационного файла

    Returns
    -------
    tuple
        features, labels, \
        sc_lst, cat_lst, \
        X_train, X_test, \
        y_train, y_test, \
        indices_train, indices_test, \
        xgb_train, xgb_test
    """

    features = pd.read_pickle(cfg.data['features_fpath'])
    labels = pd.read_pickle(cfg.data['labels_fpath'])

    # Разделяем категриальные и вещественные переменные
    cat_lst = [x for x in features[cfg.data.predictors].columns if pd.CategoricalDtype.is_dtype(features[x])]
    sc_lst = [col for col in features[cfg.data.predictors].columns if col not in cat_lst]

    # Преобразуем бинарные переменные в целые
    features[cat_lst] = features[cat_lst].astype(int)

    # Стратифицированный сплит
    X_train, X_test, \
        y_train, y_test, \
        indices_train, indices_test = \
        train_test_split(features[cfg.data.predictors],
                         labels,
                         features[cfg.data.predictors].index,
                         test_size=cfg.training.test_size,
                         random_state=cfg.random_state,
                         stratify=labels)

    if cfg.training['normalize_num_features']:
        # Нормализуем вещественные переменные
        sc = StandardScaler()
        X_train[sc_lst] = sc.fit_transform(X_train[sc_lst])
        X_test[sc_lst] = sc.transform(X_test[sc_lst])

    # Создание датасета для xgboost
    xgb_train = xgboost.DMatrix(X_train, label=y_train)
    xgb_test = xgboost.DMatrix(X_test, label=y_test)

    return features, labels, \
        sc_lst, cat_lst, \
        X_train, X_test, \
        y_train, y_test, \
        indices_train, indices_test, \
        xgb_train, xgb_test


def train_xgboost(cfg: DictConfig, xgb_train, xgb_test):
    """Обучает XGBoost

    Parameters
    ----------
    cfg : DictConfig
        Объект конфигурационного файла
    xgb_train : xgboost.DMatrix
        Обучающая выборка в формате xgboost.DMatrix
    xgb_test : xgboost.DMatrix
        Тестовая выборка в формате xgboost.DMatrix

    Returns
    -------

    """

    # Создаем модель.
    # Склеиваем параметры
    xgb_params: dict = dict(cfg.training.optim_params).update(cfg.training.model_params)
    return xgboost.train(params=xgb_params,
                         dtrain=xgb_train,
                         evals=[(xgb_test, "test")])


def create_causal_graph(cfg: DictConfig,
                        model: xgboost.Booster,
                        bg_dataset: pd.DataFrame,
                        cat_lst: Optional[list[str]],
                        method: str = 'xgboost'
                        ) -> Graph:
    """
    Создает причинный граф
    Parameters
    ----------
    cfg : DictConfig
        объект конфига
    model : xgboost.Booster
        Обученная модель для восстановления ребер причинного графа
    bg_dataset : pd.DataFrame
        Датасет, относительно которого будет построен причинный граф
    cat_lst :list, optional
        список категориальных переменных
    method : {'xgboost', 'linear'}, default='xgboost'
        метод построения причинного графа

    Returns
    -------
    Graph
        причинный граф
    """
    # Создаем объект для хранения причинного графа
    causal_links = CausalLinks()
    # Ребра учета непосредственного влияния фич на результат
    causal_links.add_causes_effects(causes=list(cfg.data.predictors),
                                    effects=cfg.data.target,
                                    models=create_xgboost_f(cfg.data.predictors, model, output_margin=True))
    # добавляем ребра влияния одних фич на другие
    for item in cfg.data.features:
        causal_links.add_causes_effects(causes=item[0], effects=item[1])

    return build_feature_graph(X=bg_dataset,
                               causal_links=causal_links,
                               categorical_feature_names=cat_lst,
                               target_name=cfg.data.target,
                               method=method)


def explain_casual_graph(cfg, causal_graph, bg_dataset, fg_dataset) -> None:
    G = GraphExplainer(graph=causal_graph, bg=bg_dataset, nruns=cfg.training.nruns)
    cf_flow_adult = G.shap_values(X=fg_dataset)
    cf_flow_adult.draw(idx=-1,
                       max_display=len(cfg.data.predictors),
                       show_fg_val=True,
                       save_fpath=cfg.result_fpath)