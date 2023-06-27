from omegaconf import DictConfig
import hydra
import process


@hydra.main(version_base='1.3', config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:
    features, labels, \
        sc_lst, cat_lst, \
        X_train, X_test, \
        y_train, y_test, \
        indices_train, indices_test, \
        xgb_train, xgb_test = process.prepare_data(cfg)

    model = process.train_xgboost(cfg=cfg,
                                  xgb_train=xgb_train,
                                  xgb_test=xgb_test)
    causal_graph = process.create_causal_graph(cfg=cfg,
                                               model=model,
                                               bg_dataset=X_train,
                                               cat_lst=cat_lst,
                                               method='xgboost')
    causal_graph.draw(rankdir="TB", show_gpath=False, save_fpath=cfg.save_fpath)

    process.explain_casual_graph(cfg=cfg,
                                 causal_graph=causal_graph,
                                 bg_dataset=X_test.loc[y_test[y_test==0].index, :],  # Здоровые на тесте
                                 fg_dataset=X_test.loc[y_test[y_test==1].index, :])  # Больные на тесте
    print('All done!')


if __name__ == "__main__":
    main()