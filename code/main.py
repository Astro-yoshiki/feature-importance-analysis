from data_preprocessing import Preprocess
from variable_selection import Modeling
from analysis_of_result import Plot

if __name__ == "__main__":
    # データの前処理
    print("***** Preprocessing Start *****")
    data_path = "../Data/master_data.csv"
    preprocessor = Preprocess(data_path)
    preprocessor.scaling()
    preprocessor.input_combination()
    print("***** Preprocessing Finished *****")

    # 異なる入力パラメータで訓練
    print("***** Modeling Start *****")
    modeling = Modeling(data_path=data_path)
    modeling.solver()
    modeling.save_best_model()
    print("***** Modeling Finished *****")

    # parity-plotと予測結果の可視化
    print("***** Visualization Start *****")
    model_data_path = "../Data/selected_data.csv"
    plot = Plot(data_path=model_data_path)
    plot.rank_table()
    plot.predict_by_best_model()
    plot.feature_importance()
    plot.parity_plot()
    print("***** Visualization Finished *****")
