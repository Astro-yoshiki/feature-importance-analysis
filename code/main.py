from preprocessing import Preprocessor
from selection import Modeling
from analysis import Plot

if __name__ == "__main__":
    verbose = 1
    # Part of Data preprocessing
    print("***** Preprocessing Start *****")
    data_path = "../Data/master_data.csv"
    preprocessor = Preprocessor(data_path)
    preprocessor.read_data()
    preprocessor.input_combination(verbose=verbose)
    print("***** Preprocessing Finished *****")

    # Part of creating models
    print("***** Modeling Start *****")
    modeling = Modeling(data_path=data_path)
    modeling.solver(verbose=verbose)
    modeling.save_best_model()
    print("***** Modeling Finished *****")

    # Part of visualizing the results
    print("***** Visualization Start *****")
    model_data_path = "../Data/selected_data.csv"
    plot = Plot(data_path=model_data_path)
    plot.rank_table()
    plot.predict_by_best_model()
    plot.feature_importance()
    plot.parity_plot()
    print("***** Visualization Finished *****")
