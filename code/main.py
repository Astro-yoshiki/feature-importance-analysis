from data_preprocessing import Preprocess
from variable_selection import Modeling
from analysis_of_result import Plot

if __name__ == "__main__":
    data_path = "../Data/"
    preprocessor = Preprocess(data_path)
