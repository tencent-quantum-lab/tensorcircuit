"""
Useful utilities for ensemble
"""

from typing import Any, List
import tensorflow as tf
import keras
import numpy as np

NDArray = Any
kwargus = Any


class bagging:  # A.K.A. voting
    def __init__(self) -> None:
        self.models: List[keras.engine.functional.Functional] = []
        self.model_trained: List[bool] = []
        self.count = 0
        self.need_confidence = True  # Help in reducing numbers of get_confidence runs
        self.permit_train = False

    def append(
        self, model: keras.engine.functional.Functional, model_trained: bool
    ) -> None:
        """
        Add model to the voting method
        """
        self.models.append(model)
        self.model_trained.append(model_trained)
        self.count += 1

    def __train_model(self, i: int, **kwargs: kwargus) -> None:
        """
        Train a model if it isn't trained already
        """
        if not self.model_trained[i]:
            self.need_confidence = True
            self.model_trained[i] = True
            self.models[i].trainable
            self.models[i].fit(**kwargs)

    def train(self, **kwargs: kwargus) -> None:
        """
        Train all models in the class, **kwargs expect to receive the argus that can be directly sent to tf.fit
        Expected to be run after finishing compile
        """
        if not self.permit_train:
            # raise Exception("Needed to be compiled before training")
            raise ValueError()
        for i in range(self.count):
            if "verbose" in kwargs:
                if kwargs["verbose"] == 1:
                    print("Model ", i + 1, "/", self.count, " is training...")
            else:
                print("Model ", i + 1, "/", self.count, " is training...")
            self.__train_model(i, **kwargs)

    def compile(self, **kwargs: kwargus) -> None:
        self.permit_train = True
        for i in range(self.count):
            self.models[i].compile(**kwargs)

    def __get_confidence(self, model_index: int, input: NDArray) -> NDArray:
        """
        Get the confidence value that is needed by voting.
        Number of calling this function is reduced by self.need_confidence
        """
        self.need_confidence = False
        prediction = self.models[model_index].predict(input, verbose=0)
        prediction_returns = np.zeros(len(prediction))
        for i in range(len(prediction)):
            prediction_returns[i] = prediction[i][0]
        return prediction_returns

    """
    Voting strategies begin
    More voting strategies can be added beneath, a single function, and a if function in self.predict
    """

    def __voting_weight(self, array: NDArray) -> NDArray:
        result = []
        for i in array:
            result.append(self.__voting_weight_single(i))
        return np.array(result)

    def __voting_average(self, array: NDArray) -> NDArray:
        result = np.mean(array, axis=1)
        return result

    def __voting_weight_single(self, array: NDArray) -> float:
        opp_array = np.ones(len(array)) - array
        weight = np.absolute(opp_array - array)
        weight_sum = np.sum(weight)
        weight = weight / weight_sum
        result = array * weight
        return float(np.sum(result))

    def predict(self, input_data: NDArray, voting_policy: str = "None") -> NDArray:
        """
        Input data is expected to be a 2D array that the first layer is different input data (into the trained models)
        """
        if self.need_confidence:
            predictions = []
            for i in range(self.count):
                predictions.append(np.array(self.__get_confidence(i, input_data)))
            self.predictions = np.transpose(np.array(predictions))
        if voting_policy == "weight":
            return self.__voting_weight(self.predictions)
        elif voting_policy == "most":
            return self.__voting_average(self.predictions)
        elif voting_policy == "average":
            return self.__voting_average(self.predictions)
        elif voting_policy == "None" or voting_policy == "none":
            return self.predictions
        else:
            raise ValueError()

    def __acc_binarify(self, array: NDArray) -> NDArray:
        """
        Needed for ACC test
        """
        result = []
        for i in array:
            result.append(1 if (i > 0.5) else 0)
        return result

    def __eval_accuracy(self, input_data: NDArray) -> float:
        input_data[1] = self.__acc_binarify(input_data[1])
        algo = tf.keras.metrics.Accuracy()
        algo.reset_state()
        algo.update_state(input_data[0], input_data[1])
        return float(algo.result().numpy())

    def __eval_auc(self, input_data: NDArray) -> float:
        algo = tf.keras.metrics.AUC()
        algo.reset_state()
        algo.update_state(input_data[0], input_data[1])
        return float(algo.result().numpy())

    def eval(self, input_data: List[NDArray], evaluation_method: str = "acc") -> float:
        """
        Expect input data to be a 2D array, which a 1D array of yTrue followed by a 1D array of yPred is expected to be the components of the 2D array
        """
        if evaluation_method == "acc":
            return self.__eval_accuracy(input_data)
        elif evaluation_method == "auc":
            return self.__eval_auc(input_data)
        else:
            raise ValueError()
