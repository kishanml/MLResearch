from abc import ABC, abstractmethod
from typing import Union
from numpy.typing import NDArray

class Base(ABC):

    def __init__(self,
                 *,
                 max_iterations : int = 10**5,
                 patience : int = 100,
                 lr : float = 1e-4,
                 verbose : bool = True) : 
        
        self.max_iterations = max_iterations
        self.patience = patience
        self.lr = lr  
        self._verbose = verbose


        self._is_fitted : bool = False
        

    @abstractmethod
    def fit(X : NDArray, Y : NDArray) -> None:
        """
        Trains your model.

        Args:
            X (NDArray): input array
            Y (NDArray): output array
        """
        ...

    @abstractmethod
    def _compute_loss(y_true : NDArray, y_pred : NDArray) -> Union[float,NDArray]:
        """
        Calculates loss between true output vs predicted output.

        Args:
            y_true (NDArray): true output
            y_pred (NDArray): predicted output

        """
        ...

    @abstractmethod
    def predict(X : NDArray) -> NDArray:
        """
        Predicts output from trained model.

        Args:
            X (NDArray): input data

        """
        ...

    def predict_proba(X : NDArray) -> NDArray:
        """
        Predicts probability from trained model.

        Args:
            X (NDArray): input data

        """
        ...

    @abstractmethod
    def score(self,y_true : NDArray, y_pred : NDArray):
        """
        Computes score.

        Args:
            y_true (NDArray): true output.
            y_pred (NDArray): predicted output.

        """

        ...
