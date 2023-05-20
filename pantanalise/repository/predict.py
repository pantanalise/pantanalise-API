from abc import ABC, abstractmethod

class Predict(ABC):

    @abstractmethod
    def predict(self, data):
        pass