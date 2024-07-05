import mlflow

from domain.diamond import Diamond


class MLflowClient:
    def __init__(self, model_uri: str):
        self.model_uri = model_uri

    def predict_price(self, diamond: Diamond):
        model = mlflow.pyfunc.load_model(self.model_uri)
        features = diamond.to_feature_array()
        # This prediction has some problems.
        # It returns wrong values. Possible causes:
        # - mismatch between the versions of the dependencies
        #   used between the workflow and the webserver
        # - model serialization while saving it to mlflow
        # - Feature preprocessing that does not match between
        #   workflow and webserver.
        #   (tip: maybe it is better to have shared preprocessing functions)
        price_prediction = model.predict(features)
        return float(price_prediction[0])
