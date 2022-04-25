from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer


class cardinal_to_degrees(BaseEstimator, TransformerMixin):
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        cardinal_directions = {
            'N':0,
            'NNE':22.5,
            'NE':45,
            'ENE':67.5,
            'E':90,
            'ESE':112.5,
            'SE':135,
            'SSE':157.5,
            'S':180,
            'SSW':202.5,
            'SW':225,
            'WSW':247.5,
            'W':270,
            'WNW':292.5,
            'NW':315,
            'NNW':337.5}
        X_ = X.copy()
    
        for direction in cardinal_directions:
            X_.loc[X_["Direction"] == direction, "Direction"] = cardinal_directions[direction]
                    
        return X_
    
    
numeric_transformer = Pipeline(steps=[
       ('imputer', SimpleImputer(strategy='mean'))
      ,('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
   transformers=[
    ('numeric', numeric_transformer, ['Speed']),
    ('decode', cardinal_to_degrees(), ['Direction'])
]) 


pipeline = Pipeline(steps = [
               ('preprocessor', preprocessor)
              ,('poly_features', PolynomialFeatures(degree=3, include_bias=False))
              ,('regressor',LinearRegression())
           ])
