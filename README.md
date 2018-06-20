# laplace
Linear regression for Laplace distributed targets

## Usage
```python
from LaplaceLinearRegressor import LaplaceLinearRegressor

x = np.linspace(0,10,20)
y = 2*x + 5
y[0] = 20
y[1] = 25
y[-1] = 10
y[-2] = 5

llr = LaplaceLinearRegressor()
llr.fit(x, y)
y_pred = llr.predict(x)
print("weights =",llr.w)
```
