import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

x1=np.linspace(0,2*np.pi,50)
y1=np.sin(x1)+np.random.randn(len(x1))/5.0
slr=LinearRegression()
x1=x1.reshape(-1,1)
slr.fit(x1,y1)
print(slr.coef_)
print(slr.intercept_)
predicted_y1=slr.predict(x1)
poly_features_3=PolynomialFeatures(degree=1,include_bias=False)
X_poly_3=poly_features_3.fit_transform(x1)
lin_reg_3=LinearRegression()
lin_reg_3.fit(X_poly_3,y1)
print(lin_reg_3.intercept_,lin_reg_3.coef_)
X_plot=np.linspace(0,6,1000).reshape(-1,1)
X_plot_poly=poly_features_3.fit_transform(X_plot)
y_plot=np.dot(X_plot_poly,lin_reg_3.coef_.T)+lin_reg_3.intercept_
plt.subplot(4,2,1)
plt.plot(X_plot,y_plot,'r-')
plt.plot(x1,y1,'b.')


x1=np.linspace(0,2*np.pi,50)
y1=np.sin(x1)+np.random.randn(len(x1))/5.0
slr=LinearRegression()
x1=x1.reshape(-1,1)
slr.fit(x1,y1)
print(slr.coef_)
print(slr.intercept_)
predicted_y1=slr.predict(x1)
poly_features_3=PolynomialFeatures(degree=3,include_bias=False)
X_poly_3=poly_features_3.fit_transform(x1)
lin_reg_3=LinearRegression()
lin_reg_3.fit(X_poly_3,y1)
print(lin_reg_3.intercept_,lin_reg_3.coef_)
X_plot=np.linspace(0,6,1000).reshape(-1,1)
X_plot_poly=poly_features_3.fit_transform(X_plot)
y_plot=np.dot(X_plot_poly,lin_reg_3.coef_.T)+lin_reg_3.intercept_
plt.subplot(4,2,2)
plt.plot(X_plot,y_plot,'r-')
plt.plot(x1,y1,'b.')


x1=np.linspace(0,2*np.pi,50)
y1=np.sin(x1)+np.random.randn(len(x1))/5.0
slr=LinearRegression()
x1=x1.reshape(-1,1)
slr.fit(x1,y1)
print(slr.coef_)
print(slr.intercept_)
predicted_y1=slr.predict(x1)
poly_features_3=PolynomialFeatures(degree=5,include_bias=False)
X_poly_3=poly_features_3.fit_transform(x1)
lin_reg_3=LinearRegression()
lin_reg_3.fit(X_poly_3,y1)
print(lin_reg_3.intercept_,lin_reg_3.coef_)
X_plot=np.linspace(0,6,1000).reshape(-1,1)
X_plot_poly=poly_features_3.fit_transform(X_plot)
y_plot=np.dot(X_plot_poly,lin_reg_3.coef_.T)+lin_reg_3.intercept_
plt.subplot(4,2,3)
plt.plot(X_plot,y_plot,'r-')
plt.plot(x1,y1,'b.')


x1=np.linspace(0,2*np.pi,50)
y1=np.sin(x1)+np.random.randn(len(x1))/5.0
slr=LinearRegression()
x1=x1.reshape(-1,1)
slr.fit(x1,y1)
print(slr.coef_)
print(slr.intercept_)
predicted_y1=slr.predict(x1)
poly_features_3=PolynomialFeatures(degree=9,include_bias=False)
X_poly_3=poly_features_3.fit_transform(x1)
lin_reg_3=LinearRegression()
lin_reg_3.fit(X_poly_3,y1)
print(lin_reg_3.intercept_,lin_reg_3.coef_)
X_plot=np.linspace(0,6,1000).reshape(-1,1)
X_plot_poly=poly_features_3.fit_transform(X_plot)
y_plot=np.dot(X_plot_poly,lin_reg_3.coef_.T)+lin_reg_3.intercept_
plt.subplot(4,2,4)
plt.plot(X_plot,y_plot,'r-')
plt.plot(x1,y1,'b.')


x1=np.linspace(0,2*np.pi,100)
y1=np.sin(x1)+np.random.randn(len(x1))/5.0
slr=LinearRegression()
x1=x1.reshape(-1,1)
slr.fit(x1,y1)
print(slr.coef_)
print(slr.intercept_)
predicted_y1=slr.predict(x1)
poly_features_3=PolynomialFeatures(degree=1,include_bias=False)
X_poly_3=poly_features_3.fit_transform(x1)
lin_reg_3=LinearRegression()
lin_reg_3.fit(X_poly_3,y1)
print(lin_reg_3.intercept_,lin_reg_3.coef_)
X_plot=np.linspace(0,6,1000).reshape(-1,1)
X_plot_poly=poly_features_3.fit_transform(X_plot)
y_plot=np.dot(X_plot_poly,lin_reg_3.coef_.T)+lin_reg_3.intercept_
plt.subplot(4,2,5)
plt.plot(X_plot,y_plot,'r-')
plt.plot(x1,y1,'b.')


x1=np.linspace(0,2*np.pi,100)
y1=np.sin(x1)+np.random.randn(len(x1))/5.0
slr=LinearRegression()
x1=x1.reshape(-1,1)
slr.fit(x1,y1)
print(slr.coef_)
print(slr.intercept_)
predicted_y1=slr.predict(x1)
poly_features_3=PolynomialFeatures(degree=3,include_bias=False)
X_poly_3=poly_features_3.fit_transform(x1)
lin_reg_3=LinearRegression()
lin_reg_3.fit(X_poly_3,y1)
print(lin_reg_3.intercept_,lin_reg_3.coef_)
X_plot=np.linspace(0,6,1000).reshape(-1,1)
X_plot_poly=poly_features_3.fit_transform(X_plot)
y_plot=np.dot(X_plot_poly,lin_reg_3.coef_.T)+lin_reg_3.intercept_
plt.subplot(4,2,6)
plt.plot(X_plot,y_plot,'r-')
plt.plot(x1,y1,'b.')


x1=np.linspace(0,2*np.pi,100)
y1=np.sin(x1)+np.random.randn(len(x1))/5.0
slr=LinearRegression()
x1=x1.reshape(-1,1)
slr.fit(x1,y1)
print(slr.coef_)
print(slr.intercept_)
predicted_y1=slr.predict(x1)
poly_features_3=PolynomialFeatures(degree=5,include_bias=False)
X_poly_3=poly_features_3.fit_transform(x1)
lin_reg_3=LinearRegression()
lin_reg_3.fit(X_poly_3,y1)
print(lin_reg_3.intercept_,lin_reg_3.coef_)
X_plot=np.linspace(0,6,1000).reshape(-1,1)
X_plot_poly=poly_features_3.fit_transform(X_plot)
y_plot=np.dot(X_plot_poly,lin_reg_3.coef_.T)+lin_reg_3.intercept_
plt.subplot(4,2,7)
plt.plot(X_plot,y_plot,'r-')
plt.plot(x1,y1,'b.')


x1=np.linspace(0,2*np.pi,100)
y1=np.sin(x1)+np.random.randn(len(x1))/5.0
slr=LinearRegression()
x1=x1.reshape(-1,1)
slr.fit(x1,y1)
print(slr.coef_)
print(slr.intercept_)
predicted_y1=slr.predict(x1)
poly_features_3=PolynomialFeatures(degree=9,include_bias=False)
X_poly_3=poly_features_3.fit_transform(x1)
lin_reg_3=LinearRegression()
lin_reg_3.fit(X_poly_3,y1)
print(lin_reg_3.intercept_,lin_reg_3.coef_)
X_plot=np.linspace(0,6,1000).reshape(-1,1)
X_plot_poly=poly_features_3.fit_transform(X_plot)
y_plot=np.dot(X_plot_poly,lin_reg_3.coef_.T)+lin_reg_3.intercept_
plt.subplot(4,2,8)
plt.plot(X_plot,y_plot,'r-')
plt.plot(x1,y1,'b.')