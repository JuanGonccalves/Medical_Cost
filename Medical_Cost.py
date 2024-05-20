#!/usr/bin/env python
# coding: utf-8

# <b>Medical Cost Personal Datasets</b>
# 
# <b>Site</b>: <i>https://www.kaggle.com/datasets/mirichoi0218/insurance</i>
# - age: age of primary beneficiary
# - sex: insurance contractor gender, female, male
# - bmi: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height,
# objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9
# - children: Number of children covered by health insurance / Number of dependents
# - smoker: Smoking
# - region: the beneficiary's residential area in the US, northeast, southeast, southwest, northwest.
# - charges: Individual medical costs billed by health insurance

# # Import libraries:

# In[198]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np
from yellowbrick.regressor import ResidualsPlot 
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve


# # Load file:

# In[142]:


base = pd.read_csv('insurance.csv')


# # Read file:

# - <i> <b>Top</b> five </i>

# In[143]:


base.head()


# - <i> <b>Bottom</b> five </i>

# In[144]:


base.tail()


# # EDA:

# - <i>Read number columns</i>

# In[145]:


base.shape


# - <i>Read number duplicated</i>

# In[146]:


print(f"Read number duplicated: {base.duplicated().sum()}")


# - <i>Remove duplicated</i>

# In[147]:


base = base.drop_duplicates()
print(f"Read number duplicated: {base.duplicated().sum()}")


# - <i>In the line bellow, We exploration <b>type</b> columns</i>

# In[148]:


base.info()


# - <i>Checking for <b>empty</b> lines:</i>

# In[149]:


base.isnull().sum()


# # Resume stats:

# In[118]:


base.describe()


# - <i>Overview</i>

# In[119]:


sns.pairplot(base, hue='smoker')
plt.show()


# * <i><b>Age</b> Distribution:</i>

# In[151]:


sns.histplot(data = base, x = "age", kde=True, stat="density",color='blue').set(title='Age')


# - <i><b>Sex</b> Distribution:</i>

# In[121]:


# Assuming 'base' is your DataFrame and you've already calculated contagem_por_categoria
sex_count = base['sex'].value_counts()

# Plotting
sns.barplot(x=sex_count.index, y=sex_count.values).set(title='Count sex')
plt.show()


# - <i>Body Mass Index <b>(BMI)</b>:</i>

# In[122]:


sns.histplot(base['bmi'], kde=True, bins=6, color='blue').set(title='BMI')


# - <i>Number of <b>children</b>:</i>

# In[155]:


# Assuming 'base' is your DataFrame and you've already calculated contagem_por_categoria
ch_count = base['children'].value_counts()

# Plotting
sns.barplot(x=ch_count.index, y=ch_count.values).set(title='children')
plt.show()


# - <i>Distribution of <b>smokers</b>:</i>

# In[156]:


# Assuming 'base' is your DataFrame and you've already calculated contagem_por_categoria
sm_count = base['smoker'].value_counts()

# Plotting
sns.barplot(x=sm_count.index, y=sm_count.values).set(title='Count smoker')
plt.show()


# - <i>Distribution of <b>regions</b>:</i>

# In[185]:


# Assuming 'base' is your DataFrame and you've already calculated contagem_por_categoria
rg_count = base['region'].value_counts()

# Plotting
sns.barplot(x=rg_count.index, y=rg_count.values).set(title='Distribution of regions')
plt.show()


# - Price x Age by number of smokers:

# In[158]:


#sns.histplot(base['charges'], kde=True, bins=10, color='blue').set(title='Custos')
sns.scatterplot(base, x = 'age',y = 'charges', hue = 'smoker').set(title='Custos')


# - Price x BMI by number of smokers:

# In[160]:


#sns.histplot(base['charges'], kde=True, bins=10, color='blue').set(title='Custos')
sns.scatterplot(base, x = 'bmi',y = 'charges', hue = 'smoker').set(title='Custos')


# - Charges per smoker

# In[128]:


sns.boxplot(base, x = 'smoker', y = 'charges')


# # Transform Data:

# In[161]:


base.head()


# - <i>Convert columns into dummy variables with a <b>specified prefix for each column</b> and concatenate them into the original DataFrame:</i>

# In[187]:


df = pd.get_dummies(base, columns=['region', 'sex', 'smoker'], prefix=['region', 'sex', 'smoker'], dtype=int)


# - Verify the updated DataFrame columns

# In[188]:


print(f"Updated columns: {df.columns}")


# In[189]:


df.head()


# In[190]:


correlation = df.corr()
# Ajustar o tamanho da figura
plt.figure(figsize=(10, 8))  # Ajuste a largura e altura conforme necessário

# Criar o heatmap
plot = sns.heatmap(correlation, annot=True, fmt=".1f", linewidths=.6)

#plot = sns.heatmap(correlation, annot = True, fmt=".1f", linewidths=.6)
plot


# - Ao traçar um mapa de correlação, nota-se que as principais correlações são: <b>age, bmi, children, smoker_no, smoker_yes, sex_female, sex_male</b>
# 
# - Sendo assim, utilizaremos as variáveis acima para nosso modelo.

# # Regressão Linear:

# In[201]:


X = df[['age','bmi', 'children', 'smoker_no', 'smoker_yes','sex_female', 'sex_male']]
y = df[['charges']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = model.score(X_test,y_test)

print(f"Visualização dos coeficientes: {model.intercept_}")
print(f"Visualização da inclinação da reta: {model.coef_}")
print(f"O Modelo possui um Coeficiente R^2 de: {score:.2f}")

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {round(mse,2)}')


# - No modelo acima, buscamos o preço com base nas variáveis: <b>age, bmi, children, smoker_no, smoker_yes, sex_female, sex_male</b>

# - <b>Bellow, Visualize the actual vs. predicted values after filtering</b>

# In[193]:


plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--', color='red')
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.title('Actual vs. Predicted Charges (After Filtering)')
plt.grid(True)
plt.show()


# - <b>Graph to visualize waste</b>

# In[191]:


visualizador = ResidualsPlot(model)
visualizador.fit(X, y)
visualizador.poof()


# In[168]:


score_teste = model.score(X_test,y_test)
score_teste


# - <b>Below, Learning Curve Graphs (Learning Curve) View Actual vs. Predicted Values ​​After Filtering</b>

# In[195]:


train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=10)

plt.figure(figsize=(10,6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='r', label='Traning score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color='g', label='Cross-validation score')
plt.title('Learning Curves')
plt.xlabel('Training examples')
plt.ylabel('Score')
plt.legend(loc='best')
plt.grid()
plt.show()


# In[ ]:





# In[ ]:




