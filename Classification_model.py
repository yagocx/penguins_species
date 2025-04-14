import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score

penguin = sns.load_dataset('penguins')

# Tratando dados nulos

# para colunas categóricas, descarte-as
penguin = penguin.dropna(subset=['species', 'island', 'sex']).copy()    
    #subset analisa apenas as colunas especificadas
    #.copy evita o aparecimento de um SettingWithCopyWarning

# para colunas numéricas, substitua pela mediana
numeric_columns = penguin.select_dtypes(include=['float64', 'int64']).columns

for col in numeric_columns:
    penguin.loc[:, col] = penguin[col].fillna(penguin[col].median())

# Padronizando variáveis numéricas
cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']

for col in cols:
    penguin.loc[:, f'{col}_std'] = (penguin[col] - penguin[col].mean()) / penguin[col].std()

# Padronizando variáveis categóricas

penguin.loc[:, 'island_Torgersen_nom'] = penguin['island'].apply(lambda island: 1 if island == 'Torgersen' else 0)
penguin.loc[:, 'island_Dream_nom'] = penguin['island'].apply(lambda island: 1 if island == 'Dream' else 0)
penguin.loc[:, 'island_Biscoe_nom'] = penguin['island'].apply(lambda island: 1 if island == 'Biscoe' else 0)

penguin.loc[:, 'sex_nom'] = penguin['sex'].apply(lambda sex: 1 if sex == 'M' else 0)

penguin = penguin.drop(['island', 'sex'], axis=1)

# Fazendo limpeza de dados
final_data_penguin = ['species'] + [col for col in penguin.columns if col.endswith(('_std', '_nom', '_ord'))] 
penguin = penguin[final_data_penguin]

print(penguin.head())

# Dividindo os dados em treino e teste
predictors_train, predictors_test, target_train, target_test = train_test_split(
	penguin.drop(['species'], axis=1),
	penguin['species'],
	test_size=1/3,
	random_state=123 #número qualquer
)

# Visualizando a Árvore
model = DecisionTreeClassifier()
model = model.fit(predictors_train, target_train)

plt.figure(figsize=(16,10))
plot_tree(model, filled=True, feature_names=predictors_train.columns, class_names=model.classes_)
#plt.show()
'''
A árvore treinada possui 8 folhas
'''

# Avaliação do modelo
target_predicted = model.predict(predictors_test) #25% dos dados totais

cm = confusion_matrix(target_test, target_predicted)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
#plt.show()
'''
A matriz de confusão mostra que o modelo tem um bom desempenho, com a maioria das previsões corretas. 
No entanto, há algumas confusões entre as classes, especialmente entre 'Adelie' e 'Chinstrap', e entre 'Chinstrap' e 'Gentoo'. 
Isso pode indicar que as características usadas para treinar o modelo não são suficientes para distinguir perfeitamente essas classes.
'''

# Determinando a acurácia
accuracy = accuracy_score(target_test, target_predicted)
print(f'{round(100 * accuracy, 2)}%') 

# Testando o modelo
penguin_test_df = sns.load_dataset('penguins')

bill_length_test = (38.2 - penguin_test_df['bill_length_mm'].mean()) / penguin_test_df['bill_length_mm'].std()
bill_depth_test = (18.1 - penguin_test_df['bill_depth_mm'].mean()) / penguin_test_df['bill_depth_mm'].std()
flipper_length_test = (185.0 - penguin_test_df['flipper_length_mm'].mean()) / penguin_test_df['flipper_length_mm'].std()
body_mass_test = (3950.0 - penguin_test_df['body_mass_g'].mean()) / penguin_test_df['body_mass_g'].std()

test_penguin = pd.DataFrame({
    'bill_length_mm_std': [bill_length_test],
    'bill_depth_mm_std': [bill_depth_test],
    'flipper_length_mm_std': [flipper_length_test],
    'body_mass_g_std': [body_mass_test],
    'island_Torgersen_nom': [0],
    'island_Dream_nom': [0],
    'island_Biscoe_nom': [1],
    'sex_nom': [1]
})

predicted_species = model.predict(test_penguin)
print(predicted_species) #predicted_body_mass = Adelie