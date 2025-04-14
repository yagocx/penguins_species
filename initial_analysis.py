import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

penguin = sns.load_dataset('penguins')

with sns.axes_style('whitegrid'):
  grafico = sns.pairplot(data=penguin.drop(['sex', 'island'], axis=1), hue="species", palette="pastel")

plt.show()
'''
Espécie Adelie:
    Possuem proporções menores, com exceção do comprimento do bico (significativamente maiores)
    Tende a se assemelhar à espécie Chinstrap, sendo seu principal diferenciador à espécie o comprimento do bico
    Significativamente diferentes da espécie Gentoo em todos os aspectos
    
Espécie Chinstrap: 
    Com exceção do comprimento do bico, é bastante similar à Adelie
    Se considerar apenas o comprimento do bico, pode ser bastante similar a Gentoo
    Tendem a ter bicos maiores, mas proporções menores

Espécie Gentoo
    É a espécie com proporções maiores
    Único atributo menor que a das outras espécies é a largura do bico
    Tende a ser bastante diferente das outras espécies
'''
with sns.axes_style('whitegrid'):
  grafico = sns.countplot(data=penguin, x='sex', hue="species", palette="pastel")

#plt.show()
'''
sexo dos animais, independente da espécie, tende a ser equilibrado
'''
with sns.axes_style('whitegrid'):
  grafico = sns.countplot(data=penguin, x='island', hue="species", palette="pastel")

#plt.show()
'''
Na ilha Torgersen, só existe a espécie Adelie
Na ilha Dream, não existe a espécie Gentoo 
Na ilha Biscoe, não existe a espécie Chinstrap e a Gentoo é bem mais abundante que a Adelie
'''