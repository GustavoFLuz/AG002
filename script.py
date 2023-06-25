### Conexão com MySQL
from pymysql import connect

### Manipulação de dados
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score 

# Coleta de dados
data_base = connect(host="127.0.0.1", user="root", passwd="1234", database="ag002")
query = "SELECT * from `breast-cancer`"
pd_dataframe = pd.read_sql(query, data_base)

length = pd_dataframe.shape[0]

# Separação dos dados
predict_data = pd_dataframe.loc[:,["age", "menopause", "tumor-size", "inv-nodes", "node-caps", "deg-malig", "breast", "breast-quad", "irradiat"]]
predict_data = np.array(predict_data)

target_data = pd_dataframe["class"]
target_data = np.array(target_data)

train_size = 0.8

predict_train, predict_test, target_train, target_test = train_test_split(predict_data, target_data, train_size=train_size, random_state = 0)

# Normalização dos dados
sc = StandardScaler()
sc.fit(predict_train)
predict_train_std = sc.transform(predict_train)
predict_test_std = sc.transform(predict_test)

# Treinamento
perceptron = Perceptron(max_iter=50000, eta0=0.1, random_state=0)
perceptron.fit(predict_train_std, target_train)

# Testes 
target_predict = perceptron.predict(predict_test_std)

print(f"Quantidade de Testes: {len(target_predict)}")
print(f"Acertos: {accuracy_score(target_test, target_predict, normalize=False)}")
print(f"Erros: {len(target_predict)-accuracy_score(target_test, target_predict, normalize=False)}")
print("Precisao: {0:.2f}%".format(accuracy_score(target_test, target_predict)*100))

# Entrada de mais dados
print("Digite os dados para prever o resultado: ")
while(True):
    input_data = np.array(input("Dados: ").split(" "))
    if(len(input_data) == 1):
        break

    if(len(input_data) != 9):
        print("Dados inválidos")
        continue

    predict = [input_data.astype(int)]
    predict_std = sc.transform(predict)
    target_predict = perceptron.predict(predict_std)

    if(target_predict[0] == 1):
        print("Resultado: Não")
    else:
        print(f"Resultado: Sim")

print("Fim do programa")

# 5 3 6 1 1 1 1 2 1 => Não
# 4 3 4 1 1 2 1 1 1 => Sim