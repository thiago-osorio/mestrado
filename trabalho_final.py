import time
import random
import numpy as np

################################ Base de Dados ################################
# Referência Bibliográfica:
#   - https://arxiv.org/pdf/1704.07706

with open('raw_data.csv', 'r') as arquivo:
    linhas = arquivo.readlines()

dados = []
for i, linha in enumerate(linhas):
    if i == 0:
        continue
    linha = linha.strip()
    colunas = linha.split(',')
    dados.append(int(round(float(colunas[1])*1000, 0)))

############################### Árvore Binária ###############################
# Referência Bibliográfica: 
#   - Claude 3.5 Sonnet

class Node:
    def __init__(self, data):
        self.data = data
        self.leftChild = None
        self.rightChild = None

    def insert(self, data):
        current = self
        while True:
            if data < current.data:
                if current.leftChild:
                    current = current.leftChild
                else:
                    current.leftChild = Node(data)
                    break
            else:
                if current.rightChild:
                    current = current.rightChild
                else:
                    current.rightChild = Node(data)
                    break

    def search(self, target):
        current = self
        while current:
            if current.data == target:
                return True
            elif target < current.data:
                current = current.leftChild
            else:
                current = current.rightChild
        return False

# Inserção de dados na Árvore Binária
execs = []
for _ in range(100):
    ini = time.time()

    ab = Node(dados[0])
    for numero in dados[1:]:
        ab.insert(numero)

    end = time.time()
    time_diff = end-ini
    execs.append(time_diff)
media = round(np.mean(execs), 4)
desv_pad = round(np.std(execs), 4)
print(f'Tempo médio de inserção dos {len(dados)} números na Árvore Binária: {media} +/- {desv_pad} segundos')

# Busca na Árvore Binária
execs = []

ab = Node(dados[0])
for numero in dados[1:]:
    ab.insert(numero)

for _ in range(100):
    ini = time.time()

    ab.search(random.randint(0, 1000000))

    end = time.time()
    time_diff = end-ini
    execs.append(time_diff)
media = round(np.mean(execs), 4)
desv_pad = round(np.std(execs), 4)
print(f'Tempo médio de busca na Árvore Binária: {media} +/- {desv_pad} segundos')

######################## Tabela Hash com Encadeamento ########################
# Referência Bibliográfica:
#   - Claude 3.5 Sonnet

class HashTable:
    def __init__(self, size):
        self.size = size
        self.table = [[] for _ in range(size)]

    def hash_function(self, number):
        return number % self.size

    def insert(self, number):
        hash_index = self.hash_function(number)
        if number not in self.table[hash_index]:
            self.table[hash_index].append(number)

    def search(self, number):
        hash_index = self.hash_function(number)
        return number in self.table[hash_index]

# Inserção de dados na Tabela Hash
execs = []
for _ in range(100):
    ini = time.time()

    hash_table = HashTable(10000)
    for numero in dados:
        hash_table.insert(numero)
    
    end = time.time()
    time_diff = end-ini
    execs.append(time_diff)
media = round(np.mean(execs), 4)
desv_pad = round(np.std(execs), 4)
print(f'Tempo médio de inserção dos {len(dados)} números na Tabela Hash: {media} +/- {desv_pad} segundos')


# Busca na Tabela Hash
execs = []

hash_table = HashTable(10000)
for numero in dados:
    hash_table.insert(numero)

for _ in range(100):
    ini = time.time()

    hash_table.search(random.randint(0, 1000000))
    
    end = time.time()
    time_diff = end-ini
    execs.append(time_diff)
media = round(np.mean(execs), 4)
desv_pad = round(np.std(execs), 4)
print(f'Tempo médio de busca na Tabela Hash: {media} +/- {desv_pad} segundos')
