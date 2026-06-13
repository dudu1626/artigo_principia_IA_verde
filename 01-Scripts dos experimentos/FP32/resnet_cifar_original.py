"""
# ResNet34 para CIFAR10 - Original
"""

# # biblioteca necessária para trabalhar com fp16 (dependências do Torch)
# !pip install apex

# # interface Python para funções de gerenciamento e monitoramento de GPU.
# !pip install nvidia-ml-py

# # Biblioteca para monitoramento abstração de nvidia-ml-py
# !pip install pynvml

# Importando bibliotecas
## Básicas
import io
import os
import random
import sys
from datetime import datetime

import matplotlib.pyplot as plt
## Manipulação de dados
import numpy as np
import pandas as pd
## Medição GPU
import pynvml
## Visualização
import seaborn as sns
## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from carbontracker import parser
## CarbonTracker
from carbontracker.tracker import CarbonTracker
## Métricas de avaliação
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
# Medição flopse parâmetros
from thop import profile
## Quantização
from torch.amp import GradScaler, autocast
## Dados
from torch.utils.data import DataLoader, random_split
## Outros
from torchsummary import summary
## Visão computacional
from torchvision import datasets, transforms


## Definindo semente
def set_seed(seed=158763):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

## Configurações
plt.style.use("seaborn-v0_8")
sns.set_theme()

## Seed
set_seed()

## Verificando disponibilidade de GPU
dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Dispositivo em uso: {dispositivo}')

## Carregamento e preparação dos dados
transformacoes = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                     transforms.RandomHorizontalFlip(),
                                     #transforms.RandomRotation(10), # testar se melhora
                                     ])

dados_treino_completo = datasets.CIFAR10(root='./dados', train=True, download=True, transform=transformacoes)
dados_teste = datasets.CIFAR10(root='./dados', train=False, download=True, transform=transformacoes)

classes = ('airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

## Divisão dos dados para treino, validação e teste
tamanho_treino = int(0.8 * len(dados_treino_completo))
tamanho_validacao = len(dados_treino_completo) - tamanho_treino

dados_treino, dados_validacao = random_split(dados_treino_completo, [tamanho_treino, tamanho_validacao])

print(f'Tamanho dos Dados de treino: {len(dados_treino)}')
print(f'Tamanho dos Dados de validação: {len(dados_validacao)}')
print(f'Tamanho dos Dados de teste: {len(dados_teste)}')

## Carregamento dos dados
tamanho_batch = 32
num_nucleos = min(4, torch.get_num_threads())  # Ajuste baseado na máquina
print(f'Número de núcleos: {num_nucleos}')

treino_loader = DataLoader(dados_treino, batch_size=tamanho_batch, shuffle=True, num_workers=num_nucleos)
validacao_loader = DataLoader(dados_validacao, batch_size=tamanho_batch, shuffle=False, num_workers=num_nucleos)
teste_loader = DataLoader(dados_teste, batch_size=tamanho_batch, shuffle=False, num_workers=num_nucleos)

# Definição do modelo
# Bloco Residual (sem atalho para entradas com mesma dimensão)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout1 = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout2 = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        out = self.bn2(self.conv2(out))
        out = self.dropout2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# Implementação da ResNet-34
class ResNet34(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, dropout_rate=0, width_multiplier=1):
        super(ResNet34, self).__init__()

        self.in_channels = int(64 * width_multiplier)

        # Camada inicial com multiplicador de largura
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)  # Ajustado para CIFAR-10
        self.bn1 = nn.BatchNorm2d(self.in_channels)

        # Camadas com número de canais ajustado
        self.layer1 = self._make_layer(block, int(64 * width_multiplier), num_blocks[0], stride=1,
                                       dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(block, int(128 * width_multiplier), num_blocks[1], stride=2,
                                       dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(block, int(256 * width_multiplier), num_blocks[2], stride=2,
                                       dropout_rate=dropout_rate)
        self.layer4 = self._make_layer(block, int(512 * width_multiplier), num_blocks[3], stride=2,
                                       dropout_rate=dropout_rate)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(int(512 * width_multiplier), num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride, dropout_rate=0):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, dropout_rate))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        #out = self.linear(out)
        out = self.fc(out)

        return out

modelo = ResNet34(block=ResidualBlock, num_blocks=[2,2,2,2]).to(dispositivo)

print(modelo)

summary(modelo, (3, 32, 32))

## Definição da função de custo e otimizador
criterio = nn.CrossEntropyLoss()
otimizador = optim.Adam(modelo.parameters(), lr=0.001) # alterar a taxa de aprendizado no futuro

## Número de épocas
epocas = 50

# tracker = CarbonTracker(epochs=epocas, components="gpu", monitor_epochs=-1, interpretable=True,
#                         log_dir=f"./{diretorio_carbon}/", log_file_prefix="cbt")

tracker = CarbonTracker(epochs=epocas)

# medição de energia
potencias_treino = []
tempos_treino = []

# Inicializa o NVML para monitoramento da GPU
pynvml.nvmlInit()

## Função de treinamento e validação
def treinar_e_validar(modelo, treino_loader, validacao_loader, criterio, otimizador, epocas=50, nome_modelo='modelo.pth'):
    # ativar o modo de treinamento
    modelo.train()

    # inicio marcação tempo
    tempo_inicio = datetime.now()
    # inicio marcação tracker
    tracker.epoch_start()

    melhor_acuracia = 0.0
    melhor_epoca = 0

    for epoca in range(epocas):
        # tracker de épocas
        tracker.epoch_start()

        # vars para treino
        perda_acumulada = 0.0
        acertos = 0.0
        total = 0.0

        for dados in treino_loader:
            imagens, rotulos = dados
            imagens, rotulos = imagens.to(dispositivo), rotulos.to(dispositivo)

            otimizador.zero_grad()

            saidas = modelo(imagens)
            perda = criterio(saidas, rotulos)
            perda.backward()
            otimizador.step()

            perda_acumulada += perda.item()

            _, preditos = torch.max(saidas, 1)
            total += rotulos.size(0)
            acertos += (preditos == rotulos).sum().item()

            # Medir o consumo de energia
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetPowerUsage(handle)
            consumo_energia = info / 1000.0
            potencias_treino.append(consumo_energia)

        perda_treino = perda_acumulada / len(treino_loader)
        acuracia_treino = acertos / total
        print(f'Época {epoca + 1}/{epocas}\nPerda Treino: {perda_treino:.4f} - Acurácia Treino: {acuracia_treino:.4f}')

        # Validação
        #ativar o modo de avaliação
        modelo.eval()

        # vars para validação
        perda_validacao = 0.0
        acertos = 0.0
        total = 0.0

        with torch.no_grad():
            for dados in validacao_loader:
                imagens, rotulos = dados
                imagens, rotulos = imagens.to(dispositivo), rotulos.to(dispositivo)

                saidas = modelo(imagens)
                perda = criterio(saidas, rotulos)

                perda_validacao += perda.item()

                _, preditos = torch.max(saidas, 1)
                acertos += (preditos == rotulos).sum().item()
                total += rotulos.size(0)

            perda_validacao = perda_validacao / len(validacao_loader)
            acuracia_validacao = acertos / total
            print(f'Perda Validação: {perda_validacao:.4f} - Acurácia Validação: {acuracia_validacao:.4f}')

        # salvar o melhor modelo
        if acuracia_validacao > melhor_acuracia:
            melhor_acuracia = acuracia_validacao
            melhor_epoca = epoca + 1
            torch.save(modelo.state_dict(), nome_modelo)

            print(f"Melhor época: {melhor_epoca} - Melhor acurácia: {melhor_acuracia}")

    # final de medição de energia e tempo
    tempo_fim = datetime.now()
    tempo_treino = (tempo_fim - tempo_inicio)
    tempos_treino.append(tempo_treino.total_seconds())
    tracker.epoch_end()

    return perda_treino, acuracia_treino, perda_validacao, acuracia_validacao, tempo_treino, consumo_energia

# Uso da biblioteca thop - Medição de FLOPs e parâmetros do modelo
entrada = torch.randn(1, 3, 32, 32).to(dispositivo)
flops, parametros = profile(modelo, inputs=(entrada,), verbose=False)

print(f'FLOPs: {flops}')
print(f'Parâmetros: {parametros}')

## Treinamento e validação
nome_modelo = 'ResNet34_CIFAR10.pth'
perda_treino, acuracia_treino, perda_validacao, acuracia_validacao, tempo_treino, consumo_energia = treinar_e_validar(modelo, treino_loader, validacao_loader, criterio, otimizador, epocas, nome_modelo)

# Calcular a média dos tempos de treino e consumo de energia
media_tempo_treino = np.mean(tempos_treino)
media_consumo_energia = np.mean(potencias_treino)
print(f'Tempo Médio de Treino: {media_tempo_treino} segundos')
print(f'Consumo Médio de Energia: {media_consumo_energia} W')

## Carregamento do melhor modelo, por causa do Thorp, houve a necessidade de manipulação de filtragem dos metadados

# Carregar o state_dict salvo
state_dict = torch.load(nome_modelo)

# Remover chaves inesperadas
chaves_inesperadas = ["layer1.0.total_ops", "layer1.0.total_params", "layer1.0.dropout1.total_ops", 
                      "layer1.0.dropout1.total_params", "layer1.0.dropout2.total_ops", "layer1.0.dropout2.total_params", 
                      "layer1.1.total_ops", "layer1.1.total_params", "layer1.1.dropout1.total_ops", 
                      "layer1.1.dropout1.total_params", "layer1.1.dropout2.total_ops", "layer1.1.dropout2.total_params", 
                      "layer2.0.total_ops", "layer2.0.total_params", "layer2.0.dropout1.total_ops", 
                      "layer2.0.dropout1.total_params", "layer2.0.dropout2.total_ops", "layer2.0.dropout2.total_params", 
                      "layer2.1.total_ops", "layer2.1.total_params", "layer2.1.dropout1.total_ops", 
                      "layer2.1.dropout1.total_params", "layer2.1.dropout2.total_ops", "layer2.1.dropout2.total_params", 
                      "layer3.0.total_ops", "layer3.0.total_params", "layer3.0.dropout1.total_ops", 
                      "layer3.0.dropout1.total_params", "layer3.0.dropout2.total_ops", "layer3.0.dropout2.total_params", 
                      "layer3.1.total_ops", "layer3.1.total_params", "layer3.1.dropout1.total_ops", 
                      "layer3.1.dropout1.total_params", "layer3.1.dropout2.total_ops", "layer3.1.dropout2.total_params", 
                      "layer4.0.total_ops", "layer4.0.total_params", "layer4.0.dropout1.total_ops", 
                      "layer4.0.dropout1.total_params", "layer4.0.dropout2.total_ops", "layer4.0.dropout2.total_params", 
                      "layer4.1.total_ops", "layer4.1.total_params", "layer4.1.dropout1.total_ops", 
                      "layer4.1.dropout1.total_params", "layer4.1.dropout2.total_ops", "layer4.1.dropout2.total_params",
                      "total_ops", "total_params"]
state_dict = {k: v for k, v in state_dict.items() if k not in chaves_inesperadas}

# Criar instância do modelo
melhor_modelo = ResNet34(block=ResidualBlock, num_blocks=[2,2,2,2])

# Carregar os pesos filtrados
melhor_modelo.load_state_dict(state_dict)
melhor_modelo.to(dispositivo)

# Avaliação do modelo
potencias_inferencia = []

## Avaliação do modelo
def inferencia_e_metricas(modelo, teste_loader):
    # ativar o modo de avaliação
    modelo.eval()

    # Medição de tempo de inferência - Início
    inicio_tempo_teste = datetime.now()

    rotulos_reais = []
    rotulos_preditos = []

    with torch.no_grad():
        for dados in teste_loader:
            imagens, rotulos = dados
            imagens, rotulos = imagens.to(dispositivo), rotulos.to(dispositivo)

            saidas = modelo(imagens)

            _, preditos = torch.max(saidas, 1)

            rotulos_reais.extend(rotulos.cpu().numpy())
            rotulos_preditos.extend(preditos.cpu().numpy())

            # Medir o consumo de energia
            handle_inferencia = pynvml.nvmlDeviceGetHandleByIndex(0)
            info_inferencia = pynvml.nvmlDeviceGetPowerUsage(handle_inferencia)
            consumo_energia_inferencia = info_inferencia / 1000.0
            potencias_inferencia.append(consumo_energia)

    # Medição de tempo de inferência - Fim
    fim_tempo_teste = datetime.now()

    acuracia = accuracy_score(rotulos_reais, rotulos_preditos)
    precisao = precision_score(rotulos_reais, rotulos_preditos, average='weighted')
    recall = recall_score(rotulos_reais, rotulos_preditos, average='weighted')
    f1 = f1_score(rotulos_reais, rotulos_preditos, average='weighted')

    tempo_inferencia = (fim_tempo_teste - inicio_tempo_teste).total_seconds()

    media_consumo_energia_inferencia = np.mean(potencias_inferencia)

    print(f'Acurácia: {acuracia:.4f}')
    print(f'Precisão: {precisao:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1: {f1:.4f}')
    print(f'Tempo de Inferência: {(fim_tempo_teste - inicio_tempo_teste).total_seconds()} segundos')
    print(f'Consumo Médio de Energia: {media_consumo_energia_inferencia:.4f} W')

    # Matriz de confusão
    matriz_confusao = confusion_matrix(rotulos_reais, rotulos_preditos)
    plt.figure(figsize=(10, 7))
    sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão')
    plt.show()

    return acuracia, precisao, recall, f1, matriz_confusao, tempo_inferencia, media_consumo_energia_inferencia

acuracia, precisao, recall, f1, matriz_confusao, tempo_inferencia, media_consumo_energia_inferencia = inferencia_e_metricas(melhor_modelo, teste_loader)

# Finalizar o monitoramento da GPU
pynvml.nvmlShutdown()   

# Finalizar o CarbonTracker
tracker.stop()

# Salvar as métricas
metricas = {
    "Nome do Modelo": nome_modelo.split(".")[0],
    "FLOPs": flops,
    "Parâmetros": parametros,
    "Perda Treino": perda_treino,
    "Acurácia Treino": acuracia_treino,
    "Perda Validação": perda_validacao,
    "Acurácia Validação": acuracia_validacao,
    "Tempo Médio de Treino": media_tempo_treino,
    "Consumo Médio de Energia Treino": media_consumo_energia,
    "Acurácia": acuracia,
    "Precisão": precisao,
    "Recall": recall,
    "F1": f1,
    "Tempo de Inferência": tempo_inferencia,
    "Consumo Médio de Energia Inferência": media_consumo_energia
}

metricas_df = pd.DataFrame(metricas, index=[0])
metricas_df.to_csv(f"metricas_{nome_modelo.split(".")[0]}.csv", index=False)
