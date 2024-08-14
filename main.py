import scipy.io as scio
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import mutual_info_classif
import random

def chunk_list(input_list, chunk_size):
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]


def init_PSO(pN, dim):
    X = np.zeros((pN, dim))  
    pbesti = np.zeros(dim)  
    p_fit = np.zeros(pN)  
    for i in range(pN): 
        for j in range(dim):  
            r = np.random.uniform(0,1)
            if r > 0.67:
                X[i][j] = 1
            else:
                X[i][j] = 0
    for j in range(dim):
        r = np.random.uniform(0,1)
        if r > 0.67:
            pbesti[j] = 1
        else:
            pbesti[j] = 0
    gbest = pbesti
    return X, pbesti, gbest, p_fit
def chaotic_jump_probability(stagnation_i):
    return 1 / (1 + np.exp(2 - stagnation_i))

def chaotic_sequence(k):
    if k == 0:
        return 0.13
    elif k >= 1:
        return 4 * chaotic_sequence(k - 1) * (1 - chaotic_sequence(k - 1))

def divide_into_neighborhoods(pi, pbesti, T):
    pi_list = chunk_list(pi, T)
    pbesti_list = chunk_list(pbesti, T)
    return pi_list, pbesti_list

def dir(i, N):
    if i <= N / 2:
        return 1
    else:
        return 0

def check_remainder_not_zero(features_number,T):
    if features_number % T != 0:
        return True
    else:
        return False
def feature_selected(pbesti, k, T):
    ai = []
    for i in range(T):
        ai.append(pbesti[i + (k-1) * T])
    return ai
def calculating_accuracy(xi):
    file_path = r'your dataset'
    X_scaled, y = load_and_prepare_data(file_path)
    mean_accuracy = KNN_with_cross_validation(X_scaled, y, xi)
    return mean_accuracy
def update_mechanism(xi, pbesti, gbest, Pcj_i, d, i, FEs, stagnation):
    xi = np.array(xi)
    r = np.random.rand()
    if r > Pcj_i:
        xi[d] = np.random.normal((pbesti[d] + gbest[d]) / 2, abs(pbesti[d] - gbest[d]))
        if xi[d] > 0.5: 
            xi[d] = 1
        else:
            xi[d] = 0
    else:
        zk = chaotic_sequence(FEs)
        xi[d] = pbesti[d] * (1 + (2 * zk - 1))
        if xi[d] > 0.5:
            xi[d] = 1
        else:
            xi[d] = 0
    if len(stagnation) <= i:
        stagnation += [0] * (i - len(stagnation) + 1)
    if calculating_accuracy(xi) > calculating_accuracy(pbesti):
        pbesti = xi
        stagnation[i]=0
    else:
        stagnation[i] = stagnation[i]+1
    return xi[d], stagnation, pbesti
# Input: The ith particle pi to be updated, the total number of features D, the number of features T in feature neighborhood, the
# historical optimal position pbesti of pi, the position xi of pi
# Output: The updated particle pi

def update(pi, D, T, pbesti, gbest, N, i, FEs, direction):
    p_besti = pbesti
    stagnation = []
    for r in range(N):
        stagnation.append(0)
    for j in range(D):
        # Calculate neighborhood index
        k = int(j / T) + 1
        # Check direction
        if direction[i] == 1:
            # If no feature in neighborhood k is selected in pbesti
            if not any(feature_selected(p_besti, k, T)):
                pi[j] = p_besti[j]
                continue
        elif direction[i] == 0:
            # If all features in neighborhood k are selected in pbesti
            if all(feature_selected(p_besti, k, T)):
                pi[j] = p_besti[j]
                continue
        # Use any chosen position update mechanism to update xi,j
        Pcj_i = chaotic_jump_probability(stagnation[i])
        pi[j], stagnation, p_besti = update_mechanism(pi, p_besti, gbest, Pcj_i, i, j, FEs, stagnation)
        for m in range(len(pi)):
            if pi[m] > 0.5:
                pi[m] = 1
            else:
                 pi[m] = 0
        for n in range(len(p_besti)):
            if p_besti[n] > 0.5:
                p_besti[n] = 1
            else:
                p_besti[n] = 0
    return pi, p_besti
def evaluate_fitness(xi, pbesti):
    xi_fitness_value = calculate_accuracy(xi)
    pbesti_fitness_value = calculate_accuracy(pbesti)
    if xi_fitness_value > pbesti_fitness_value:
        pbesti = xi
    return pbesti
def evaluate_feature_number(xi):
    xi_feature_number = np.sum(xi == 1)
    return xi_feature_number

def update_gbest(swarm, gbest_t, fitness_values, N, gbest_accuracy, gbest_feature_number):
    # Update the global best solution gbest based on fitness values
    # Replace with actual implementation
    for i in range(N):
        if fitness_values[i] > gbest_accuracy:
            gbest_accuracy = fitness_values[i]
            gbest = swarm[i]
        elif fitness_values[i] == gbest_accuracy and evaluate_feature_number(swarm[i]) < gbest_feature_number:
            gbest_accuracy = fitness_values[i]
            gbest = swarm[i]
        else:
            gbest = gbest_t
        gbest_feature_number = evaluate_feature_number(gbest)
    return gbest, gbest_accuracy, gbest_feature_number

def calculate_AI(subswarm, pbest_history, generation, window_size):
    size_subswarm = subswarm
    sum_ai = 0.0

    for particle_index in range(len(size_subswarm)):
        if generation > window_size:
            pbest_index = pbest_history.iloc[particle_index, generation]
            pbest_w_index = pbest_history.iloc[particle_index, (generation - window_size)]
            fitness_difference = pbest_index - pbest_w_index
            sum_ai += fitness_difference
        else:
            sum_ai = pbest_index

    ai = (1.0 / size_subswarm) * sum_ai
    return ai


def calculate_AF(subswarm, pbest_history, generation):
    size_subswarm = subswarm
    sum_af = 0.0

    for particle_index in range(len(size_subswarm)):
        pbest_index = pbest_history.iloc[particle_index,generation]  # Use the latest generation pbest
        sum_af += pbest_index

    af = (1.0 / size_subswarm) * sum_af
    return af

def change_direction(direction, pbest_history, generation, window_size):
    import random
    nl = direction.count(1)
    nu = direction.count(0)
    print(nl)
    print(nu)
    if nl == 0 or nu == 0:
        return direction
    AIl = calculate_AI(nl, pbest_history, generation, window_size)
    AIu = calculate_AI(nu, pbest_history, generation, window_size)
    if AIl < AIu:
        indices = [i for i, value in enumerate(direction) if value == 1]
        pr = np.random.choice(indices)
        direction[pr] = 0
    elif AIl > AIu:
        indices = [i for i, value in enumerate(direction) if value == 0]
        pr = np.random.choice(indices)
        direction[pr] = 1
    if AIl == AIu:
        AFl = calculate_AF(nl, pbest_history, generation)
        AFu = calculate_AF(nu, pbest_history, generation)

        if AFl < AFu:

            indices = [i for i, value in enumerate(direction) if value == 1]
            pr = np.random.choice(indices)
            direction[pr] = 0
        elif AFl > AFu:
            indices = [i for i, value in enumerate(direction) if value == 0]
            pr = np.random.choice(indices)
            direction[pr] = 1

    return direction

def load_and_prepare_data(file_path):
    data = scio.loadmat(file_path)
    X = pd.DataFrame(data['X']).values 
    y = pd.DataFrame(data['Y']).values.ravel()  
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def KNN_with_cross_validation(X_scaled, y, xi):
    boolean_array = xi.astype(bool)
    X_selected = X_scaled[:, boolean_array]
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    scores = cross_val_score(knn_classifier, X_selected, y, cv=5)
    mean_accuracy = np.mean(scores)
    print("5倍交叉验证的平均分类精度:", mean_accuracy)
    return mean_accuracy
def calculate_accuracy(xi):
    file_path = r'your dataset'
    X_scaled, y = load_and_prepare_data(file_path)
    mean_accuracy = KNN_with_cross_validation(X_scaled, y, xi)
    print("选定特征的5倍交叉验证平均分类精度:", mean_accuracy)
    return mean_accuracy

def BDFF(D,T,MAX_FE,N,W,pbest_history,X_scaled,y):
    FEs=0
    # 第一步
    data = scio.loadmat(r'C:\Users\11741\Pycharm\pythonProject1\data\GLIOMA (1).mat')
    print(type(data))
    dic1 = data['X']
    dic2 = data['Y']
    df1 = pd.DataFrame(dic1)
    df2 = pd.DataFrame(dic2)
    dim=len(df1.columns)
    X, pbesti, gbest, p_fit = init_PSO(N, dim)
    gbest_accuracy = calculate_accuracy(gbest, X_scaled,y)
    gbest_feature_number = evaluate_feature_number(gbest)
    swarm = X
    target = df2
    # 第二步
    direction = []
    for i in range(N):
        if i <= N / 2:
            direction.append(1)
        else:
            direction.append(0)
    # 第三步
    # 第四步，使用SU策略，目前还不知道怎么搞
    neighborhoods = divide_into_neighborhoods(D, T)
    k=1
    while FEs < MAX_FE:
        for i in range(N):
            xi = swarm[i]
            pbesti = evaluate_fitness(xi,pbesti,X_scaled,y,)
            pbest_history.iloc[i,FEs] = calculate_accuracy(pbesti,X_scaled,y)
            xi, pbesti = update(xi,len(df1.columns),3,pbesti,gbest,N,i,FEs)
            swarm[i] = xi

        gbest, gbest_accuracy, gbest_feature_number = update_gbest(swarm, gbest,[calculate_accuracy(xi,X_scaled,y) for xi in swarm],N,gbest_accuracy,gbest_feature_number)
        print(np.sum(gbest == 1))
        if k % W == 0:
            direction = change_direction(direction, pbest_history, FEs, W)
        print(direction)

        k += 1
        FEs += 1 # Assuming each particle evaluation counts as one fitness evaluation

    return gbest

# examples
import numpy as np
import pandas as pd
import scipy.io as scio
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler


data = scio.loadmat(r'your dataset')
dic1 = data['X']
dic2 = data['Y']
df1 = pd.DataFrame(dic1)
df2 = pd.DataFrame(dic2)
feats = df1
labels = df2

MAX_FE = 250
N = 20
pb_history = [[None for _ in range(MAX_FE)] for _ in range(N)]
pbest_history = pd.DataFrame(pb_history)

gbest = BDFF(len(df1.columns), 3, 250, 20, 10, pbest_history)
