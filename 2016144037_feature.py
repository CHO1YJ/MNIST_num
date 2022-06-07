# # Setting Module
import math as ma
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 기대값 계산 - E[x] = np.sum(x * p_x)
# 분산 계산 - Var[x] = np.sum(np.pow(x - E_x, 2) * p_x)

#################################################################################
def feature_1(input_data):
    # 특징 후보 1번: 가로축 Projection => 확률밀도함수로 변환 => 기댓값    
    output_value = 0
    sum_input_data = []
    E_x = 0
    for n in range(input_data.shape[1]):
        sum_input_data.append(np.sum(input_data[:, n]))
    sum_input_data = np.array(sum_input_data) / np.sum(sum_input_data)
    for x in range(input_data.shape[1]):
        E_x = E_x + sum_input_data[x] * x
    output_value = E_x
    return output_value
#################################################################################

#################################################################################
def feature_2(input_data):
    # 특징 후보 2번: 가로축 Projection => 확률밀도함수로 변환 => 분산
    output_value = 0
    sum_input_data = []
    E_x = 0
    Var_x = 0
    for n in range(input_data.shape[1]):
        sum_input_data.append(np.sum(input_data[:, n]))
    sum_input_data = np.array(sum_input_data) / np.sum(sum_input_data)
    for x in range(input_data.shape[1]):
        E_x = E_x + sum_input_data[x] * x
    for x in range(input_data.shape[1]):
        Var_x = Var_x + np.power(sum_input_data[x] - E_x, 2) * sum_input_data[x]
    output_value = Var_x
    return output_value
#################################################################################

#################################################################################
def feature_3(input_data):
    # 특징 후보 5번: Diagonal 원소 배열 추출 => 확률밀도함수로 변환 => 기댓값
    output_value = 0
    E_x = 0
    sum_input_data = np.diag(input_data)
    sum_input_data = np.array(sum_input_data) / np.sum(sum_input_data)
    for x in range(input_data.shape[0]):
        E_x = E_x + sum_input_data[x] * x
    output_value = E_x
    return output_value
#################################################################################

#################################################################################
def feature_4(input_data):
    # 특징 후보 6번: Diagonal 원소 배열 추출 => 확률밀도함수로 변환 => 분산
    output_value = 0
    E_x = 0
    Var_x = 0
    sum_input_data = np.diag(input_data)
    sum_input_data = np.array(sum_input_data) / np.sum(sum_input_data)
    for x in range(input_data.shape[0]):
        E_x = E_x + sum_input_data[x] * x
    for x in range(input_data.shape[0]):
        Var_x = Var_x + np.power(sum_input_data[x] - E_x, 2) * sum_input_data[x]
    output_value = Var_x                                   
    return output_value
#################################################################################

#################################################################################
def feature_5(input_data):
    # 특징 후보 7번: Diagonal 원소 배열 추출 => 0의 개수
    output_value = 0
    sum_input_data = []
    count_num = 0
    sum_input_data = np.diag(input_data)
    for n in range(input_data.shape[0]):
        if sum_input_data[n] == 0:
            count_num = count_num + 1
    output_value = count_num   
    return output_value
#################################################################################

# 숫자 0에 대한 데이터를 생성
x_0_set = np.array([], dtype='float32')
x_0_set = np.resize(x_0_set, (0, 5)) # 배열을 쌓기 위해서 size를 맞춰줌
for i in range(1, 501): # 숫자 이미지 0에 대해서 500개의 Training set을 특징 set으로 변환
    temp_name_0 = '0_' + str(i) + '.csv'
    temp_image_0 = pd.read_csv('Data_Base/' + temp_name_0, header=None)
    temp_image_0 = temp_image_0.to_numpy(dtype='float32')
    
    x0_0 = feature_1(temp_image_0)
    x1_0 = feature_2(temp_image_0)
    x2_0 = feature_3(temp_image_0)
    x3_0 = feature_4(temp_image_0)
    x4_0 = feature_5(temp_image_0)
    
    x_feature_0 = np.array([x0_0, x1_0, x2_0, x3_0, x4_0], dtype='float32') # 하나의 숫자 이미지를 5개의 특징 set으로
    x_feature_0 = np.resize(x_feature_0, (1, 5)) # 배열을 쌓기 위해서 size를 맞춰줌
    x_0_set = np.concatenate((x_0_set, x_feature_0), axis=0)

# 숫자 1에 대한 데이터를 생성
x_1_set = np.array([], dtype='float32')
x_1_set = np.resize(x_1_set, (0, 5)) # 배열을 쌓기 위해서 size를 맞춰줌
for i in range(1, 501): # 숫자 이미지 1에 대해서 500개의 Training set을 특징 set으로 변환
    temp_name_1 = '1_' + str(i) + '.csv'
    temp_image_1 = pd.read_csv('Data_Base/' + temp_name_1, header=None)
    temp_image_1 = temp_image_1.to_numpy(dtype='float32')
    
    x0_1 = feature_1(temp_image_1)
    x1_1 = feature_2(temp_image_1)
    x2_1 = feature_3(temp_image_1)
    x3_1 = feature_4(temp_image_1)
    x4_1 = feature_5(temp_image_1)
    
    x_feature_1 = np.array([x0_1, x1_1, x2_1, x3_1, x4_1], dtype='float32') # 하나의 숫자 이미지를 5개의 특징 set으로
    x_feature_1 = np.resize(x_feature_1, (1, 5)) # 배열을 쌓기 위해서 size를 맞춰줌
    x_1_set = np.concatenate((x_1_set, x_feature_1), axis=0)

# 숫자 2에 대한 데이터를 생성
x_2_set = np.array([], dtype='float32')
x_2_set = np.resize(x_2_set, (0, 5)) # 배열을 쌓기 위해서 size를 맞춰줌
for i in range(1, 501): # 숫자 이미지 2에 대해서 500개의 Training set을 특징 set으로 변환
    temp_name_2 = '2_' + str(i) + '.csv'
    temp_image_2 = pd.read_csv('Data_Base/' + temp_name_2, header=None)
    temp_image_2 = temp_image_2.to_numpy(dtype='float32')
    
    x0_2 = feature_1(temp_image_2)
    x1_2 = feature_2(temp_image_2)
    x2_2 = feature_3(temp_image_2)
    x3_2 = feature_4(temp_image_2)
    x4_2 = feature_5(temp_image_2)
    
    x_feature_2 = np.array([x0_2, x1_2, x2_2, x3_2, x4_2], dtype='float32') # 하나의 숫자 이미지를 5개의 특징 set으로
    x_feature_2 = np.resize(x_feature_2, (1, 5)) # 배열을 쌓기 위해서 size를 맞춰줌
    x_2_set = np.concatenate((x_2_set, x_feature_2), axis=0)













