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
        Var_x = Var_x + np.power(x - E_x, 2) * sum_input_data[x]
    output_value = Var_x
    return output_value
#################################################################################

#################################################################################
def feature_3(input_data):
    # 특징 후보 3번: 세로축 Projection => 확률밀도함수로 변환 => 기댓값
    output_value = 0
    sum_input_data = []
    E_x = 0
    for m in range(input_data.shape[0]):
        sum_input_data.append(np.sum(input_data[m, :]))
    sum_input_data = np.array(sum_input_data) / np.sum(sum_input_data)
    for x in range(input_data.shape[0]):
        E_x = E_x + sum_input_data[x] * x
    output_value = E_x
    return output_value
#################################################################################

#################################################################################
def feature_4(input_data):
    # 특징 후보 4번: 세로축 Projection => 확률밀도함수로 변환 => 분산
    output_value = 0
    sum_input_data = []
    E_x = 0
    Var_x = 0
    for m in range(input_data.shape[0]):
        sum_input_data.append(np.sum(input_data[m, :]))
    sum_input_data = np.array(sum_input_data) / np.sum(sum_input_data)
    for x in range(input_data.shape[0]):
        E_x = E_x + sum_input_data[x] * x
    for x in range(input_data.shape[0]):
        Var_x = Var_x + np.power(x - E_x, 2) * sum_input_data[x]
    output_value = Var_x                                  
    return output_value
#################################################################################

#################################################################################
def feature_5(input_data):
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
def feature_6(input_data):
    # 특징 후보 6번: Diagonal 원소 배열 추출 => 확률밀도함수로 변환 => 분산
    output_value = 0
    E_x = 0
    Var_x = 0
    sum_input_data = np.diag(input_data)
    sum_input_data = np.array(sum_input_data) / np.sum(sum_input_data)
    for x in range(input_data.shape[0]):
        E_x = E_x + sum_input_data[x] * x
    for x in range(input_data.shape[0]):
        Var_x = Var_x + np.power(x - E_x, 2) * sum_input_data[x]
    output_value = Var_x
    return output_value
#################################################################################

#################################################################################
def feature_7(input_data):
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

#################################################################################
def feature_8(input_data):
    # 특징 후보 8번: Anti-Diagonal 원소 배열 추출 => 기댓값
    output_value = 0
    E_x = 0
    sum_input_data = np.fliplr(input_data).diagonal()
    sum_input_data = np.array(sum_input_data) / np.sum(sum_input_data)
    for x in range(input_data.shape[0]):
        E_x = E_x + sum_input_data[x] * x
    output_value = E_x
    return output_value
#################################################################################

#################################################################################
def feature_9(input_data):
    # 특징 후보 9번: Anti-Diagonal 원소 배열 추출 => 분산
    output_value = 0
    E_x = 0
    Var_x = 0
    sum_input_data = np.fliplr(input_data).diagonal()
    sum_input_data = np.array(sum_input_data) / np.sum(sum_input_data)
    for x in range(input_data.shape[0]):
        E_x = E_x + sum_input_data[x] * x
    for x in range(input_data.shape[0]):
        Var_x = Var_x + np.power(x - E_x, 2) * sum_input_data[x]
    output_value = Var_x
    return output_value
#################################################################################

#################################################################################
def feature_10(input_data):
    # 특징 후보 10번: Anti-Diagonal 원소 배열 추출 => 0의 개수
    output_value = 0
    sum_input_data = []
    count_num = 0
    sum_input_data = np.fliplr(input_data).diagonal()
    for n in range(input_data.shape[0]):
        if sum_input_data[n] == 0:
            count_num = count_num + 1
    output_value = count_num                 
    return output_value
#################################################################################

# 숫자 0, 1, 2에 대한 데이터를 생성
x_0_set = np.array([], dtype='float32')
x_0_set = np.resize(x_0_set, (0, 10)) # 배열을 쌓기 위해서 size를 맞춰줌

x_1_set = np.array([], dtype='float32')
x_1_set = np.resize(x_1_set, (0, 10)) # 배열을 쌓기 위해서 size를 맞춰줌

x_2_set = np.array([], dtype='float32')
x_2_set = np.resize(x_2_set, (0, 10)) # 배열을 쌓기 위해서 size를 맞춰줌
for i in range(1, 501): # 숫자 이미지 0에 대해서 500개의 Training set을 특징 set으로 변환
    temp_name_0 = '0_' + str(i) + '.csv'
    temp_image_0 = pd.read_csv('Data_Base/' + temp_name_0, header=None)
    temp_image_0 = temp_image_0.to_numpy(dtype='float32')
    
    temp_name_1 = '1_' + str(i) + '.csv'
    temp_image_1 = pd.read_csv('Data_Base/' + temp_name_1, header=None)
    temp_image_1 = temp_image_1.to_numpy(dtype='float32')
    
    temp_name_2 = '2_' + str(i) + '.csv'
    temp_image_2 = pd.read_csv('Data_Base/' + temp_name_2, header=None)
    temp_image_2 = temp_image_2.to_numpy(dtype='float32')
    
    x0_0 = feature_1(temp_image_0)
    x1_0 = feature_2(temp_image_0)
    x2_0 = feature_3(temp_image_0)
    x3_0 = feature_4(temp_image_0)
    x4_0 = feature_5(temp_image_0)
    x5_0 = feature_6(temp_image_0)
    x6_0 = feature_7(temp_image_0)
    x7_0 = feature_8(temp_image_0)
    x8_0 = feature_9(temp_image_0)
    x9_0 = feature_10(temp_image_0)
    
    x0_1 = feature_1(temp_image_1)
    x1_1 = feature_2(temp_image_1)
    x2_1 = feature_3(temp_image_1)
    x3_1 = feature_4(temp_image_1)
    x4_1 = feature_5(temp_image_1)
    x5_1 = feature_6(temp_image_1)
    x6_1 = feature_7(temp_image_1)
    x7_1 = feature_8(temp_image_1)
    x8_1 = feature_9(temp_image_1)
    x9_1 = feature_10(temp_image_1)
    
    x0_2 = feature_1(temp_image_2)
    x1_2 = feature_2(temp_image_2)
    x2_2 = feature_3(temp_image_2)
    x3_2 = feature_4(temp_image_2)
    x4_2 = feature_5(temp_image_2)
    x5_2 = feature_6(temp_image_2)
    x6_2 = feature_7(temp_image_2)
    x7_2 = feature_8(temp_image_2)
    x8_2 = feature_9(temp_image_2)
    x9_2 = feature_10(temp_image_2)
    
    x_feature_0 = np.array([x0_0, x1_0, x2_0, x3_0, x4_0, x5_0, x6_0, x7_0, x8_0, x9_0], dtype='float32') # 하나의 숫자 이미지를 5개의 특징 set으로
    x_feature_0 = np.resize(x_feature_0, (1, 10)) # 배열을 쌓기 위해서 size를 맞춰줌
    x_0_set = np.concatenate((x_0_set, x_feature_0), axis=0)
    
    x_feature_1 = np.array([x0_1, x1_1, x2_1, x3_1, x4_1, x5_1, x6_1, x7_1, x8_1, x9_1], dtype='float32') # 하나의 숫자 이미지를 5개의 특징 set으로
    x_feature_1 = np.resize(x_feature_1, (1, 10)) # 배열을 쌓기 위해서 size를 맞춰줌
    x_1_set = np.concatenate((x_1_set, x_feature_1), axis=0)
    
    x_feature_2 = np.array([x0_2, x1_2, x2_2, x3_2, x4_2, x5_2, x6_2, x7_2, x8_2, x9_2], dtype='float32') # 하나의 숫자 이미지를 5개의 특징 set으로
    x_feature_2 = np.resize(x_feature_2, (1, 10)) # 배열을 쌓기 위해서 size를 맞춰줌
    x_2_set = np.concatenate((x_2_set, x_feature_2), axis=0)

# Setting step
epoch1 = np.arange(0, 500, 1)
epoch2 = np.arange(500, 1000, 1)
epoch3 = np.arange(1000, 1500, 1)

# Drawing data; 가로축 기댓값
plt.figure()
plt.plot(epoch1, x_0_set[:, 0], 'r')
plt.plot(epoch2, x_1_set[:, 0], 'g')
plt.plot(epoch3, x_2_set[:, 0], 'b')
plt.legend(['set1', 'set2', 'set3'], loc='center right')
plt.xlabel('epoch')
plt.ylabel('data')
plt.title('1. set')
plt.grid(True, alpha=0.5)
plt.show()

# Drawing data; 가로축 분산
plt.figure()
plt.plot(epoch1, x_0_set[:, 1], 'r')
plt.plot(epoch2, x_1_set[:, 1], 'g')
plt.plot(epoch3, x_2_set[:, 1], 'b')
plt.legend(['set1', 'set2', 'set3'], loc='center right')
plt.xlabel('epoch')
plt.ylabel('data')
plt.title('2. set')
plt.grid(True, alpha=0.5)
plt.show()

# Drawing data; 세로축 기댓값
plt.figure()
plt.plot(epoch1, x_0_set[:, 2], 'r')
plt.plot(epoch2, x_1_set[:, 2], 'g')
plt.plot(epoch3, x_2_set[:, 2], 'b')
plt.legend(['set1', 'set2', 'set3'], loc='center right')
plt.xlabel('epoch')
plt.ylabel('data')
plt.title('3. set')
plt.grid(True, alpha=0.5)
plt.show()

# Drawing data; 세로축 분산
plt.figure()
plt.plot(epoch1, x_0_set[:, 3], 'r')
plt.plot(epoch2, x_1_set[:, 3], 'g')
plt.plot(epoch3, x_2_set[:, 3], 'b')
plt.legend(['set1', 'set2', 'set3'], loc='center right')
plt.xlabel('epoch')
plt.ylabel('data')
plt.title('4. set')
plt.grid(True, alpha=0.5)
plt.show()

# Drawing data; 대각 기댓값
plt.figure()
plt.plot(epoch1, x_0_set[:, 4], 'r')
plt.plot(epoch2, x_1_set[:, 4], 'g')
plt.plot(epoch3, x_2_set[:, 4], 'b')
plt.legend(['set1', 'set2', 'set3'], loc='center right')
plt.xlabel('epoch')
plt.ylabel('data')
plt.title('5. set')
plt.grid(True, alpha=0.5)
plt.show()

# Drawing data; 대각 분산
plt.figure()
plt.plot(epoch1, x_0_set[:, 5], 'r')
plt.plot(epoch2, x_1_set[:, 5], 'g')
plt.plot(epoch3, x_2_set[:, 5], 'b')
plt.legend(['set1', 'set2', 'set3'], loc='center right')
plt.xlabel('epoch')
plt.ylabel('data')
plt.title('6. set')
plt.grid(True, alpha=0.5)
plt.show()

# Drawing data; 대각 0 개수
plt.figure()
plt.plot(epoch1, x_0_set[:, 6], 'r')
plt.plot(epoch2, x_1_set[:, 6], 'g')
plt.plot(epoch3, x_2_set[:, 6], 'b')
plt.legend(['set1', 'set2', 'set3'], loc='center right')
plt.xlabel('epoch')
plt.ylabel('data')
plt.title('7. set')
plt.grid(True, alpha=0.5)
plt.show()

# Drawing data; 역대각 기댓값
plt.figure()
plt.plot(epoch1, x_0_set[:, 7], 'r')
plt.plot(epoch2, x_1_set[:, 7], 'g')
plt.plot(epoch3, x_2_set[:, 7], 'b')
plt.legend(['set1', 'set2', 'set3'], loc='center right')
plt.xlabel('epoch')
plt.ylabel('data')
plt.title('8. set')
plt.grid(True, alpha=0.5)
plt.show()

# Drawing data; 역대각 분산
plt.figure()
plt.plot(epoch1, x_0_set[:, 8], 'r')
plt.plot(epoch2, x_1_set[:, 8], 'g')
plt.plot(epoch3, x_2_set[:, 8], 'b')
plt.legend(['set1', 'set2', 'set3'], loc='center right')
plt.xlabel('epoch')
plt.ylabel('data')
plt.title('9. set')
plt.grid(True, alpha=0.5)
plt.show()

# Drawing data; 역대각 0 개수
plt.figure()
plt.plot(epoch1, x_0_set[:, 9], 'r')
plt.plot(epoch2, x_1_set[:, 9], 'g')
plt.plot(epoch3, x_2_set[:, 9], 'b')
plt.legend(['set1', 'set2', 'set3'], loc='center right')
plt.xlabel('epoch')
plt.ylabel('data')
plt.title('10. set')
plt.grid(True, alpha=0.5)
plt.show()


