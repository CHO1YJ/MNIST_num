# # Setting Module
import math as ma
import numpy as np

# 기대값 계산 - E[x] = np.sum(x * p_x)
# 분산 계산 - Var[x] = np.sum(np.pow(x - E_x, 2) * p_x)

#################################################################################
def feature_1(input_data):
    # 특징 후보 2번: 가로축 Projection => 확률밀도함수로 변환 => 분산
    output_value = 0 # 출력 값 정의 및 초기화
    sum_input_data = [] # 입력 성분 합계 정의 및 초기화
    E_x = 0 # 기댓값 정의 및 초기화
    Var_x = 0 # 분산값 정의 및 초기화
    # 가로축 성분들에 대하여 확률밀도함수로 변환 과정
    for n in range(input_data.shape[1]):
        sum_input_data.append(np.sum(input_data[:, n]))
    sum_input_data = np.array(sum_input_data) / np.sum(sum_input_data)
    # 초기화된 확률밀도함수의 기댓값 계산 과정
    for x in range(input_data.shape[1]):
        E_x = E_x + sum_input_data[x] * x
    # 초기화된 확률밀도함수의 분산값 계산 과정
    for x in range(input_data.shape[1]):
        Var_x = Var_x + np.power(x - E_x, 2) * sum_input_data[x]
    # 출력 값 변수에 분산값을 적재
    output_value = Var_x
    return output_value # 출력값 반환
#################################################################################

#################################################################################
def feature_2(input_data):
    # 특징 후보 4번: 세로축 Projection => 확률밀도함수로 변환 => 분산
    output_value = 0 # 출력 값 정의 및 초기화
    sum_input_data = [] # 입력 성분 합계 정의 및 초기화
    E_x = 0 # 기댓값 정의 및 초기화
    Var_x = 0 # 분산값 정의 및 초기화
    # 세로축 성분들에 대하여 확률밀도함수로 변환 과정
    for m in range(input_data.shape[0]):
        sum_input_data.append(np.sum(input_data[m, :]))
    sum_input_data = np.array(sum_input_data) / np.sum(sum_input_data)
    # 초기화된 확률밀도함수의 기댓값 계산 과정
    for x in range(input_data.shape[0]):
        E_x = E_x + sum_input_data[x] * x
    # 초기화된 확률밀도함수의 분산값 계산 과정
    for x in range(input_data.shape[0]):
        Var_x = Var_x + np.power(x - E_x, 2) * sum_input_data[x]
    # 출력 값 변수에 분산값을 적재
    output_value = Var_x                                  
    return output_value # 출력값 반환
#################################################################################

#################################################################################
def feature_3(input_data):
    # 특징 후보 5번: Diagonal 원소 배열 추출 => 확률밀도함수로 변환 => 기댓값
    output_value = 0 # 출력 값 정의 및 초기화
    E_x = 0 # 기댓값 정의 및 초기화
    # 행렬의 대각선 성분들에 대하여 확률밀도함수로 변환 과정
    sum_input_data = np.diag(input_data)
    sum_input_data = np.array(sum_input_data) / np.sum(sum_input_data)
    # 초기화된 확률밀도함수의 기댓값 계산 과정
    for x in range(input_data.shape[0]):
        E_x = E_x + sum_input_data[x] * x
    # 출력 값 변수에 분산값을 적재
    output_value = E_x
    return output_value # 출력값 반환
#################################################################################

#################################################################################
def feature_4(input_data):
    # 특징 후보 6번: Diagonal 원소 배열 추출 => 확률밀도함수로 변환 => 분산
    output_value = 0 # 출력 값 정의 및 초기화
    E_x = 0 # 기댓값 정의 및 초기화
    Var_x = 0 # 분산값 정의 및 초기화
    # 행렬의 대각선 성분들에 대하여 확률밀도함수로 변환 과정
    sum_input_data = np.diag(input_data)
    sum_input_data = np.array(sum_input_data) / np.sum(sum_input_data)
    # 초기화된 확률밀도함수의 기댓값 계산 과정
    for x in range(input_data.shape[0]):
        E_x = E_x + sum_input_data[x] * x
    # 초기화된 확률밀도함수의 분산값 계산 과정
    for x in range(input_data.shape[0]):
        Var_x = Var_x + np.power(x - E_x, 2) * sum_input_data[x]
    # 출력 값 변수에 분산값을 적재
    output_value = Var_x
    return output_value # 출력값 반환
#################################################################################

#################################################################################
def feature_5(input_data):
    # 특징 후보 9번: Anti-Diagonal 원소 배열 추출 => 분산
    output_value = 0 # 출력 값 정의 및 초기화
    E_x = 0 # 기댓값 정의 및 초기화
    Var_x = 0 # 분산값 정의 및 초기화
    # 행렬의 역대각선 성분들에 대하여 확률밀도함수로 변환 과정
    sum_input_data = np.fliplr(input_data).diagonal()
    sum_input_data = np.array(sum_input_data) / np.sum(sum_input_data)
    # 초기화된 확률밀도함수의 기댓값 계산 과정
    for x in range(input_data.shape[0]):
        E_x = E_x + sum_input_data[x] * x
    # 초기화된 확률밀도함수의 분산값 계산 과정
    for x in range(input_data.shape[0]):
        Var_x = Var_x + np.power(x - E_x, 2) * sum_input_data[x]
    # 출력 값 변수에 분산값을 적재
    output_value = Var_x
    return output_value # 출력값 반환
#################################################################################