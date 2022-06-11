# # Setting Module
import math as ma
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# 숫자 0, 1, 2에 대한 데이터를 생성
# 숫자 0에 대한 DB 정의 및 초기화
x_0_set = np.array([], dtype='float32')
x_0_set = np.resize(x_0_set, (0, 5)) # 배열을 쌓기 위해서 size를 맞춰줌

# 숫자 1에 대한 DB 정의 및 초기화
x_1_set = np.array([], dtype='float32')
x_1_set = np.resize(x_1_set, (0, 5)) # 배열을 쌓기 위해서 size를 맞춰줌

# 숫자 2에 대한 DB 정의 및 초기화
x_2_set = np.array([], dtype='float32')
x_2_set = np.resize(x_2_set, (0, 5)) # 배열을 쌓기 위해서 size를 맞춰줌
for i in range(1, 501): # 숫자 이미지 0, 1, 2에 대해서 500개의 Training set을 특징 set으로 변환
    # 숫자 0에 대한 파일 불러오기
    temp_name_0 = '0_' + str(i) + '.csv'
    temp_image_0 = pd.read_csv('Data_Base/' + temp_name_0, header=None)
    temp_image_0 = temp_image_0.to_numpy(dtype='float32')
    
    # 숫자 1에 대한 파일 불러오기
    temp_name_1 = '1_' + str(i) + '.csv'
    temp_image_1 = pd.read_csv('Data_Base/' + temp_name_1, header=None)
    temp_image_1 = temp_image_1.to_numpy(dtype='float32')
    
    # 숫자 2에 대한 파일 불러오기
    temp_name_2 = '2_' + str(i) + '.csv'
    temp_image_2 = pd.read_csv('Data_Base/' + temp_name_2, header=None)
    temp_image_2 = temp_image_2.to_numpy(dtype='float32')
    
    # 숫자 0 특징 추출
    x0_0 = feature_1(temp_image_0)
    x1_0 = feature_2(temp_image_0)
    x2_0 = feature_3(temp_image_0)
    x3_0 = feature_4(temp_image_0)
    x4_0 = feature_5(temp_image_0)
    
    # 숫자 1 특징 추출
    x0_1 = feature_1(temp_image_1)
    x1_1 = feature_2(temp_image_1)
    x2_1 = feature_3(temp_image_1)
    x3_1 = feature_4(temp_image_1)
    x4_1 = feature_5(temp_image_1)
    
    # 숫자 2 특징 추출
    x0_2 = feature_1(temp_image_2)
    x1_2 = feature_2(temp_image_2)
    x2_2 = feature_3(temp_image_2)
    x3_2 = feature_4(temp_image_2)
    x4_2 = feature_5(temp_image_2)
    
    # 숫자 0 특징 DB 최종 초기화
    x_feature_0 = np.array([x0_0, x1_0, x2_0, x3_0, x4_0], dtype='float32') # 하나의 숫자 이미지를 5개의 특징 set으로
    x_feature_0 = np.resize(x_feature_0, (1, 5)) # 배열을 쌓기 위해서 size를 맞춰줌
    x_0_set = np.concatenate((x_0_set, x_feature_0), axis=0)
    
    # 숫자 1 특징 DB 최종 초기화
    x_feature_1 = np.array([x0_1, x1_1, x2_1, x3_1, x4_1], dtype='float32') # 하나의 숫자 이미지를 5개의 특징 set으로
    x_feature_1 = np.resize(x_feature_1, (1, 5)) # 배열을 쌓기 위해서 size를 맞춰줌
    x_1_set = np.concatenate((x_1_set, x_feature_1), axis=0)
    
    # 숫자 2 특징 DB 최종 초기화
    x_feature_2 = np.array([x0_2, x1_2, x2_2, x3_2, x4_2], dtype='float32') # 하나의 숫자 이미지를 5개의 특징 set으로
    x_feature_2 = np.resize(x_feature_2, (1, 5)) # 배열을 쌓기 위해서 size를 맞춰줌
    x_2_set = np.concatenate((x_2_set, x_feature_2), axis=0)

# ANN 입력 데이터 생성
kind_of_num = 3 # 분류 숫자의 수
num_data = 500 # 각 숫자 데이터의 수
num_features = 5 # 특징의 수

# ANN 입력 데이터
ANN_input_data = np.zeros((kind_of_num * num_data, num_features))
for n in range(kind_of_num * num_data):
    if 0 <= n < 500:
        ANN_input_data[n] = x_0_set[n]
    elif 500 <= n < 1000:
        ANN_input_data[n] = x_1_set[n - 500]
    elif 1000 <= n < 1500:
        ANN_input_data[n] = x_2_set[n - 1000]

# Bias 추가한 ANN 입력 데이터
ANN_input_data_added_bias = \
    np.concatenate((ANN_input_data , np.ones((kind_of_num * num_data, 1))), axis=1)

# y의 class 초기화
y_ANN_input_data = np.ones((kind_of_num * num_data, 1))
for n in range(kind_of_num * num_data):
    if 0 <= n < 500:
        y_ANN_input_data[n] = 0
    elif 500 <= n < 1000:
        y_ANN_input_data[n] = 1
    elif 1000 <= n < 1500:
        y_ANN_input_data[n] = 2

ANN_input_data_added_y = \
    np.concatenate((ANN_input_data , y_ANN_input_data), axis=1)

# One-Hot Encoding 구현
def One_Hot_Encoding(data):
    Encoding_y = [] # One-Hot Encoding 결과값 기록함
    data_y = data[:, 5] # 원본 데이터의 출력 초기화
    y_class = [] # class 종류 및 개수 데이터
    
    # Check Class
    y_class.append(data_y[0])
    for q in range(len(data) - 1):
        if data_y[q] != data_y[q + 1]:
            y_class.append(data_y[q + 1])
    y_class.append(len(y_class)) # 마지막에 class 개수를 첨부
    # 출력에 따라 정렬이 되어있으므로 따로 정렬할 필요는 없음
    # 임의의 데이터라면 정렬이 필요할 것으로 생각됨
    # 성분의 앞에서부터 오름차순으로 class의 성분들이 나타남
    # 마지막 성분은 class의 개수로 초기화
    print(y_class)
    
    # One-Hot Encoding
    for q in range(len(data)):
        if y_class[0] == data_y[q]:
            Encoding_y.append([1, 0, 0])
        elif y_class[1] == data_y[q]:
            Encoding_y.append([0, 1, 0])
        elif y_class[2] == data_y[q]:
            Encoding_y.append([0, 0, 1])
            
    Encoding_y = np.array(Encoding_y) # numpy array로 초기화
    return Encoding_y, y_class
    
# One-Hot Encoding 함수 호출
y_One_Hot_Encoding, y_class = One_Hot_Encoding(ANN_input_data_added_y)

# (2) Two-Layer Neural Network 구현
# Setting Variable
num_hidden_layer = 20 # Hidden Layer의 속성 수
num_output_layer = y_class[-1] # Output layer의 Node 수

def Two_Layer_Neural_Network(data, num_l, num_q):
    # Hidden Layer
    # bias가 붙기 전 입력의 열의 개수가 속성 수
    M = ANN_input_data.shape[1]
    print("Input 속성 수 : ", M)
    
    # 가중치 v는 입력과 Hidden Layer의 node 수에 따라 size가 결정
    list_v = np.zeros((data.shape[1], num_l))
    for n in range(data.shape[1]):
        for l in range(num_l):
            # 가우시안 함수에 따라 랜덤하게 가중치 값 초기화
            list_v[n][l] = np.random.randn()
    alpha = data.dot(list_v) # Hidden Layer의 입력 초기화
    b = 1 / (1 + np.exp(-alpha)) # Hidden Layer의 출력 초기화
    b = np.concatenate((b , np.ones((len(data), 1))), axis=1) # bias 첨부
          
    # Output Layer
    # Output Layer의 속성 수는 출력 class의 수
    Q = num_q
    print("Output 속성 수 : ", Q)
    
    # 가중치 w는 Hidden Layer와 출력의 node 수에 따라 size가 결정
    list_w = np.zeros((num_l + 1, Q))
    for l in range(num_l + 1):
        for q in range(Q):
            list_w[l][q] = np.random.randn()
    
    beta = b.dot(list_w) # Output Layer의 입력 초기화
    y_hat = 1 / (1 + np.exp(-beta)) # Output Layer의 출력 초기화
    
    # 가중치 v와 w 그리고 Hidden, Output Layer의 출력을 반환
    return list_v, list_w, y_hat

# (3) Accuracy 함수 구현
# y_hat을 확률 값에 따라 1과 0으로 구분하는 함수 정의; 입력값은 확률값 p
def Decide_y_hat(p, data):
    for n in range(data.shape[0]):
        for m in range(data.shape[1]):
            if p[n][m] >= 0.5:
                decided_y_hat[n][m] = 1
            else:
                decided_y_hat[n][m] = 0
    return True


# 정확도 값 정의
accuracy = 0.
# 정확도 측정 함수 정의; 입력은 결정된 y_hat 값과 훈련 DB의 y(0과 1로 구성)값
def Measure_Accuracy(dcd_y_hat, data):
    # 정확도 기록함 행렬 정의; 각각의 성분에 대하여 True와 False로 표현
    matrix_accuracy = np.zeros((len(data), 1))
    for m in range(len(data)):
        # 결정된 y_hat과 훈련 DB의 y값이 같으면 True 아니면 False로 초기화
        if dcd_y_hat[m].tolist() == data[m].tolist():
            matrix_accuracy[m] = True
        else:
            matrix_accuracy[m] = False
    # 정확도의 정도에 대한 카운트 정의
    count_accuracy = 0
    for m in range(len(data)):
        # True는 곧 1이므로 1이 많으면 정확도가 높은 것!
        if matrix_accuracy[m] == 1:
            count_accuracy = count_accuracy + 1
    # 정확도 초기화
    acc = (count_accuracy / len(data)) * 100
    # 학습횟수마다 count를 해야하므로 0으로 초기화 후 다시 count
    count_accuracy = 0
    
    return acc

# (4) 데이터 분할 함수
# 데이터를 학습 데이터, 검증 데이터, 평가 데이터로 분할하는 함수
def Dist_Set(data): # 원본 데이터를 입력값으로 함
    # 입력 전 방법 안내
    print("Data set에서 train, validation, test set으로의 분할 비율을 입력하세요.")
    print("(주의) 비율의 총합은 10입니다.")
    
    # 예외처리
    while(True):
        # 입력 직전 방법 안내 및 비율값 입력
        print("(Example) 입력 비율 : 2  (입력값은 0부터 9까지의 자연수))")
        ratio_train = int(input("Train set의 입력 비율 : "))
        ratio_val = int(input("Validaion set의 입력 비율 : "))
        ratio_test = int(input("Test set의 입력 비율 : "))
        ratio_sum = ratio_train + ratio_val + ratio_test
        if ratio_sum == 10: # 비율값들의 합이 10이어야만 탈출
            break
        else: # 주의사항 상기 후 재입력 유도
            print("")
            print("비율의 합이 10이 아닙니다.")
            print("(주의) 비율의 총합은 10이어야 합니다.")
            print("다시 입력해주세요")
    
    print(ratio_train, ratio_val, ratio_test)
    # if ratio_train == 7 and ratio_val == 0 and ratio_test == 3:
    #     flag = True
    
    # 학습, 검증, 평가 데이터의 개수 초기화
    num_train = int(round(len(data) * ratio_train / 10, 0))
    num_val = int(round(len(data) * ratio_val / 10, 0))
    num_test = len(data) - num_train - num_val
    
    # 앞서와 마찬가지로 간편한 데이터 정제를 위해 pandas 활용
    # 데이터 랜덤 비복원 추출 방식
    df_xy = pd.DataFrame(data)
    # 학습 데이터 초기화
    train_set = df_xy.sample(n=num_train, replace=False)
    df_xy = df_xy.drop(train_set.index)
    # 검증 데이터 초기화
    val_set = df_xy.sample(n=num_val, replace=False)
    df_xy = df_xy.drop(val_set.index)
    # 평가 데이터 초기
    test_set = df_xy.sample(n=num_test, replace=False)
    
    # 정제된 데이터를 numpy 데이터로 변환
    train_set = train_set.to_numpy()
    val_set = val_set.to_numpy()
    test_set = test_set.to_numpy()
    
    return train_set, val_set, test_set # 분할된 DB 반환

# 함수 호출 및 반환값에 대한 학습, 검증, 평가 DB 초기화
training_set, validation_set, test_set = Dist_Set(ANN_input_data_added_bias)

# (5)
def shuffle_data(data):
    np.random.shuffle(data)
    OHE_y = np.zeros((len(data), 3))
    for m in range(data.shape[0]): # 630
        for n in range(len(ANN_input_data_added_bias)): # 900
            if data[m].tolist() == ANN_input_data_added_bias[n].tolist():
                OHE_y[m] = y_One_Hot_Encoding[n]
    return OHE_y

# (6) 신경망 학습 함수 및 가중치 갱신
# 신경망 학습 함수
learning_rate = 0.02 # 학습률
# ANN Model 구현
def learning_ANN(data, v, w, y_real, num_l, num_q):
    u = learning_rate # 학습률
    list_mem2 = [] # 가중치 v의 dms 기록함 - 코드가 정상적인지 확인하기 위함
    list_mem1 = [] # 가중치 w의 dms 기록함 - 코드가 정상적인지 확인하기 위함
    MSE = 0 # 모델의 MSE 값 정의
    
    # 가중치 v, w 초기값 및 갱신값 적용
    renewal_v = v
    renewal_w = w
    
    # Error Back Propagation Algorithm of ANN 구현
    for n in range(data.shape[0]):
        # 가중치 v, w 초기화
        dmse_vml = np.zeros((train_data.shape[1], num_hidden_layer))
        dmse_wlq = np.zeros((num_hidden_layer + 1, num_output_layer))
        
        # 갱신된 v, w를 통한 y_hat 계산
        renewal_alpha = data.dot(renewal_v) # Hidden Layer의 입력 초기화
        renewal_b = 1 / (1 + np.exp(-renewal_alpha)) # Hidden Layer의 출력 초기화
        renewal_b = np.concatenate((renewal_b , np.ones((len(data), 1))), axis=1) # bias 첨부
        
        renewal_beta = renewal_b.dot(renewal_w) # Output Layer의 입력 초기화
        renewal_y_hat = 1 / (1 + np.exp(-renewal_beta)) # Output Layer의 출력 초기화
        
        # 가중치 v 갱신
        for m in range(data.shape[1]):
            for l in range(num_l):
                dmse_vml[m][l] = 0
                for q in range(num_q):
                    dmse_vml[m][l] = dmse_vml[m][l] + 2 * (renewal_y_hat[n][q] - y_real[n][q]) \
                        * renewal_y_hat[n][q] * (1 - renewal_y_hat[n][q]) * renewal_w[l][q]
                dmse_vml[m][l] = dmse_vml[m][l] * renewal_b[n][l] * (1 - renewal_b[n][l]) * data[n][m]
        list_mem1.append(dmse_vml)
        renewal_v = renewal_v - u * dmse_vml
    
        # 가중치 w 갱신
        for l in range(num_l + 1):
            for q in range(num_q):
                dmse_wlq[l][q] = 2 * (renewal_y_hat[n][q] - y_real[n][q]) \
                    * renewal_y_hat[n][q] * (1 - renewal_y_hat[n][q]) * renewal_b[n][l]
        list_mem2.append(dmse_wlq)
        renewal_w = renewal_w - u * dmse_wlq
    
    # 모델의 MSE 계산
    for q in range(num_q):
        MSE = MSE + np.sum((renewal_y_hat[:, q] - y_real[:, q]) ** 2) / len(renewal_y_hat)
    mem_MSE.append(MSE)
    
    # y_hat 결정 함수 호출
    Decide_y_hat(renewal_y_hat, renewal_y_hat)
        
    # 정확도 측정 함수 호출 $ 모델의 정확도 측정
    training_accuracy = Measure_Accuracy(decided_y_hat, sorted_OHE_y)
    mem_accuracy.append(training_accuracy)
    print("모델 정확도 :", training_accuracy)
    
    # 가중치 v와 w 그리고 b, y_hat, y_real의 출력을 반환
    return renewal_v, renewal_w, list_mem1, list_mem2

# 가중치 갱신
train_data = training_set
# Shuffle data
epoch = 500 # 학습 횟수
# 초기값 설정
initial_v, initial_w, y_hat = \
                Two_Layer_Neural_Network(train_data, num_hidden_layer, num_output_layer)
# 초기값으로 초기화
weight_v = initial_v
weight_w = initial_w

# y_hat 값 기록함 정의
decided_y_hat = np.zeros((y_hat.shape[0], y_hat.shape[1]))

# 모델 정확도 및 MSE 기록함
mem_accuracy = []
mem_MSE = []

for n in range(epoch):
    print("훈련 횟수 :", n)
    
    sorted_OHE_y = shuffle_data(train_data) # Shuffle 함수 호출 및 Train data 입력  
    
    # 학습 함수 호출 및 가중치와 그 외 변수들 초기화
    weight_v, weight_w, mem1, mem2 = \
        learning_ANN(train_data, weight_v, weight_w, sorted_OHE_y, num_hidden_layer, num_output_layer)

# 모델 평가 결과 확인
test_set = test_set
# 평가 결과에 따른 y_hat 발생 함수
def test_model(data, test_v, test_w):
    # 갱신된 v, w를 통한 y_hat 계산
    test_alpha = data.dot(test_v) # Hidden Layer의 입력 초기화
    test_b = 1 / (1 + np.exp(-test_alpha)) # Hidden Layer의 출력 초기화
    test_b = np.concatenate((test_b , np.ones((len(data), 1))), axis=1) # bias 첨부
    
    test_beta = test_b.dot(test_w) # Output Layer의 입력 초기화
    test_y_hat = 1 / (1 + np.exp(-test_beta)) # Output Layer의 출력 초기화
    return test_y_hat

# 평가 결과 y_hat 반환
result_y_hat = test_model(test_set, weight_v, weight_w)

# 평가 DB에 대응되는 index의 One-Hot Encoding y 발생 함수
def sorted_OHE_y(data):
    OHE_y = np.zeros((len(data), 3))
    for m in range(data.shape[0]):
        for n in range(len(ANN_input_data_added_bias)):
            if data[m].tolist() == ANN_input_data_added_bias[n].tolist():
                OHE_y[m] = y_One_Hot_Encoding[n]
    return OHE_y

# 결과 y_hat 값 기록함 정의
decided_y_hat = np.zeros((result_y_hat.shape[0], result_y_hat.shape[1]))
# y_hat 결정 함수 호출
Decide_y_hat(result_y_hat, result_y_hat)
# test set에 대응하는 One-Hot-Encoding 구현
test_OHE_y = sorted_OHE_y(test_set)
# 정확도 측정 함수 호출
test_accuracy = Measure_Accuracy(decided_y_hat, test_OHE_y)
print("평가 정확도 :", test_accuracy)

# 평가 MSE
result_MSE = 0
for q in range(num_output_layer):
    result_MSE = result_MSE + np.sum((result_y_hat[:, q] - test_OHE_y[:, q]) ** 2) / len(result_y_hat)

# Setting step
epoch1 = np.arange(0, epoch, 1)

# Drawing Accuracy
plt.figure()
plt.plot(epoch1, mem_accuracy, 'ko-', markevery = 50)
plt.scatter(epoch, test_accuracy, c='r')
plt.legend(['Accuracy', 'Test Result'], loc='center right')
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.title('ANN (Aritificial Neural Network)')
plt.grid(True, alpha=0.5)
plt.show()

# Drawing MSE
plt.figure()
plt.plot(epoch1, mem_MSE, 'ko-', markevery = 50)
plt.scatter(epoch, result_MSE, c='r')
plt.legend(['MSE', 'Test MSE'], loc='center right')
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.title('ANN - MSE')
plt.grid(True, alpha=0.5)
plt.show()

# 나의 ANN 알고리즘에서의 가중치 파일은
# 과제에서 요구하는 바와 transpose 관계로
# 변환 후 새로 저장한다
project_w_hidden = np.transpose(weight_v)
project_w_output = np.transpose(weight_w)

# 가중치 파일을 현재 경로에 저장
np.savetxt('w_hidden.csv', project_w_hidden, delimiter=",")
np.savetxt('w_output.csv', project_w_output, delimiter=",")