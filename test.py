import pandas as pd
# import seaborn as sns
import numpy as np
# import matplotlib.pyplot as plt

w_hidden = pd.read_csv('result_hidden_10_rate_0.02_epoch_500/w_hidden.csv', header=None)
w_output = pd.read_csv('result_hidden_10_rate_0.02_epoch_500/w_output.csv', header=None)

# decided_y_hat = pd.read_csv('decided_y_hat.csv', header=None).to_numpy(dtype='float')
# result_y_hat = pd.read_csv('result_y_hat.csv', header=None).to_numpy(dtype='float')
# test_OHE_y = pd.read_csv('test_OHE_y.csv', header=None).to_numpy(dtype='float')
# num_output_layer = 3

# project_w_hidden = np.transpose(w_hidden)
# project_w_output = np.transpose(w_output)

# np.savetxt('w_hidden.csv', project_w_hidden, delimiter=",")
# np.savetxt('w_output.csv', project_w_output, delimiter=",")

# # Generate Confusion Matrix
# # Confusion Matrix 정의
# Confusion_Matrix_by_Result = np.zeros((len(result_y_hat), num_output_layer))
# # Confusion Matrix의 참과 거짓에 따른 성분 정의 및 초기화
# # 1행 성분들
# t_element_100 = 0
# f_element_100_010 = 0
# f_element_100_001 = 0
# f_element_100_else = 0
# #2행 성분들
# t_element_010 = 0
# f_element_010_100 = 0
# f_element_010_001 = 0
# f_element_010_else = 0
# # 3행 성분들
# t_element_001 = 0
# f_element_001_100 = 0
# f_element_001_010 = 0
# f_element_001_else = 0
# # 4행 성분 - "000"을 걸러내기 위한 행으로 나머지 3개 성분은 필요 없는 성분!
# f_element_else = 0
# # Algorithm Counting Elements
# for n in range(len(decided_y_hat)):
#     # Filtering diagonal elements
#     if decided_y_hat[n].tolist() == test_OHE_y[n].tolist() and decided_y_hat[n][0] == 1: # 100-100
#         t_element_100 = t_element_100 + 1
#     elif decided_y_hat[n].tolist() == test_OHE_y[n].tolist() and decided_y_hat[n][1] == 1: # 010-010
#         t_element_010 = t_element_010 + 1
#     elif decided_y_hat[n].tolist() == test_OHE_y[n].tolist() and decided_y_hat[n][2] == 1: # 001-001
#         t_element_001 = t_element_001 + 1 
    
#     # Filtering elements of first row
#     elif decided_y_hat[n].tolist() != test_OHE_y[n].tolist() and decided_y_hat[n][0] == 1:
#         if test_OHE_y[n].tolist() == [0, 1, 0]: # 100-010
#             f_element_100_010 = f_element_100_010 + 1
#         elif test_OHE_y[n].tolist() == [0, 0, 1]: # 100-001
#             f_element_100_001 = f_element_100_001 + 1
#         else: # 100-else
#             f_element_100_else = f_element_100_else + 1
    
#     # Filtering elements of second row
#     elif decided_y_hat[n].tolist() != test_OHE_y[n].tolist() and decided_y_hat[n][1] == 1:
#         if test_OHE_y[n].tolist() == [1, 0, 0]: # 010-100
#             f_element_010_100 = f_element_010_100 + 1
#         elif test_OHE_y[n].tolist() == [0, 0, 1]: # 010-001
#             f_element_010_001 = f_element_010_001 + 1
#         else: # 010-else
#             f_element_010_else = f_element_010_else + 1
#     # Filtering elements of third row
#     elif decided_y_hat[n].tolist() != test_OHE_y[n].tolist() and decided_y_hat[n][2] == 1:
#         if test_OHE_y[n].tolist() == [1, 0, 0]: # 001-100
#             f_element_001_100 = f_element_001_100 + 1
#         elif test_OHE_y[n].tolist() == [0, 1, 0]: # 001-010
#             f_element_001_010 = f_element_001_010 + 1
#         else: # 001-else
#             f_element_001_else = f_element_001_else + 1
#     # Filtering elements of fourth row
#     elif decided_y_hat[n].tolist() != test_OHE_y[n].tolist() and decided_y_hat[n].tolist() == [0, 0, 0]:
#         f_element_else = f_element_else + 1

# # 성분들을 확률로 나타내기 위한 행 별로 전체 개수 계산
# total_100 = t_element_100 + f_element_100_010 + f_element_100_001 + f_element_100_else
# total_010 = f_element_010_100 + t_element_010 + f_element_010_001 + f_element_010_else
# total_001 = f_element_001_100 + f_element_001_010 + t_element_001 + f_element_001_else
# list_total = np.array([total_100, total_010, total_001, f_element_else])

# # Confusion_Matrix_Result 초기화
# Confusion_Matrix_by_Result = \
#     np.array([[t_element_100, f_element_100_010, f_element_100_001, f_element_100_else], \
#               [f_element_010_100, t_element_010, f_element_010_001, f_element_010_else], \
#                   [f_element_001_100, f_element_001_010, t_element_001, f_element_001_else], \
#                       [0, 0, 0, f_element_else]], dtype='float')

# # Confusion Matrix_Result 성분들의 확률화
# for n in range(Confusion_Matrix_by_Result.shape[0]):
#     Confusion_Matrix_by_Result[n] = Confusion_Matrix_by_Result[n] / list_total[n]

# # 데이터 프레임으로 변환
# df_cm = pd.DataFrame(Confusion_Matrix_by_Result, index = \
#                       [i for i in ["0", "1", "2", "Else"]], \
#                       columns=[i for i in ["0", "1", "2", "Else"]])
# # Confusion Matrix 시각화
# plt.figure(figsize=(8, 8))
# plt.title("Confusion Matrix - MNIST", fontsize=35)
# sns.heatmap(df_cm, annot=True, linewidths=.5, annot_kws={"size": 20})
# plt.xlabel("Target - One Hot Encoding of y", fontsize=20)
# plt.ylabel("Output Class - y_hat", fontsize=20)
# plt.tight_layout()
# plt.show()
