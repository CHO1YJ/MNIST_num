# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 21:36:55 2022

@author: HwangGitae
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from datetime import datetime #학습 시간 측정용 라이브러리
import matplotlib.cm as cm    #confusion matrix 그래프 출력 색상 용 ColorMap 라이브러리

# =============================================================================
# Numpy 라이브러리
# =============================================================================
np.arange()     # 일정 간격의 배열 생성
np.linspace()   

np.random.randn()   # 0, 1 또는 랜덤값으로 초기화된 행렬 생성
np.random.rand()
np.zeros()
np.ones()

np.linalg.inv() #역행렬
np.transpose()  #트랜스포즈
np.dot()        #행렬곱

np.exp()    # e, log, 평균, 합
np.log()
np.mean()
np.sum()


np.array()  #리스트를 배열로 변환

np.array_equiv(a1, a2)  #배열이 같은지 아닌지 확인(리턴: True, False)
np.count_nonzero()      #배열 요소 중 0이 아닌 값의 개수 확인
np.where()              #(조건, 조건에 맞을 때 값, 조건과 다를 때 값)
x[np.where()] or np.where()[0]           #조건 만족하는 요소의 인덱스 반환
np.sin()                #사인 그래프 생성     
np.argmin()             #최소값 인덱스 찾기
np.argmax()             #최대값 인덱스 찾기
np.min()                #최소값 반환
np.max()                #최대값 반환
np.delete()             #(배열, 인덱스, axis=) 삭제
np.random.shuffle()     #행만 섞어줌


# =============================================================================
# 일정 간격의 배열 생성
# =============================================================================
range()


# =============================================================================
# 변수 타입 변경
# =============================================================================
pd.DataFrame()
.to_numpy()
.tolist()
np.array()  #리스트를 배열로 변환



.copy()         #원본 변경 없이 복사
.append()       #마지막에 데이터 추가
math.turnc()    #버림
.sample()       #Pandas, 무작위 데이터 추출
.drop()         #Pandas, 행/열 삭제(index=, axis=)

pd.read_csv()   #csv 데이터 가져오기
.to_csv()       #csv 데이터 내보내기
.loc[]          #데이터 프레임 접근, label 이용 접근

pd.concat()     #데이터 프레임 합치기([데이터1, 데이터2], axis=)

.insert()
list()
.fill()
.reshape()
.transpose()

int()
min()
max()
.index()        #리스트의 요소 인덱스 찾기
str()