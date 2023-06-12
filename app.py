
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Iris 데이터 로드
iris = load_iris()
X = iris.data[:, :2]  # 꽃받침 길이와 폭만 사용
y = iris.target

# 데이터 분할: 훈련 세트와 테스트 세트
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# k-NN 분류기 생성
knn = KNeighborsClassifier(n_neighbors=3)

# 훈련 데이터로 모델 학습
knn.fit(X_train, y_train)

# 테스트 데이터로 예측 수행
y_pred = knn.predict(X_test)

# 정확도 계산
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# 분류 결과 시각화
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Iris Classification')
plt.show()