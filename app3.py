import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

# 전달함수의 분자와 분모 계수
num = [100]
den = [1, 5, 6]  # (s+2)(s+3) = s^2 + 5s + 6

# 폐루프 전달함수 계산
def feedback(num, den):
    num_feedback = np.polymul(num, den)
    den_feedback = np.polyadd(np.polymul(den, den), np.polymul(num, num))
    return num_feedback, den_feedback

num_feedback, den_feedback = feedback(num, den)

# 폐루프 전달함수 출력
print("폐루프 전달함수:")
print(f"num: {num_feedback}")
print(f"den: {den_feedback}")

# Unit step 입력에 대한 응답곡선 그리기
t, y = signal.step((num_feedback, den_feedback))
plt.plot(t, y)
plt.xlabel('Time')
plt.ylabel('Output')
plt.title('Step Response')
plt.grid(True)
plt.show()

# 주파수 응답 보드선도 그리기
omega, mag, phase = signal.bode((num_feedback, den_feedback))
plt.figure()
plt.semilogx(omega, mag)
plt.xlabel('Frequency [rad/s]')
plt.ylabel('Magnitude [dB]')
plt.title('Bode Plot - Magnitude')
plt.grid(True)

plt.figure()
plt.semilogx(omega, phase)
plt.xlabel('Frequency [rad/s]')
plt.ylabel('Phase [degrees]')
plt.title('Bode Plot - Phase')
plt.grid(True)

plt.show()


