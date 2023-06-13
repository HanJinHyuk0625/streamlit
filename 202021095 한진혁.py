import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

s1 = signal.lti([100], [5, 6])

frequencies = np.logspace(-2, 2, 500)

w, mag, phase = s1.bode(frequencies)

plt.figure()
plt.subplot(2, 1, 1)
plt.semilogx(w, mag)
plt.title('Bode plot of G(s) = 100/(s(s+10))')
plt.ylabel('Magnitude [dB]')

plt.subplot(2, 1, 2)
plt.semilogx(w, phase)
plt.ylabel('Phase [degrees]')
plt.xlabel('Frequency [Hz]')
plt.show()