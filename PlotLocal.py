import numpy as np
import matplotlib.pyplot as plt
a=np.load('./Plotdata2_.npy')
a=a.tolist()
print(a)
# plt.plot(a[0], a[1],'bx:', label=u'JPEG')
# plt.plot(a[2], a[3], 'rx:',label=u'preprocess_JPEG')
#
# plt.xlabel(u'Bit-rate[bpp]', fontsize=15)
# plt.ylabel(u'PSNR[dB]', fontsize=15)
#
# plt.legend()  # 让图例生效
# plt.show()

# print(a[1][0]-a[3][3])