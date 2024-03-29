import matplotlib.pyplot as plt
import pickle
with open('./Cek2/pltdata8.pkl', 'rb') as f:
    data = pickle.load(f)

paraGroup = {}
paraGroup['SGD_1e-3_1e-3'] = '/data/wym123/paradata/diffjpeg_cek2/SGD/epoch_3.pth'
paraGroup['Lr1e-5_1e-4'] = '/data/wym123/paradata/diffjpeg_cek2/Lr1e_4/epoch_2.pth'
paraGroup['Lr1e-4_1e-3'] = '/data/wym123/paradata/diffjpeg_cek2/Lr1e_3/epoch_3.pth'
paraGroup['Lr1e-4_1e-2'] = '/data/wym123/paradata/diffjpeg_cek2/epoch_6.pth'

ls=list(paraGroup.keys())
ls.reverse()
print(ls)
plt.plot(data['JPGX'], data['JPGY'], 'x:', label='JPG')
for i in ls:
    plt.plot(data[i+'X'], data[i+'Y'], 'x:', label=(i if i!='1' else 'SSIM'))

handles, labels = plt.gca().get_legend_handles_labels()
order = [0,4,3,2,1]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
plt.xlabel(u'Bit-rate[bpp]', fontsize=15)
plt.ylabel(u'PSNR[dB]', fontsize=15)
plt.show()













