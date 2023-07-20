import matplotlib.pyplot as plt
import pickle
with open('./plotdata_lowbpp.pkl', 'rb') as f:
    data = pickle.load(f)

paraGroup = {}
paraGroup['MAE'] = '/data/wym123/paradata/RDCek1_MAE/epoch_3.pth'
paraGroup['1'] = '/data/wym123/paradata/RDCek1_SSIM/epoch_4.pth'
paraGroup['5e-4'] = '/data/wym123/paradata/RDtrainPara_7/epoch_5.pth'

ls=list(paraGroup.keys())
ls.reverse()
print(ls)
plt.plot(data['JPGX'], data['JPGY'], 'x:', label='JPG')
for i in ls:
    plt.plot(data[i+'X'], data[i+'Y'], 'x:', label=(i if i!='1' else 'SSIM'))

handles, labels = plt.gca().get_legend_handles_labels()
order = [0,1,2,3]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
plt.xlabel(u'Bit-rate[bpp]', fontsize=15)
plt.ylabel(u'PSNR[dB]', fontsize=15)
plt.show()













