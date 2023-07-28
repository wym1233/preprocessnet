import matplotlib.pyplot as plt
import pickle
with open('./diffJPEG_ep7clp_pltdata.pkl', 'rb') as f:
    data = pickle.load(f)

paraGroup = {}
paraGroup['diffJPEG']='/data/wym123/paradata/diffjpeg_cek1/epoch_7.pth'

ls=list(paraGroup.keys())
ls.reverse()
print(ls)
plt.plot(data['JPGX'], data['JPGY'], 'x:', label='JPG')
for i in ls:
    plt.plot(data[i+'X'], data[i+'Y'], 'x:', label=(i if i!='1' else 'SSIM'))

handles, labels = plt.gca().get_legend_handles_labels()
order = [0,1]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
plt.xlabel(u'Bit-rate[bpp]', fontsize=15)
plt.ylabel(u'PSNR[dB]', fontsize=15)
plt.show()













