import matplotlib.pyplot as plt
import pickle
with open('./plotdata_lowbpp.pkl', 'rb') as f:
    data = pickle.load(f)


paraGroup={}
paraGroup['10']='/data/wym123/paradata/RDtrainPara/epoch_5.pth'
paraGroup['1']='/data/wym123/paradata/RDtrainPara_1/epoch_5.pth'
paraGroup['1e-1']='/data/wym123/paradata/RDtrainPara_2/epoch_2.pth'
paraGroup['1e-2']='/data/wym123/paradata/RDtrainPara_3/epoch_2.pth'
paraGroup['1e-6']='/data/wym123/paradata/RDtrainPara_4/epoch_4.pth'
paraGroup['1e-4']='/data/wym123/paradata/RDtrainPara_5/epoch_5.pth'
paraGroup['1e-3'] = '/data/wym123/paradata/RDtrainPara_6/epoch_3.pth'
ls=list(paraGroup.keys())
ls.reverse()
print(ls)
plt.plot(data['JPGX'], data['JPGY'], 'x:', label='JPG')
for i in ls:
    plt.plot(data[i+'X'], data[i+'Y'], 'x:', label='process_JPG_'+i)

handles, labels = plt.gca().get_legend_handles_labels()
order = [0,3,2,1,4,5,6,7]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
plt.xlabel(u'Bit-rate[bpp]', fontsize=15)
plt.ylabel(u'PSNR[dB]', fontsize=15)
plt.show()













