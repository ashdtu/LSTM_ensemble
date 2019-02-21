import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader,Dataset
from sklearn.preprocessing import MinMaxScaler
import random
from matplotlib import pyplot as plt

random.seed(100)

file="phpVeNa5j.csv"
data=pd.read_csv(file)
t_n=int(0.7*len(data))
v_n=int(0.9*len(data))
train_data=data.iloc[:t_n]
valid_data=data.iloc[t_n:]

scaler=MinMaxScaler()
norm_train_x=scaler.fit_transform(train_data.loc[:,train_data.columns!='Class'])
norm_train_y=train_data['Class'].values-1      # Making Class labels in 0-3

norm_valid_x=scaler.transform(valid_data.loc[:,valid_data.columns!='Class'])
norm_valid_y=valid_data['Class'].values-1


def prepare_dataset(inputs,label,seq_length):
    dataX,dataY=[],[]
    for i in range(len(inputs)-seq_length):
        dataX.append(inputs[i:i+seq_length])
        dataY.append(label[i+seq_length])
    return np.array(dataX),np.array(dataY)

train_x,train_y=prepare_dataset(norm_train_x,norm_train_y,10)
valid_x,valid_y=prepare_dataset(norm_valid_x,norm_valid_y,10)

train_x=torch.Tensor(train_x).cuda()
train_y=torch.Tensor(train_y).cuda()
train_y=train_y.long()

valid_x=torch.Tensor(valid_x).cuda()
#valid_x=torch.nn.functional.dropout(valid_x,0.1)*(1-0.1)
valid_y=torch.Tensor(valid_y).cuda()
valid_y=valid_y.long()

class Data(Dataset):
    def __init__(self,x,y):
        self.x=x
        self.y=y

    def __len__(self):
        assert self.x.shape[0]==self.y.shape[0]
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx],self.y[idx]

bootstrap_sample_size=1800
bootstrap_idx=[random.randint(0,len(train_x)-bootstrap_sample_size) for _ in range(4)]
train_ensemble=[(train_x[elem:elem+bootstrap_sample_size],train_y[elem:elem+bootstrap_sample_size]) for elem in bootstrap_idx]
train_dataset=[Data(item[0],item[1]) for item in train_ensemble]

n_batch=1
trainloader=[DataLoader(elem,n_batch,drop_last=True) for elem in train_dataset]
complete_train=DataLoader(Data(train_x,train_y))
valid=Data(valid_x,valid_y)
validloader=DataLoader(valid,batch_size=1)


class LSTM(nn.Module):
    def __init__(self,inp_dim,batch_size,hidden_dim1,hidden_dim2):
        super(LSTM,self).__init__()
        self.batch_size=batch_size
        self.inp_dim=inp_dim
        self.hidden_dim1=hidden_dim1
        self.hidden_dim2=hidden_dim2
        self.lstm1=nn.LSTM(self.inp_dim,self.hidden_dim1,batch_first=True)
        self.lstm2=nn.LSTM(self.hidden_dim1,self.hidden_dim2,batch_first=True)
        self.linear=nn.Linear(self.hidden_dim2,4)

    def init_hidden(self,batch_dim,hidden_dim):
        return tuple(torch.nn.init.xavier_normal_(torch.Tensor(1,batch_dim,hidden_dim)).cuda() for some in range(2))

    def forward(self,input,drop_mask):
        lstm1_out,(h_n1,c_n1)=self.lstm1(input,self.init_hidden(self.batch_size,self.hidden_dim1))
        lstm1_out=nn.functional.dropout2d(lstm1_out,drop_mask)
        lstm2_out,(h_n2,c_n2)=self.lstm2(lstm1_out,self.init_hidden(self.batch_size,self.hidden_dim2))
        lstm2_out=nn.functional.dropout2d(lstm2_out,drop_mask)
        linear_out=self.linear(lstm2_out[:,-1,:])   #Extract last timestep output(lstm_out=BatchxSeqXDim]
        softmax_score=torch.nn.functional.softmax(linear_out,dim=1)
        return softmax_score


model1=LSTM(24,n_batch,64,32).cuda()
model2=LSTM(24,n_batch,16,8).cuda()
model3=LSTM(24,n_batch,32,16,).cuda()
model4=LSTM(24,n_batch,64,16).cuda()
model_ensemble=[model1,model2,model3,model4]

optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.005)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.005)
optimizer3 = torch.optim.Adam(model3.parameters(), lr=0.005)
optimizer4 = torch.optim.Adam(model4.parameters(), lr=0.005)

optimizer_ensemble=[optimizer1,optimizer2,optimizer3,optimizer4]
loss_fn = torch.nn.CrossEntropyLoss()

checkpoint_file="/home/ash/LSTM/model/complete_lstm_2000epoch.pth"
checkpoint=torch.load(checkpoint_file)


model1.load_state_dict(checkpoint['state_dict'][0])
optimizer1.load_state_dict(checkpoint['optimizer'][0])
model2.load_state_dict(checkpoint['state_dict'][1])
optimizer2.load_state_dict(checkpoint['optimizer'][1])
model3.load_state_dict(checkpoint['state_dict'][2])
optimizer3.load_state_dict(checkpoint['optimizer'][2])
model4.load_state_dict(checkpoint['state_dict'][3])
optimizer4.load_state_dict(checkpoint['optimizer'][3])

####Training
"""
loss_log={0:[],1:[],2:[],3:[]}
num_epoch=1000
try:
    for m in range(len(model_ensemble)):
        for i in range(num_epoch):
            accum_loss, _ = 0, 0
            for x, y in trainloader[m]:
                model_ensemble[m].zero_grad()
                optimizer_ensemble[m].zero_grad()
                y_pred = model_ensemble[m].forward(x,0.1)
                loss = loss_fn(y_pred, y)
                loss.backward()
                optimizer_ensemble[m].step()
                accum_loss += loss
                _ += 1
            loss_log[m].append(accum_loss.item() / _)
            if i % 50 == 0:
                print("Epoch: {} Model:{} Loss : {}".format(i, m, loss_log[m][-1]))

    state = {
        'epoch': i,
        'state_dict': [model1.state_dict(), model2.state_dict(), model3.state_dict(), model4.state_dict()],
        'optimizer': [optimizer1.state_dict(), optimizer2.state_dict(), optimizer3.state_dict(),
                      optimizer4.state_dict()],
        'loss_log': loss_log}
    filepath = "/home/ash/LSTM/model/complete_lstm_2000epoch.pth"
    torch.save(state, filepath)

except:
    state = {
        'epoch': i,
        'state_dict': [model1.state_dict(), model2.state_dict(), model3.state_dict(), model4.state_dict()],
        'optimizer': [optimizer1.state_dict(), optimizer2.state_dict(), optimizer3.state_dict(),
                      optimizer4.state_dict()],
        'loss_log': loss_log}
    filepath = "/home/ash/LSTM/model/ensemble_checkpoint1.pth"
    torch.save(state, filepath)

"""

### Evaluation
class_correct = list(0. for j in range(4))
class_total = list(0. for t in range(4))
c=0
total=0
pred={0:[],1:[],2:[],3:[],4:[]}
with torch.no_grad():
    for x,y in validloader:
        y_pred=[]
        for m in range(4):
            for dropout in np.arange(0,0.2,0.02):
                out=model_ensemble[m].forward(x, dropout)
                y_pred.append(out.cpu().squeeze().tolist())
        label=int(np.argmax(np.mean(y_pred,axis=0)))
        class_total[y.item()]+=1
        if label==y.item():
            class_correct[label]+=1
            c+=1
            pred[total]=y_pred
        total+=1

print("Accuracy: Class wise: {},{},{},{}".format(100*class_correct[0]/class_total[0],100*class_correct[1]/class_total[1],100*class_correct[2]/class_total[2],100*class_correct[3]/class_total[3]))
print(class_correct,class_total)
print("Total Accuracy:",100*c/total)
"""
import seaborn as sns
sns.set()
for i in range(4):
    plt.figure()
    sns.distplot(np.array(pred[2])[:,i],bins=50)
plt.show()
"""