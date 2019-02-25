import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import random
import torch.multiprocessing as mp
import seaborn as sns
from matplotlib import pyplot as plt

random.seed(50)

file = "/home/ash/LSTM/dataset/tensor_data.pth"
data_dict = torch.load(file)
train_x = data_dict['train_x'].cuda()
train_y = data_dict['train_y'].cuda()
valid_x = data_dict['test_x'].cuda()
valid_y = data_dict['test_y'].cuda()

for i in range(train_x.shape[2]):
	train_x[:, :, i] = (train_x[:, :, i] - torch.mean(train_x[:, :, i]).item()) / torch.std(train_x[:, :, i]).item()
	valid_x[:, :, i] = (valid_x[:, :, i] - torch.mean(valid_x[:, :, i]).item()) / torch.std(valid_x[:, :, i]).item()

for i in range(train_y.shape[2]):
	train_y[:, :, i] = (train_y[:, :, i] - torch.mean(train_y[:, :, i]).item()) / torch.std(train_y[:, :, i]).item()
	valid_y[:, :, i] = (valid_y[:, :, i] - torch.mean(valid_y[:, :, i]).item()) / torch.std(valid_y[:, :, i]).item()


class Data(Dataset):
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def __len__(self):
		assert self.x.shape[0] == self.y.shape[0]
		return self.x.shape[0]

	def __getitem__(self, idx):
		return self.x[idx], self.y[idx]


bootstrap_sample_size = 8000
bootstrap_idx = [random.randint(0, train_x.shape[0] - bootstrap_sample_size) for _ in range(4)]
train_ensemble = [(train_x[elem:elem + bootstrap_sample_size], train_y[elem:elem + bootstrap_sample_size]) for elem in
				  bootstrap_idx]
train_dataset = [Data(item[0], item[1]) for item in train_ensemble]

n_batch = 1
trainloader = [DataLoader(elem, n_batch, drop_last=True,shuffle=True) for elem in train_dataset]
complete_train=DataLoader(Data(train_x,train_y),shuffle=True)
valid=Data(valid_x,valid_y)
validloader=DataLoader(valid,batch_size=1,shuffle=True)

out_seq_length=5

class LSTM(nn.Module):
	def __init__(self, inp_dim, batch_size, hidden_dim1, hidden_dim2,out_dim):
		super(LSTM, self).__init__()
		self.batch_size = batch_size
		self.inp_dim = inp_dim
		self.hidden_dim1 = hidden_dim1
		self.hidden_dim2 = hidden_dim2
		self.out_dim=out_dim
		self.lstm1 = nn.LSTM(self.inp_dim, self.hidden_dim1, batch_first=True)
		self.lstm2 = nn.LSTM(self.hidden_dim1, self.hidden_dim2, batch_first=True)
		self.linear = nn.Linear(self.hidden_dim2, self.out_dim)

	def init_hidden(self, batch_dim, hidden_dim):
		return tuple(torch.nn.init.xavier_normal_(torch.Tensor(1, batch_dim, hidden_dim)).cuda() for some in range(2))

	def forward(self, input, drop_mask):
		lstm1_out, (h_n1, c_n1) = self.lstm1(input, self.init_hidden(self.batch_size, self.hidden_dim1))
		lstm1_out = nn.functional.dropout2d(lstm1_out, drop_mask)
		lstm2_out, (h_n2, c_n2) = self.lstm2(lstm1_out, self.init_hidden(self.batch_size, self.hidden_dim2))
		lstm2_out = nn.functional.dropout2d(lstm2_out, drop_mask)
		linear_out = self.linear(lstm2_out[:,out_seq_length:,:])  # Extract last timestep output(lstm_out=BatchxSeqXDim]
		return linear_out


model1 = LSTM(4, n_batch, 64, 32, 2).cuda()
model2 = LSTM(4, n_batch, 16, 8,  2).cuda()
model3 = LSTM(4, n_batch, 32, 16, 2).cuda()
model4 = LSTM(4, n_batch, 64, 16, 2).cuda()
model_ensemble = [model1, model2, model3, model4]

optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.005)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.005)
optimizer3 = torch.optim.Adam(model3.parameters(), lr=0.005)
optimizer4 = torch.optim.Adam(model4.parameters(), lr=0.005)

optimizer_ensemble = [optimizer1, optimizer2, optimizer3, optimizer4]
loss_fn = torch.nn.MSELoss()

loss_log={0:[],1:[],2:[],3:[]}
num_epoch = 1000

def train(m):
	for i in range(num_epoch):
		accum_loss, _ = 0, 0
		for x, y in trainloader[m]:
			model_ensemble[m].zero_grad()
			optimizer_ensemble[m].zero_grad()
			y_pred = model_ensemble[m].forward(x, 0.1)
			loss = loss_fn(y_pred, y)
			loss.backward()
			optimizer_ensemble[m].step()
			accum_loss += loss
			_ += 1
		loss_log[m].append(accum_loss.item() / _)
		if i % 50 == 0:
			print("Epoch: {} Model:{} Loss : {}".format(i, m, loss_log[m][-1]))


####Training
"""
if __name__=="__main__":

	processes=[]
	mp.set_start_method('spawn')
	for m in range(4):
		p=mp.Process(target=train,args=(m,))
		p.start()
		processes.append(p)

	for p in processes:
		p.join()

	state = {
		'epoch': num_epoch,
		'state_dict': [model1.state_dict(), model2.state_dict(), model3.state_dict(), model4.state_dict()],
		'optimizer': [optimizer1.state_dict(), optimizer2.state_dict(), optimizer3.state_dict(),
					  optimizer4.state_dict()],
		'loss_log': loss_log}

	filepath = "/home/ash/LSTM/model/drone_ensembleLSTM.pth"
	torch.save(state, filepath)
"""

### Evaluation

checkpoint_file="/home/ash/LSTM/model/drone_ensembleLSTM.pth"
checkpoint=torch.load(checkpoint_file)


model1.load_state_dict(checkpoint['state_dict'][0])
optimizer1.load_state_dict(checkpoint['optimizer'][0])
model2.load_state_dict(checkpoint['state_dict'][1])
optimizer2.load_state_dict(checkpoint['optimizer'][1])
model3.load_state_dict(checkpoint['state_dict'][2])
optimizer3.load_state_dict(checkpoint['optimizer'][2])
model4.load_state_dict(checkpoint['state_dict'][3])
optimizer4.load_state_dict(checkpoint['optimizer'][3])

loss_log=[]
with torch.no_grad():
	for x,y in validloader:
		y_pred=torch.Tensor([])
		for m in range(4):
			for dropout in [0,0.2,0.02]:
				out=model_ensemble[m].forward(x, dropout).squeeze()
				if y_pred.shape[0]==0:
					y_pred=out.unsqueeze(dim=2)
					continue
				else:
					out=torch.unsqueeze(out,dim=2)
					y_pred=torch.cat((y_pred,out),dim=2)

		loss=loss_fn(torch.mean(y_pred,dim=2).squeeze(),y)
		#print(torch.mean(y_pred,dim=2).squeeze(), y,loss.item())
	

		loss_log.append(loss.item())

sns.set()
sns.distplot(loss_log,bins=100,kde=False)
plt.show()