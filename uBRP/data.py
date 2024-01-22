import torch
import os
import re
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
def transform_format(instance_file, H_plus, type='greedy'):
    # Read the instance file
    with open(instance_file, 'r') as file:
        lines = file.readlines()

    # Extracting number_of_stacks and number_of_blocks
    num_stacks, num_blocks = map(int, lines[0].split())
    # Initializing the result list
    result = []

    # Loop through each row and transform the data
    for i in range(1, num_stacks + 1):
        if type == 'greedy':
            block_values = list(map(lambda x: 1/int(x), lines[i].split()[1:]))
        else:
            block_values = list(map(lambda x: ((num_blocks+1) - int(x))/(num_blocks+1), lines[i].split()[1:]))
        row = block_values + [0] * H_plus
        result.append(row)

    return torch.tensor(result)
def process_files_with_regex(directory_path, file_regex, H_plus, type = 'greedy'):
    # Use re to find files matching the specified regex pattern
    files = [file for file in os.listdir(directory_path) if re.search(file_regex, file)]
    transform_datas = []
    # Process each matching file
    for file_name in files:
        file_path = os.path.join(directory_path, file_name)
        transformed_data = transform_format(file_path,H_plus, type)
        transform_datas.append(transformed_data.unsqueeze(0))
    return torch.cat(transform_datas)
def data_from_caserta_for_greedy(file_regex="data3-3.*", H_plus=2): #dataH-W-N.data, H_plus = Hmax-H
    directory_path  = './uBRP\\brp-instances-caserta-etal-2012\\CRPTestcases_Caserta'
    transform_datas = process_files_with_regex(directory_path, file_regex, H_plus)
    return transform_datas
def data_from_caserta(file_regex="data3-3.*", H_plus=2): #dataH-W-N.data, H_plus = Hmax-H
    directory_path  = './uBRP\\brp-instances-caserta-etal-2012\\CRPTestcases_Caserta'
    transform_datas = process_files_with_regex(directory_path, file_regex, H_plus, type='caserta')
    return transform_datas


def generate_data(device,n_samples=10,n_containers = 8,max_stacks=4,max_tiers=4, seed = None, plus_tiers = 2):

	if seed is not None:
		torch.manual_seed(seed)
		np.random.seed(seed)
	#
	if n_containers <= 8:
		dataset=np.zeros((n_samples,max_stacks,max_tiers),dtype=float)
		if max_stacks*max_tiers<n_containers : #放不下就寄
			print("max_stacks*max_tiers<n_containers")
			assert max_stacks*max_tiers>=n_containers

		for i in range(n_samples):

			tiers=np.zeros((max_stacks),dtype=int)
			per=np.arange(0,n_containers,1)
			np.random.shuffle(per)
			pos=np.zeros((n_containers,2),dtype=int)

			for j in range(n_containers):

				while True:
					#[low,high)
					idx=np.random.randint(low=0,high=max_stacks)
					if tiers[idx]+1<max_tiers:

						pos[per[j]][0]=idx
						pos[per[j]][1]=tiers[idx]
						dataset[i][idx][tiers[idx]]=(per[j]+1.)/10.
						tiers[idx] += 1
						break

		dataset=torch.FloatTensor(dataset).to(device)
		#返回的是(n_samples,max_stacks,max_tiers)，存的数据就是(编号+1)/10.如果再大那就再改
		return dataset

	else:
		#数据的生成都是h*max_stacks个，然后max_tiers=h+2
		dataset = torch.zeros((n_samples, max_stacks, max_tiers + plus_tiers - 2), dtype=float).to(device)
		if max_stacks * max_tiers < n_containers:  # 放不下就寄
			print("max_stacks*max_tiers<n_containers")
			assert max_stacks * max_tiers >= n_containers

		for i in range(n_samples):
			per = np.arange(0, n_containers, 1)
			np.random.shuffle(per)
			per=torch.FloatTensor((per+1)/(n_containers+1.0))
			data=torch.reshape(per,(max_stacks,max_tiers-2)).to(device)
			add_empty=torch.zeros((max_stacks,plus_tiers),dtype=float).to(device)
			dataset[i]=torch.cat( (data,add_empty) ,dim=1).to(device)

		dataset=dataset.to(torch.float32)
		return dataset


class Generator(Dataset):
	""" https://github.com/utkuozbulak/pytorch-custom-dataset-examples
		https://github.com/wouterkool/attention-learn-to-route/blob/master/problems/vrp/problem_vrp.py
		https://github.com/nperlmut31/Vehicle-Routing-Problem/blob/master/dataloader.py
	    https://github.com/Rintarooo/VRP_DRL_MHA/pytorch/data.py
	"""
	def __init__(self, device, n_samples = 5120,
				 n_containers = 8,max_stacks=4,max_tiers=4, seed = None):
		self.data_pos = generate_data(device, n_samples,n_containers, max_stacks,max_tiers)
		self.n_samples=n_samples

	def __getitem__(self, idx):
		return self.data_pos[idx]

	def __len__(self):
		#这里怎么写？
		return self.n_samples
	
if __name__ == '__main__':
    #print(data_from_caserta())
    print(generate_data(device = 'cpu', n_samples = 1, n_containers = 20, max_stacks = 4, max_tiers = 7, plus_tiers = 5))