import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from args import read_args
import numpy as np
import string
import re
import math
args = read_args()


class HetAgg(nn.Module):
	def __init__(self, args, feature_list, a_neigh_list_train, p_neigh_list_train, v_neigh_list_train,\
		 a_train_id_list, p_train_id_list, v_train_id_list):
		super(HetAgg, self).__init__()
		embed_d = args.embed_d
		in_f_d = args.in_f_d
		self.args = args 
		self.P_n = args.P_n
		self.A_n = args.A_n
		self.V_n = args.V_n
		self.feature_list = feature_list
		self.a_neigh_list_train = a_neigh_list_train
		self.p_neigh_list_train = p_neigh_list_train
		self.v_neigh_list_train = v_neigh_list_train
		self.a_train_id_list = a_train_id_list
		self.p_train_id_list = p_train_id_list
		self.v_train_id_list = v_train_id_list

		self.relation_count = 5
		self.embed_d = embed_d
		self.relations_emb = self._init_relation_emb()
		self.norm = 1
		self.criterion = nn.MarginRankingLoss(margin=1, reduction='none')	


		#self.fc_a_agg = nn.Linear(embed_d * 4, embed_d)
		self.fc_a1 = nn.Linear(embed_d , embed_d)
		self.fc_a2 = nn.Linear(embed_d , embed_d)
		self.fc_a3 = nn.Linear(embed_d , embed_d)
		self.fc_a4 = nn.Linear(embed_d , embed_d)

		self.fc_p1 = nn.Linear(embed_d , embed_d)
		self.fc_p2 = nn.Linear(embed_d , embed_d)
		self.fc_p3 = nn.Linear(embed_d , embed_d)
		self.fc_p4 = nn.Linear(embed_d , embed_d)
		self.fc_p5 = nn.Linear(embed_d , embed_d)

		self.fc_v1 = nn.Linear(embed_d , embed_d)
		self.fc_v2 = nn.Linear(embed_d , embed_d)
		self.fc_v3 = nn.Linear(embed_d , embed_d)
		self.fc_v4 = nn.Linear(embed_d , embed_d)
		self.fc_v5 = nn.Linear(embed_d , embed_d)
		self.fc_v6 = nn.Linear(embed_d , embed_d)

		self.a_content_rnn = nn.LSTM(embed_d, int(embed_d/2), 1, bidirectional = True)
		self.p_content_rnn = nn.LSTM(embed_d, int(embed_d/2), 1, bidirectional = True)
		self.v_content_rnn = nn.LSTM(embed_d, int(embed_d/2), 1, bidirectional = True)

		self.a_content_att = nn.MultiheadAttention(embed_d, 8)
		self.p_content_att = nn.MultiheadAttention(embed_d, 8)
		self.v_content_att = nn.MultiheadAttention(embed_d, 8)

		# self.a_content_att = nn.Parameter(torch.ones(embed_d, 1), requires_grad = True)
		# self.p_content_att = nn.Parameter(torch.ones(embed_d, 1), requires_grad = True)
		# self.v_content_att = nn.Parameter(torch.ones(embed_d, 1), requires_grad = True)

		self.a_neigh_rnn = nn.LSTM(embed_d, int(embed_d/2), 1, bidirectional = True)
		self.p_neigh_rnn = nn.LSTM(embed_d, int(embed_d/2), 1, bidirectional = True)
		self.v_neigh_rnn = nn.LSTM(embed_d, int(embed_d/2), 1, bidirectional = True)

		self.a_neigh_att = nn.Parameter(torch.ones(embed_d * 2, 1), requires_grad = True)
		self.p_neigh_att = nn.Parameter(torch.ones(embed_d * 2, 1), requires_grad = True)
		self.v_neigh_att = nn.Parameter(torch.ones(embed_d * 2, 1), requires_grad = True)

		self.softmax = nn.Softmax(dim = 1)
		self.act = nn.LeakyReLU()
		self.drop = nn.Dropout(p = 0.5)
		self.bn = nn.BatchNorm1d(embed_d)
		# self.fc = nn.Linear(, embed_d)

	def init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Linear) or isinstance(m, nn.Parameter):
				nn.init.xavier_normal_(m.weight.data)
				#nn.init.normal_(m.weight.data)
				m.bias.data.fill_(0.1)


	def a_content_agg(self, id_batch): #heterogeneous content aggregation
		embed_d = self.embed_d
		# print(len(id_batch))
		# embed_d = in_f_d, it is flexible to add feature transformer (e.g., FC) here 
		# print (id_batch)
		a_net_embed_batch = self.feature_list[6][id_batch]

		a_text_embed_batch_1 = self.feature_list[7][id_batch, :embed_d][0]
		a_text_embed_batch_2 = self.feature_list[7][id_batch, embed_d : embed_d * 2][0]
		a_text_embed_batch_3 = self.feature_list[7][id_batch, embed_d * 2 : embed_d * 3][0]

		a_net_embed_batch = self.fc_a1(a_net_embed_batch)
		a_text_embed_batch_1 = self.fc_a2(a_text_embed_batch_1)
		a_text_embed_batch_2 = self.fc_a3(a_text_embed_batch_2)
		a_text_embed_batch_3 = self.fc_a4(a_text_embed_batch_3)

		concate_embed = torch.cat((a_net_embed_batch, a_text_embed_batch_1, a_text_embed_batch_2,\
		 a_text_embed_batch_3), 1).view(len(id_batch[0]), 4, embed_d)

		concate_embed = torch.transpose(concate_embed, 0, 1)
		# all_state, last_state = self.a_content_rnn(concate_embed)
		all_state, last_state = self.a_content_att(concate_embed, concate_embed, concate_embed)
		# print(all_state.size())
		return torch.mean(all_state, 0)
		# return all_state


	def p_content_agg(self, id_batch):
		embed_d = self.embed_d
		p_a_embed_batch = self.feature_list[0][id_batch]
		p_t_embed_batch = self.feature_list[1][id_batch]
		p_v_net_embed_batch = self.feature_list[2][id_batch]
		p_a_net_embed_batch = self.feature_list[3][id_batch]
		p_net_embed_batch = self.feature_list[5][id_batch]

		p_a_embed_batch = self.fc_p1(p_a_embed_batch)
		p_t_embed_batch = self.fc_p2(p_t_embed_batch)
		p_v_net_embed_batch = self.fc_p3(p_v_net_embed_batch)
		p_a_net_embed_batch = self.fc_p4(p_a_net_embed_batch)
		p_net_embed_batch = self.fc_p5(p_net_embed_batch)

		concate_embed = torch.cat((p_a_embed_batch, p_t_embed_batch, p_v_net_embed_batch,\
		 p_a_net_embed_batch, p_net_embed_batch), 1).view(len(id_batch[0]), 5, embed_d)

		concate_embed = torch.transpose(concate_embed, 0, 1)
		# all_state, last_state = self.p_content_rnn(concate_embed)
		all_state, last_state = self.p_content_att(concate_embed, concate_embed, concate_embed)

		return torch.mean(all_state, 0)
		# return all_state


	def v_content_agg(self, id_batch):
		embed_d = self.embed_d
		v_net_embed_batch = self.feature_list[8][id_batch]
		v_text_embed_batch_1 = self.feature_list[9][id_batch, :embed_d][0]
		v_text_embed_batch_2 = self.feature_list[9][id_batch, embed_d: 2 * embed_d][0]
		v_text_embed_batch_3 = self.feature_list[9][id_batch, 2 * embed_d: 3 * embed_d][0]
		v_text_embed_batch_4 = self.feature_list[9][id_batch, 3 * embed_d: 4 * embed_d][0]
		v_text_embed_batch_5 = self.feature_list[9][id_batch, 4 * embed_d:][0]

		v_net_embed_batch = self.fc_v1(v_net_embed_batch)
		v_text_embed_batch_1 = self.fc_v2(v_text_embed_batch_1)
		v_text_embed_batch_2 = self.fc_v3(v_text_embed_batch_2)
		v_text_embed_batch_3 = self.fc_v4(v_text_embed_batch_3)
		v_text_embed_batch_4 = self.fc_v5(v_text_embed_batch_4)
		v_text_embed_batch_5 = self.fc_v6(v_text_embed_batch_5)

		concate_embed = torch.cat((v_net_embed_batch, v_text_embed_batch_1, v_text_embed_batch_2, v_text_embed_batch_3,\
			v_text_embed_batch_4, v_text_embed_batch_5), 1).view(len(id_batch[0]), 6, embed_d)

		concate_embed = torch.transpose(concate_embed, 0, 1)
		# print(concate_embed.size())
		# all_state, last_state = self.v_content_rnn(concate_embed)
		all_state, last_state = self.v_content_att(concate_embed, concate_embed, concate_embed)
		
		return torch.mean(all_state, 0)
		# return all_state


	def node_neigh_agg(self, id_batch, node_type): #type based neighbor aggregation with rnn 
		embed_d = self.embed_d
		print("node_neigh_agg_id_batch: ", id_batch)

		if node_type == 1 or node_type == 2:
			batch_s = int(len(id_batch[0]) / 10)
		else:
			#print (len(id_batch[0]))
			batch_s = int(len(id_batch[0]) / 3)

		if node_type == 1:
			neigh_agg = self.a_content_agg(id_batch).view(batch_s, 10, embed_d)
			neigh_agg = torch.transpose(neigh_agg, 0, 1)
			all_state, last_state  = self.a_neigh_rnn(neigh_agg)
		elif node_type == 2:
			neigh_agg = self.p_content_agg(id_batch).view(batch_s, 10, embed_d)
			neigh_agg = torch.transpose(neigh_agg, 0, 1)
			all_state, last_state  = self.p_neigh_rnn(neigh_agg)
		else:
			neigh_agg = self.v_content_agg(id_batch).view(batch_s, 3, embed_d)
			neigh_agg = torch.transpose(neigh_agg, 0, 1)
			all_state, last_state  = self.v_neigh_rnn(neigh_agg)
		neigh_agg = torch.mean(all_state, 0).view(batch_s, embed_d)
		print("neigh_agg: ", neigh_agg.size())
		return neigh_agg


	def node_het_agg(self, id_batch, node_type): #heterogeneous neighbor aggregation
		print("node_het_agg_id_batch: ", len(id_batch))
		a_neigh_batch = [[0] * 10] * len(id_batch)
		p_neigh_batch = [[0] * 10] * len(id_batch)
		v_neigh_batch = [[0] * 3] * len(id_batch)
		for i in range(len(id_batch)):
			if node_type == 1:
				a_neigh_batch[i] = self.a_neigh_list_train[0][id_batch[i]]
				p_neigh_batch[i] = self.a_neigh_list_train[1][id_batch[i]]
				v_neigh_batch[i] = self.a_neigh_list_train[2][id_batch[i]]
			elif node_type == 2:
				a_neigh_batch[i] = self.p_neigh_list_train[0][id_batch[i]]
				p_neigh_batch[i] = self.p_neigh_list_train[1][id_batch[i]]
				v_neigh_batch[i] = self.p_neigh_list_train[2][id_batch[i]]
			else:
				a_neigh_batch[i] = self.v_neigh_list_train[0][id_batch[i]]
				p_neigh_batch[i] = self.v_neigh_list_train[1][id_batch[i]]
				v_neigh_batch[i] = self.v_neigh_list_train[2][id_batch[i]]

		a_neigh_batch = np.reshape(a_neigh_batch, (1, -1))
		a_agg_batch = self.node_neigh_agg(a_neigh_batch, 1)
		p_neigh_batch = np.reshape(p_neigh_batch, (1, -1))
		p_agg_batch = self.node_neigh_agg(p_neigh_batch, 2)
		v_neigh_batch = np.reshape(v_neigh_batch, (1, -1))
		v_agg_batch = self.node_neigh_agg(v_neigh_batch, 3)

		#attention module
		id_batch = np.reshape(id_batch, (1, -1))
		if node_type == 1:
			c_agg_batch = self.a_content_agg(id_batch)
		elif node_type == 2:
			c_agg_batch = self.p_content_agg(id_batch)
		else:
			c_agg_batch = self.v_content_agg(id_batch)

		c_agg_batch_2 = torch.cat((c_agg_batch, c_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)
		a_agg_batch_2 = torch.cat((c_agg_batch, a_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)
		p_agg_batch_2 = torch.cat((c_agg_batch, p_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)
		v_agg_batch_2 = torch.cat((c_agg_batch, v_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)

		#compute weights
		concate_embed = torch.cat((c_agg_batch_2, a_agg_batch_2, p_agg_batch_2,\
		 v_agg_batch_2), 1).view(len(c_agg_batch), 4, self.embed_d * 2)
		if node_type == 1:
			atten_w = self.act(torch.bmm(concate_embed, self.a_neigh_att.unsqueeze(0).expand(len(c_agg_batch),\
			 *self.a_neigh_att.size())))
		elif node_type == 2:
			atten_w = self.act(torch.bmm(concate_embed, self.p_neigh_att.unsqueeze(0).expand(len(c_agg_batch),\
			 *self.p_neigh_att.size())))
		else:
			atten_w = self.act(torch.bmm(concate_embed, self.v_neigh_att.unsqueeze(0).expand(len(c_agg_batch),\
			 *self.v_neigh_att.size())))
		atten_w = self.softmax(atten_w).view(len(c_agg_batch), 1, 4)

		#weighted combination
		concate_embed = torch.cat((c_agg_batch, a_agg_batch, p_agg_batch,\
		 v_agg_batch), 1).view(len(c_agg_batch), 4, self.embed_d)
		weight_agg_batch = torch.bmm(atten_w, concate_embed).view(len(c_agg_batch), self.embed_d)
		print("weight_agg_batch: ", weight_agg_batch.size())

		return weight_agg_batch


	def het_agg(self, triple_index, c_id_batch, pos_id_batch, neg_id_batch):
		embed_d = self.embed_d
		# batch processing
		# nine cases for academic data (author, paper, venue)
		if triple_index == 0: # change to relation type (5)
			c_agg = self.a_content_agg([c_id_batch])
			p_agg = self.a_content_agg([pos_id_batch])
			n_agg = self.a_content_agg([neg_id_batch])
		elif triple_index == 1:
			c_agg = self.a_content_agg([c_id_batch])
			p_agg = self.p_content_agg([pos_id_batch])
			n_agg = self.p_content_agg([neg_id_batch])
		elif triple_index == 2:
			c_agg = self.a_content_agg([c_id_batch])
			p_agg = self.v_content_agg([pos_id_batch])
			n_agg = self.v_content_agg([neg_id_batch])
		elif triple_index == 3:
			c_agg = self.p_content_agg([c_id_batch])
			p_agg = self.a_content_agg([pos_id_batch])
			n_agg = self.a_content_agg([neg_id_batch])
		elif triple_index == 4:
			c_agg = self.p_content_agg([c_id_batch])
			p_agg = self.p_content_agg([pos_id_batch])
			n_agg = self.p_content_agg([neg_id_batch])	
		elif triple_index == 5:
			c_agg = self.p_content_agg([c_id_batch])
			p_agg = self.v_content_agg([pos_id_batch])
			n_agg = self.v_content_agg([neg_id_batch])	
		elif triple_index == 6:
			c_agg = self.v_content_agg([c_id_batch])
			p_agg = self.a_content_agg([pos_id_batch])
			n_agg = self.a_content_agg([neg_id_batch])		
		elif triple_index == 7:
			c_agg = self.v_content_agg([c_id_batch])
			p_agg = self.p_content_agg([pos_id_batch])
			n_agg = self.p_content_agg([neg_id_batch])	
		elif triple_index == 8:
			c_agg = self.v_content_agg([c_id_batch])
			p_agg = self.v_content_agg([pos_id_batch])
			n_agg = self.v_content_agg([neg_id_batch])
		elif triple_index == 9: #save learned node embedding
			embed_file = open(self.args.data_path + "node_embedding4_transE.txt", "w")
			save_batch_s = self.args.mini_batch_s
			for i in range(3):
				if i == 0:
					batch_number = int(len(self.a_train_id_list) / save_batch_s)
				elif i == 1:
					batch_number = int(len(self.p_train_id_list) / save_batch_s)
				else:
					batch_number = int(len(self.v_train_id_list) / save_batch_s)
				for j in range(batch_number):
					if i == 0:
						id_batch = self.a_train_id_list[j * save_batch_s : (j + 1) * save_batch_s]
						out_temp = self.a_content_agg([id_batch]) 
					elif i == 1:
						id_batch = self.p_train_id_list[j * save_batch_s : (j + 1) * save_batch_s]
						out_temp = self.p_content_agg([id_batch])
					else:
						id_batch = self.v_train_id_list[j * save_batch_s : (j + 1) * save_batch_s]
						out_temp = self.v_content_agg([id_batch])
					out_temp = out_temp.data.cpu().numpy()
					for k in range(len(id_batch)):
						index = id_batch[k]
						if i == 0:
							embed_file.write('a' + str(index) + " ")
						elif i == 1:
							embed_file.write('p' + str(index) + " ")
						else:
							embed_file.write('v' + str(index) + " ")
						for l in range(embed_d - 1):
							embed_file.write(str(out_temp[k][l]) + " ")
						embed_file.write(str(out_temp[k][-1]) + "\n")

				if i == 0:
					id_batch = self.a_train_id_list[batch_number * save_batch_s : -1]
					out_temp = self.a_content_agg([id_batch])
				elif i == 1:
					id_batch = self.p_train_id_list[batch_number * save_batch_s : -1]
					out_temp = self.p_content_agg([id_batch])
				else:
					id_batch = self.v_train_id_list[batch_number * save_batch_s : -1]
					out_temp = self.v_content_agg([id_batch])
				out_temp = out_temp.data.cpu().numpy()
				for k in range(len(id_batch)):
					index = id_batch[k]
					if i == 0:
						embed_file.write('a' + str(index) + " ")
					elif i == 1:
						embed_file.write('p' + str(index) + " ")
					else:
						embed_file.write('v' + str(index) + " ")
					for l in range(embed_d - 1):
						embed_file.write(str(out_temp[k][l]) + " ")
					embed_file.write(str(out_temp[k][-1]) + "\n")
			embed_file.close()
			return [], [], []

		return c_agg, p_agg, n_agg


	def aggregate_all(self, triple_list_batch, triple_index):
		c_id_batch = [x[0] for x in triple_list_batch]
		pos_id_batch = [x[2] for x in triple_list_batch]
		# neg_id_batch = [x[2] for x in triple_list_batch]

		c_agg, pos_agg, neg_agg = self.het_agg(triple_index, c_id_batch, pos_id_batch, neg_id_batch)

		return c_agg, pos_agg, neg_agg


	
	def _init_relation_emb(self):
		relations_emb = nn.Embedding(num_embeddings=self.relation_count + 1,
										embedding_dim=self.embed_d,
										padding_idx=self.relation_count)
		uniform_range = 6 / np.sqrt(self.embed_d)
		relations_emb.weight.data.uniform_(-uniform_range, uniform_range)
		# -1 to avoid nan for OOV vector
		relations_emb.weight.data[:-1, :].div_(relations_emb.weight.data[:-1, :].norm(p=1, dim=1, keepdim=True))
		return relations_emb

	def forward(self, triple_list_batch, triple_index):
		c_out, p_out, n_out = self.aggregate_all(triple_list_batch, triple_index)
		
		# assert positive_triplets.size()[1] == 3
		positive_distances = self._distance(c_out, triple_list_batch[:, 1], p_out)

		# assert negative_triplets.size()[1] == 3
		negative_distances = self._distance(c_out, triple_list_batch[:, 1], p_out)

		return self.loss(positive_distances, negative_distances), positive_distances, negative_distances		
		# return c_out, p_out, n_out
	
	# def forward(self, positive_triplets: torch.LongTensor, negative_triplets: torch.LongTensor):
	# 	"""Return model losses based on the input.

	# 	:param positive_triplets: triplets of positives in Bx3 shape (B - batch, 3 - head, relation and tail)
	# 	:param negative_triplets: triplets of negatives in Bx3 shape (B - batch, 3 - head, relation and tail)
	# 	:return: tuple of the model loss, positive triplets loss component, negative triples loss component
	# 	"""
	# 	# -1 to avoid nan for OOV vector
	# 	self.entities_emb.weight.data[:-1, :].div_(self.entities_emb.weight.data[:-1, :].norm(p=2, dim=1, keepdim=True))

	# 	assert positive_triplets.size()[1] == 3
	# 	positive_distances = self._distance(positive_triplets)

	# 	assert negative_triplets.size()[1] == 3
	# 	negative_distances = self._distance(negative_triplets)

	# 	return self.loss(positive_distances, negative_distances), positive_distances, negative_distances

	def predict(self, triplets: torch.LongTensor):
		"""Calculated dissimilarity score for given triplets.

		:param triplets: triplets in Bx3 shape (B - batch, 3 - head, relation and tail)
		:return: dissimilarity score for given triplets
		"""
		return self._distance(triplets)

	def loss(self, positive_distances, negative_distances):
		target = torch.tensor([-1], dtype=torch.long)
		return self.criterion(positive_distances, negative_distances, target)

	def _distance(self, head, relation_id, tail): #add head, tail after attention as input
		"""Triplets should have shape Bx3 where dim 3 are head id, relation id, tail id."""
		# assert triplets.size()[1] == 3
		# heads = triplets[:, 0]
		# relations = triplets[:, 1]
		# tails = triplets[:, 2]
		return (head + self.relations_emb(relation_id) - tail).norm(p=self.norm, dim=1)


def cross_entropy_loss(c_embed_batch, pos_embed_batch, neg_embed_batch, embed_d):
	batch_size = c_embed_batch.shape[0] * c_embed_batch.shape[1]
	# print(c_embed_batch.shape[0], "|", c_embed_batch.shape[1])
	# print(c_embed)
	c_embed = c_embed_batch.view(batch_size, 1, embed_d) #batch_size = 9 (triple_index)* 200 (batch_size in a triple index)
	pos_embed = pos_embed_batch.view(batch_size, embed_d, 1)
	neg_embed = neg_embed_batch.view(batch_size, embed_d, 1)

	out_p = torch.bmm(c_embed, pos_embed)
	out_n = - torch.bmm(c_embed, neg_embed)

	sum_p = F.logsigmoid(out_p)
	sum_n = F.logsigmoid(out_n)
	loss_sum = - (sum_p + sum_n)

	#loss_sum = loss_sum.sum() / batch_size

	return loss_sum.mean()

