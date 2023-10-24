import torch.nn as nn
import torch
from transformers import BertPreTrainedModel,BertModel

class TC(nn.Module):
	def __init__(self, vocab_size, embed_size, class_num, dropout_rate, p,
				 criteration="CrossEntropyLoss", ):
		super(TC, self).__init__()
		self.p = p
		self.vocab_size = vocab_size
		self.edge_weight = nn.Embedding((vocab_size) * (vocab_size) + 1, 1, padding_idx=0)
		self.node_embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
		self.node_weight = nn.Embedding(vocab_size, 1, padding_idx=0)
		self.normalization = nn.BatchNorm1d(embed_size)
		self.fc = nn.Sequential(
			nn.Linear(embed_size, class_num),
			nn.ReLU(),
			nn.Dropout(dropout_rate),
			nn.LogSoftmax(dim=1)
		)
		self.reset_params()
		if criteration == "CrossEntropyLoss":
			self.criteration = nn.CrossEntropyLoss()
		else:
			# default loss
			self.criteration = nn.CrossEntropyLoss()


	def reset_params(self):
		nn.init.xavier_uniform_(self.edge_weight.weight)
		nn.init.xavier_uniform_(self.node_weight.weight)


	# def forward(self, input_ids, labels, attention_mask=None, token_type_ids=None):
	# 	clf_input = self.embedding(input_ids.long())
	# 	logits = self.classifier(clf_input)
	# 	loss = self.criteration(logits, labels)
	# 	return loss, logits


	def forward(self, input_ids,  attention_mask=None,token_type_ids=None, labels=None, ):
		'''
        :param X: (bz, max_seq_len)  sentence nodes
        :param NX: (bz, max_seq_len, neighbor_num)  neighbor nodes of each node in X
        :param EW: (bz, max_seq_len, neighbor_num)  neighbor weights of each node in X
        :return:
        '''
		# neighbor (bz, seq_len, neighbor_num, embed_dim)
		X = input_ids.long()
		NX, EW = self.get_neighbors(X, nb_neighbor=self.p)
		NX = NX.long()
		EW = EW.long()
		# NX = input_ids
		# EW = input_ids
		Ra = self.node_embedding(NX)
		# edge weight  (bz, seq_len, neighbor_num, 1)
		Ean = self.edge_weight(EW)
		# neighbor representation  (bz, seq_len, embed_dim)
		Mn = (Ra * Ean).max(dim=2)[0]  # max pooling
		# self representation (bz, seq_len, embed_dim)
		Rn = self.node_embedding(X)
		# self node weight  (bz, seq_len, 1)
		Nn = self.node_weight(X)
		# aggregate node features
		y = (1 - Nn) * Mn + Nn * Rn
		logits = self.fc(y.sum(dim=1))
		if labels is not None:
			loss = self.criteration(logits, labels)
		else:
			loss = None
		return loss, logits, labels

	def get_neighbors(self, x_ids, nb_neighbor=2):
		B, L = x_ids.size()
		neighbours = torch.zeros(size=(L, B, 2*nb_neighbor))
		ew_ids = torch.zeros(size=(L, B, 2*nb_neighbor))
		# pad = [0] * nb_neighbor
		pad = torch.zeros(size=(B, nb_neighbor)).to(x_ids.device)
		# x_ids_ = pad + list(x_ids) + pad
		x_ids_ = torch.cat([pad, x_ids, pad], dim=-1)
		for i in range(nb_neighbor, L + nb_neighbor):
			# x = x_ids_[i - nb_neighbor: i] + x_ids_[i + 1: i + nb_neighbor + 1]
			neighbours[i-nb_neighbor, :, :] = torch.cat([x_ids_[:, i - nb_neighbor: i], x_ids_[:, i + 1: i + nb_neighbor + 1]], dim=-1)
			# ew_ids[i-nb_neighbor,:,:] = (x_ids[i-nb_neighbor,:] -1) * self.vocab_size + nb_neighbor[i-nb_neighbor,:,:]
		neighbours=neighbours.permute(1,0,2).to(x_ids.device)
		ew_ids = ((x_ids)*(self.vocab_size)).reshape(B,L,1) + neighbours
		ew_ids[neighbours== 0] = 0
		return neighbours, ew_ids


class BERT_TC(BertPreTrainedModel):
	def __init__(self, config, **kwargs):
		super().__init__(config)
		self.p = kwargs["p"]
		self.class_num = kwargs["class_num"]
		self.embed_size = config.hidden_size
		self.dropout_rate = kwargs["dropout_rate"]
		self.plm_config = config
		self.vocab_size = config.vocab_size
		self.edge_weight = nn.Embedding((self.vocab_size) * (self.vocab_size), 1, padding_idx=0)
		# self.node_embedding = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=0)
		# self.bert = BertModel(self.plm_config)
		self.node_embedding = BertModel(self.plm_config).embeddings.word_embeddings
		self.node_weight = nn.Embedding(self.vocab_size, 1, padding_idx=0)
		self.fc = nn.Sequential(
			nn.Linear(self.embed_size, self.class_num),
			nn.ReLU(),
			nn.Dropout(self.dropout_rate),
			nn.LogSoftmax(dim=1)
		)
		self.reset_params()
		if kwargs["criteration"] == "CrossEntropyLoss":
			self.criteration = nn.CrossEntropyLoss()
		else:
			# default loss
			self.criteration = nn.CrossEntropyLoss()


	def reset_params(self):
		nn.init.xavier_uniform_(self.edge_weight.weight)
		nn.init.xavier_uniform_(self.node_weight.weight)

	def extra_repr(self) -> str:
		return 'bert word embedding dim:{}'.format(
			self.embed_size
		)

	# def forward(self, input_ids, labels, attention_mask=None, token_type_ids=None):
	# 	clf_input = self.embedding(input_ids.long())
	# 	logits = self.classifier(clf_input)
	# 	loss = self.criteration(logits, labels)
	# 	return loss, logits


	def forward(self, input_ids, attention_mask=None,token_type_ids=None, labels=None, ):
		'''
        :param X: (bz, max_seq_len)  sentence nodes
        :param NX: (bz, max_seq_len, neighbor_num)  neighbor nodes of each node in X
        :param EW: (bz, max_seq_len, neighbor_num)  neighbor weights of each node in X
        :return:
        '''
		# neighbor (bz, seq_len, neighbor_num, embed_dim)
		X = input_ids.long()
		NX, EW = self.get_neighbors(X, nb_neighbor=self.p)
		NX = NX.long()
		EW = EW.long()
		# NX = input_ids
		# EW = input_ids
		Ra = self.node_embedding(NX)
		# edge weight  (bz, seq_len, neighbor_num, 1)
		Ean = self.edge_weight(EW)
		# neighbor representation  (bz, seq_len, embed_dim)
		Mn = (Ra * Ean).max(dim=2)[0]  # max pooling
		# self representation (bz, seq_len, embed_dim)
		Rn = self.node_embedding(X)
		# self node weight  (bz, seq_len, 1)
		Nn = self.node_weight(X)
		# aggregate node features
		y = (1 - Nn) * Mn + Nn * Rn
		logits = self.fc(y.sum(dim=1))
		if labels is not None:
			loss = self.criteration(logits, labels)
		else:
			loss = None
		return loss, logits, labels

	def get_neighbors(self, x_ids, nb_neighbor=2):
		B, L = x_ids.size()
		neighbours = torch.zeros(size=(L, B, 2*nb_neighbor))
		ew_ids = torch.zeros(size=(L, B, 2*nb_neighbor))
		# pad = [0] * nb_neighbor
		pad = torch.zeros(size=(B,nb_neighbor)).to(x_ids.device)
		# x_ids_ = pad + list(x_ids) + pad
		x_ids_ = torch.cat([pad, x_ids, pad], dim=-1)
		for i in range(nb_neighbor, L + nb_neighbor):
			# x = x_ids_[i - nb_neighbor: i] + x_ids_[i + 1: i + nb_neighbor + 1]
			neighbours[i-nb_neighbor, :, :] = torch.cat([x_ids_[:, i - nb_neighbor: i], x_ids_[:, i + 1: i + nb_neighbor + 1]], dim=-1)
			# ew_ids[i-nb_neighbor,:,:] = (x_ids[i-nb_neighbor,:] -1) * self.vocab_size + nb_neighbor[i-nb_neighbor,:,:]
		neighbours=neighbours.permute(1,0,2).to(x_ids.device)
		ew_ids = ((x_ids)*(self.vocab_size)).reshape(B,L,1) + neighbours
		ew_ids[neighbours== 0] = 0
		return neighbours, ew_ids