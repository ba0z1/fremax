import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel
import torch
import torch.nn as nn
# import torch_scatter
import pickle as pkl
import numpy as np


class CNN_base(nn.Module):
    def __init__(self, in_features, filter_num, filter_size, class_num, dropout_rate, ):
        super(CNN_base, self).__init__()

        self.cnn_list = nn.ModuleList()
        for size in filter_size:
            self.cnn_list.append(nn.Conv1d(in_features, filter_num, size))
        self.relu = nn.ReLU()
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.out_hidden_dim = len(filter_size) * filter_num

    def forward(self, word_feature):
        _ = word_feature.permute(0, 2, 1)
        result = []

        for self.cnn in self.cnn_list:
            __ = self.cnn(_)
            __ = self.relu(__)
            __ = self.max_pool(__)
            result.append(__.squeeze(dim=2))

        _ = torch.cat(result, dim=1)
        _ = self.dropout(_)
        return _


class GNN_base(nn.Module):
    def __init__(self, in_features, vocab_size, class_num, dropout_rate, p):
        super(GNN_base, self).__init__()
        self.p = p
        self.vocab_size = vocab_size
        self.edge_weight = nn.Embedding((vocab_size) * (vocab_size) +1, 1, padding_idx=0)
        self.node_embedding = nn.Embedding(vocab_size, in_features, padding_idx=0)
        self.node_weight = nn.Embedding(vocab_size, 1, padding_idx=0)
        # self.fc = nn.Sequential(
        #     nn.Linear(in_features, class_num),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_rate),
        #     nn.LogSoftmax(dim=1)
        # )
        self.reset_params()

    def reset_params(self):
        nn.init.xavier_uniform_(self.edge_weight.weight)
        nn.init.xavier_uniform_(self.node_weight.weight)

    def forward(self, input_ids,):
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
        return y.sum(dim=1)
        # logits = self.fc(y.sum(dim=1))


    def get_neighbors(self, x_ids, nb_neighbor=2):
        B, L = x_ids.size()
        neighbours = torch.zeros(size=(L, B, 2 * nb_neighbor))
        ew_ids = torch.zeros(size=(L, B, 2 * nb_neighbor))
        # pad = [0] * nb_neighbor
        pad = torch.zeros(size=(B, nb_neighbor)).to(x_ids.device)
        # x_ids_ = pad + list(x_ids) + pad
        x_ids_ = torch.cat([pad, x_ids, pad], dim=-1)
        for i in range(nb_neighbor, L + nb_neighbor):
            # x = x_ids_[i - nb_neighbor: i] + x_ids_[i + 1: i + nb_neighbor + 1]
            neighbours[i - nb_neighbor, :, :] = torch.cat(
                [x_ids_[:, i - nb_neighbor: i], x_ids_[:, i + 1: i + nb_neighbor + 1]], dim=-1)
        # ew_ids[i-nb_neighbor,:,:] = (x_ids[i-nb_neighbor,:] -1) * self.vocab_size + nb_neighbor[i-nb_neighbor,:,:]
        neighbours = neighbours.permute(1, 0, 2).to(x_ids.device)
        ew_ids = ((x_ids) * (self.vocab_size)).reshape(B, L, 1) + neighbours
        ew_ids[neighbours == 0] = 0
        return neighbours, ew_ids


class FCN_base(nn.Module):
    def __init__(self, in_features, class_num, dropout_rate, ):
        super(FCN_base, self).__init__()
        # self.in_features = in_features
        self.dropout_prob = dropout_rate
        # self.num_labels = class_num  #
        self.dropout = nn.Dropout(self.dropout_prob)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_hidden_dim = out_hidden_dim = in_features
        # self.classifier = nn.Linear(self.in_features, self.num_labels)

    def forward(self, features):
        clf_input = self.pool(features.permute(0, 2, 1)).squeeze()
        return clf_input


class RNN_base(nn.Module):
    def __init__(self, cell, in_features, hidden_dim, num_layers, class_num, dropout_rate, ):
        super(RNN_base, self).__init__()
        self._cell = cell
        self.in_features = in_features
        # self.class_num = class_num
        self.rnn = None
        if cell == 'rnn':
            self.rnn = nn.RNN(in_features, hidden_dim, num_layers, dropout=dropout_rate)
            out_hidden_dim = hidden_dim * num_layers
        elif cell == 'bi-rnn':
            self.rnn = nn.RNN(in_features, hidden_dim, num_layers, dropout=dropout_rate, bidirectional=True)
            out_hidden_dim = 2 * hidden_dim * num_layers
        elif cell == 'gru':
            self.rnn = nn.GRU(in_features, hidden_dim, num_layers, dropout=dropout_rate)
            out_hidden_dim = hidden_dim * num_layers
        elif cell == 'bi-gru':
            self.rnn = nn.GRU(in_features, hidden_dim, num_layers, dropout=dropout_rate, bidirectional=True)
            out_hidden_dim = 2 * hidden_dim * num_layers
        elif cell == 'lstm':
            self.rnn = nn.LSTM(in_features, hidden_dim, num_layers, dropout=dropout_rate)
            out_hidden_dim = 1 * hidden_dim * num_layers
        elif cell == 'bi-lstm':
            self.rnn = nn.LSTM(in_features, hidden_dim, num_layers, dropout=dropout_rate, bidirectional=True)
            out_hidden_dim = 2 * hidden_dim * num_layers
        else:
            raise Exception("no such rnn cell")
        self.out_hidden_dim = out_hidden_dim
        # self.output_layer = nn.Linear(self.out_hidden_dim, class_num)
        # self.softmax = nn.Softmax(dim=1)
        # self.output_layer = nn.Linear(out_hidden_dim, class_num)

    def forward(self, features):
        _ = features.permute(1, 0, 2)
        __, h_out = self.rnn(_)
        if self._cell in ["lstm", "bi-lstm"]:
            # h_out = torch.cat([h_out[0], h_out[1]], dim=2)
            h_out = h_out[-1]
        h_out = h_out.permute(1, 0, 2)
        h_out = h_out.reshape(-1, self.out_hidden_dim)
        return h_out
        # logits = self.output_layer(h_out)
        # return logits


class LSTMATT_base(nn.Module):
    def __init__(self, in_features, hidden_dim, class_num, dropout_rate, bidirectional):
        super(LSTMATT_base, self).__init__()
        self.in_features = in_features
        self.dropout_prob = dropout_rate
        self.num_labels = class_num
        self.hidden_size = hidden_dim
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(self.dropout_prob)
        self.lstm = nn.LSTM(self.in_features, self.hidden_size, 1, dropout=self.dropout_prob, bidirectional=True)
        self.attn = nn.MultiheadAttention(embed_dim=self.hidden_size * 2, num_heads=1, dropout=self.dropout_prob)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_hidden_dim = self.hidden_size * 2
        # self.classifier = nn.Linear(self.out_hidden_dim, self.num_labels)

    def forward(self, features, mask, input_ids_len):
        output, _ = self.lstm(features.permute(1, 0, 2))
        output = self.dropout(output)
        output = self.attn(output, output, output, key_padding_mask=mask.bool())[0].permute(1, 0, 2).permute(0, 2, 1)
        return self.pool(output).squeeze()
        # logits = self.classifier(self.pool(output).squeeze())
        # return logits


class RBC_base(nn.Module):
    def __init__(self, in_features, class_num, num_layers, hidden_size, Ci, kernel_num, kernel_sizes, LSTM_dropout,
                 CNN_dropout):
        super(RBC_base, self).__init__()
        self.in_features = in_features
        self.class_num = class_num
        D = in_features
        C = class_num
        N = num_layers
        H = hidden_size
        Ci = Ci
        Co = kernel_num
        Ks = kernel_sizes
        self.lstm = nn.LSTM(D, H, num_layers=N, \
                            bidirectional=True,
                            batch_first=True,
                            dropout=LSTM_dropout)

        self.conv1_D = nn.Conv2d(Ci, Co, (1, 2 * H))

        self.convK_1 = nn.ModuleList(
            [nn.Conv2d(Co, Co, (K, 1)) for K in Ks])

        self.conv3 = nn.Conv2d(Co, Co, (3, 1))

        self.conv4 = nn.Conv2d(Co, Co, (3, 1), padding=(1, 0))

        self.CNN_dropout = nn.Dropout(CNN_dropout)
        self.out_hidden_dim = len(Ks) * Co

    def forward(self, features):
        out, _ = self.lstm(features)  # [batch_size, sen_len, H*2]
        x = out.unsqueeze(1)
        x = self.conv1_D(x)

        x = [F.relu(conv(x)) for conv in self.convK_1]
        x3 = [F.relu(self.conv3(i)) for i in x]
        x4 = [F.relu(self.conv4(i)) for i in x3]
        inception = []
        for i in range(len(x4)):
            res = torch.add(x3[i], x4[i])
            inception.append(res)

        x = [i.squeeze(3) for i in inception]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)

        x = self.CNN_dropout(x)
        return x
        # logits = self.fc1(x)
        # return logits


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True, get_att=False):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.get_att = get_att
        self.dropout = dropout
        # self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)).cuda())
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        # self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)).cuda())
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        #print("input")
        #print(input.size())
        #print(self.W.size())
        #print("-----------------------------")
        h = torch.matmul(input, self.W)
        a_input = self._prepare_attentional_mechanism_input(h)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))

        zero_vec = -9e15 * torch.ones_like(e)

        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        B, M, E = Wh.shape  # (batch_zize, number_nodes, out_features)
        Wh_repeated_in_chunks = Wh.repeat_interleave(M, dim=1)  # (B, M*M, E)
        Wh_repeated_alternating = Wh.repeat(1, M, 1)  # (B, M*M, E)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=-1)  # (B, M*M,2E)
        return all_combinations_matrix.view(B, M, M, 2 * E)

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.in_features) + '->' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, n_feat, n_hid, out_features, dropout, alpha, n_heads):
        super(GAT, self).__init__()
        self.hidden = n_hid
        self.max_length = 128
        self.dropout = 0.1
        self.attentions = [GATLayer(n_feat, n_hid, dropout=self.dropout, alpha=alpha, concat=True, get_att=False) for _
                           in range(n_heads)]
        # self.attentions_adj = [GATLayer(n_feat, self.max_length,  alpha=alpha, concat=True,get_att=True) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GATLayer(n_hid * n_heads, out_features, dropout=self.dropout, alpha=alpha, concat=False)

    def forward(self, x_input, adj):
        x = F.dropout(x_input, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x


class SESY_base(nn.Module):
    def __init__(self, clf, TC_configs, in_features, class_num, hidden_dim, readout_size, gat_alpha, gat_heads,
                 dropout_rate, strategy="cas", ):
        super(SESY_base, self).__init__()
        self.embed_size = in_features
        self.clf_name = clf
        self.dropout_rate = dropout_rate
        self.class_num = class_num
        self.hidden_dim = hidden_dim
        self.readout_size = readout_size
        self.gat_alpha = gat_alpha
        self.gat_heads = gat_heads
        self.strategy = strategy

        if self.clf_name == "cnn":
            TC_base = CNN_base
            self.clf_configs = TC_configs.cnn
        elif self.clf_name == "fc":
            TC_base = FCN_base
            self.clf_configs = TC_configs.fc
        elif self.clf_name == "rnn":
            TC_base = RNN_base
            self.clf_configs = TC_configs.rnn
        else:
            assert 0, "No such clf, only support cnn rnn & fc"
        self.gat = GAT(
            n_feat=self.embed_size,
            n_hid=self.hidden_dim,
            out_features=self.readout_size,
            alpha=self.gat_alpha,
            n_heads=self.gat_heads,
            dropout=self.dropout_rate
        )
        if self.strategy.lower() == "cas":
            self.clf_configs.in_features = self.readout_size
        elif self.strategy.lower() == "parl":
            self.clf_configs.in_features = self.readout_size + self.embed_size
        self.classifier = TC_base(
            **{**self.clf_configs, "class_num": self.class_num, "dropout_rate": self.dropout_rate})
        self.out_hidden_dim = self.classifier.out_hidden_dim
        # covert my dict to standard dict
        self.clf_configs = {**self.clf_configs}

    def forward(self, embedding, graph):
        gat_out = self.gat(embedding, graph)

        if self.strategy.lower() == "cas":
            features = self.classifier(gat_out)
        elif self.strategy.lower() == "parl":
            features = self.classifier(torch.cat([gat_out, embedding], dim=2))
        return features
        # loss = self.criteration(logits, labels)
        # return loss, logits


class TC(nn.Module):
    def __init__(self, vocab_size, tfe, tfe_configs, class_num, readout_size, gat_alpha,
                 gat_heads, dropout_rate, concept_embedding_path, criteration="CrossEntropyLoss", ):
        super(TC, self).__init__()
        TFE_configs = tfe_configs
        # print(TFE_configs)
        self.vocab_size = vocab_size+1 if tfe.lower()  == "sesy" else vocab_size
        #self.embed_size = tfe_configs.tfe.hidden_size
        self.TFE_name = tfe
        #self.embed_size = TFE_configs.TFE_name.hidden_size
        self.dropout_rate = dropout_rate
        self.class_num = class_num
        self.readout_size = readout_size
        self.gat_alpha = gat_alpha
        self.gat_heads = gat_heads
        self.concept_embedding_path = concept_embedding_path
        self.num = 0

        if self.TFE_name == "cnn":
            self.TFE_configs = TFE_configs.cnn
            self.TFE_configs.in_features = 128
            # self.text_out_features = self.TFE_configs.in_features * len(self.TFE_configs.filter_size)
            self.text_feature_extractor = CNN_base(
                **{**self.TFE_configs, "class_num": self.class_num, "dropout_rate": self.dropout_rate})
            self.text_out_features = self.text_feature_extractor.out_hidden_dim
        elif self.TFE_name == "fcn":
            self.TFE_configs = TFE_configs.fc
            self.TFE_configs.in_features = 128
            # self.text_out_features = self.embed_size
            self.text_feature_extractor = FCN_base(
                **{**self.TFE_configs, "class_num": self.class_num, "dropout_rate": self.dropout_rate})
            self.text_out_features = self.text_feature_extractor.out_hidden_dim
        elif self.TFE_name == "rnn":
            self.TFE_configs = TFE_configs.rnn
            self.TFE_configs.in_features = 128
            self.text_feature_extractor = RNN_base(
                **{**self.TFE_configs, "class_num": self.class_num, "dropout_rate": self.dropout_rate})
            self.text_out_features = self.text_feature_extractor.out_hidden_dim
        elif self.TFE_name == "lstmatt":
            self.TFE_configs = TFE_configs.lstmatt
            #self.TFE_configs.in_features = self.TFE_configs.hidden_dim
            self.TFE_configs.in_features = 128

            self.text_feature_extractor = LSTMATT_base(
                **{**self.TFE_configs, "class_num": self.class_num, "dropout_rate": self.dropout_rate})
            self.text_out_features = self.text_feature_extractor.out_hidden_dim
        elif self.TFE_name == "sesy":
            self.TFE_configs = TFE_configs.sesy
            self.TFE_configs.in_features = 128
            self.text_feature_extractor = SESY_base(
                **{**self.TFE_configs, "class_num": self.class_num, "dropout_rate": self.dropout_rate})
            self.text_out_features = self.text_feature_extractor.out_hidden_dim
        # elif self.TFE_name == "rbc":
        #     self.TFE_configs = TFE_configs.rbc
        #     self.TFE_configs.in_features = self.self.TFE_configs.embed_dim
        #     self.text_feature_extractor = RBC_base(
        #         **{**self.TFE_configs, "class_num": self.class_num, })
        #     self.text_out_features = self.text_feature_extractor.out_hidden_dim
        elif self.TFE_name == "gnn":
            self.TFE_configs = TFE_configs.gnn
            self.TFE_configs.in_features = 128
            self.text_feature_extractor = GNN_base(**{**self.TFE_configs, "class_num": self.class_num,"vocab_size":self.vocab_size })
            self.text_out_features = self.TFE_configs.in_features
        else:
            assert 0, "No such text features extractor, only support cnn rnn & fc"

        concept_embedding = pkl.load(open(self.concept_embedding_path, "rb"))
        self.concept_embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(torch.from_numpy(concept_embedding["embedding"].astype(np.float32))))
        # self.concept_embedding = nn.Embedding(len(list(concept_embedding["vocab"].items())), 128)
        # self.concept_embedding = nn.Embedding(10000, 128)
        self.embedding = nn.Embedding(self.vocab_size, 128)

        self.gat = GAT(
            n_feat=300,
            n_hid=self.readout_size,
            out_features=self.readout_size,
            alpha=self.gat_alpha,
            n_heads=self.gat_heads,
            dropout=self.dropout_rate
        )

        self.output_layer = nn.Linear(self.text_out_features + self.readout_size, self.class_num)
        self.normalization = nn.BatchNorm1d(self.text_out_features + self.readout_size)
        if criteration == "CrossEntropyLoss":
            self.criteration = nn.CrossEntropyLoss()
            # self.criteration = focal_loss()
        else:
            # default loss
            self.criteration = nn.CrossEntropyLoss()
            # self.criteration = focal_loss()

    def forward(self, input_ids, concept_ids, head=None, tail=None, relation=None, distance=None, triple_label=None,
                attention_mask=None, token_type_ids=None,
                labels=None, graph=None, concepts_adj=None):
        embedding = self.embedding(input_ids)
        #print(embedding.size())
        #print(concept_ids.size())
        #print(concepts_adj.size())
        node_repr = self.gat(self.concept_embedding(concept_ids), concepts_adj)
        gat_out_pool = node_repr[:, 0, :]

        if self.TFE_name == "lstmatt":
            input_ids_len = torch.sum(input_ids != 0, dim=-1).float()
            mask = torch.ones_like(input_ids.long())
            mask[input_ids.long() != 0] = 0
            text_features = self.text_feature_extractor(embedding, mask, input_ids_len)
        elif self.TFE_name == "sesy":
            text_features = self.text_feature_extractor(embedding, graph)
        elif self.TFE_name == "gnn":
            text_features = self.text_feature_extractor(input_ids)
        else:
            text_features = self.text_feature_extractor(embedding)

        features = torch.cat([text_features, gat_out_pool], dim=1)
        # features = torch.cat([text_features, gat_out_pool[:,self.num,:], ], dim=1)
        # features = self.normalization(features)
        logits = self.output_layer(features)
        loss = self.criteration(logits, labels)
        return loss, logits, labels


class BERT_TC(BertPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        TFE_configs = kwargs["tfe_configs"]
        # print(TC_configs)
        self.embed_size = config.hidden_size
        self.TFE_name = kwargs["tfe"]
        self.dropout_rate = kwargs["dropout_rate"]
        self.class_num = kwargs["class_num"]
        self.readout_size = kwargs["readout_size"]
        self.gat_alpha = kwargs["gat_alpha"]
        self.gat_heads = kwargs["gat_heads"]
        self.concept_embedding_path = kwargs["concept_embedding_path"]
        self.bert_config = config
        self.num = 0

        if self.TFE_name == "cnn":
            self.TFE_configs = TFE_configs.cnn
            self.TFE_configs.in_features = self.embed_size
            # self.text_out_features = self.TFE_configs.in_features * len(self.TFE_configs.filter_size)
            self.text_feature_extractor = CNN_base(
                **{**self.TFE_configs, "class_num": self.class_num, "dropout_rate": self.dropout_rate})
            self.text_out_features = self.text_feature_extractor.out_hidden_dim
        elif self.TFE_name == "fcn":
            self.TFE_configs = TFE_configs.fc
            self.TFE_configs.in_features = self.embed_size
            # self.text_out_features = self.embed_size
            self.text_feature_extractor = FCN_base(
                **{**self.TFE_configs, "class_num": self.class_num, "dropout_rate": self.dropout_rate})
            self.text_out_features = self.text_feature_extractor.out_hidden_dim

        elif self.TFE_name == "rnn":
            self.TFE_configs = TFE_configs.rnn
            self.TFE_configs.in_features = self.embed_size
            self.text_feature_extractor = RNN_base(
                **{**self.TFE_configs, "class_num": self.class_num, "dropout_rate": self.dropout_rate})
            self.text_out_features = self.text_feature_extractor.out_hidden_dim
        elif self.TFE_name == "lstmatt":
            self.TFE_configs = TFE_configs.lstmatt
            self.TFE_configs.in_features = self.embed_size

            self.text_feature_extractor = LSTMATT_base(
                **{**self.TFE_configs, "class_num": self.class_num, "dropout_rate": self.dropout_rate})
            self.text_out_features = self.text_feature_extractor.out_hidden_dim
        elif self.TFE_name == "sesy":
            self.TFE_configs = TFE_configs.sesy
            self.TFE_configs.in_features = self.embed_size
            self.text_feature_extractor = SESY_base(
                **{**self.TFE_configs, "class_num": self.class_num, "dropout_rate": self.dropout_rate})
            self.text_out_features = self.text_feature_extractor.out_hidden_dim
        elif self.TFE_name == "rbc":
            self.TFE_configs = TFE_configs.rbc
            self.TFE_configs.in_features = self.embed_size
            self.text_feature_extractor = RBC_base(
                **{**self.TFE_configs, "class_num": self.class_num, })
            self.text_out_features = self.text_feature_extractor.out_hidden_dim
        else:
            assert 0, "No such text features extractor, only support cnn rnn & fc"

        concept_embedding = pkl.load(open(self.concept_embedding_path, "rb"))
        # concept_size, concept_dim = concept_embedding["embedding"].shape
        self.concept_embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(torch.from_numpy(concept_embedding["embedding"].astype(np.float32))))
        #self.bert = BertModel(config) #文字embedding
        self.embedding = nn.Embedding(50000, self.embed_size)

        self.gat = GAT(
            n_feat=300,
            n_hid=self.readout_size,
            out_features=self.readout_size,
            alpha=self.gat_alpha,
            n_heads=self.gat_heads,
            dropout=self.dropout_rate
        )

        self.output_layer = nn.Linear(self.text_out_features + self.readout_size, self.class_num)
        self.normalization = nn.BatchNorm1d(self.text_out_features + self.readout_size)
        if kwargs["criteration"] == "CrossEntropyLoss":
            self.criteration = nn.CrossEntropyLoss()
            # self.criteration = focal_loss()
        else:
            # default loss
            self.criteration = nn.CrossEntropyLoss()
            # self.criteration = focal_loss()

    def forward(self, input_ids, concept_ids, head=None, tail=None, relation=None, distance=None, triple_label=None,
                attention_mask=None, token_type_ids=None,
                labels=None, graph=None, concepts_adj=None):
        embedding = self.embedding(input_ids)

        node_repr = self.gat(self.concept_embedding(concept_ids), concepts_adj)
        gat_out_pool = node_repr[:, 0, :]

        if self.TFE_name == "lstmatt":
            input_ids_len = torch.sum(input_ids != 0, dim=-1).float()
            mask = torch.ones_like(input_ids.long())
            mask[input_ids.long() != 0] = 0
            text_features = self.text_feature_extractor(embedding, mask, input_ids_len)
        elif self.TFE_name == "sesy":
            text_features = self.text_feature_extractor(embedding, graph)
        else:
            text_features = self.text_feature_extractor(embedding)
        features = torch.cat([text_features, gat_out_pool], dim=1)
        #print(self.embed_size)
        # features = torch.cat([text_features, gat_out_pool[:,self.num,:], ], dim=1)
        # features = self.normalization(features)
        logits = self.output_layer(features)
        loss = self.criteration(logits, labels)
        return loss, logits, labels


