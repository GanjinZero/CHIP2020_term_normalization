import torch
from torch import nn
from torch.nn import functional as F
from tokenize_util import load_w2v, tokenize, decode
import ipdb


def normalize(tensor):
    return tensor / torch.clamp(tensor.norm(dim=-1, keepdim=True), min=1e-12)

class Similarity(nn.Module):
    def __init__(self, sim_type, x_h_dim=None, y_h_dim=None):
        super(Similarity, self).__init__()
        self.sim_type = sim_type
        self.x_h_dim = x_h_dim
        self.y_h_dim = y_h_dim
        if self.sim_type == "perceptron":
            self.linear = nn.Linear(self.x_h_dim + self.y_h_dim, self.x_h_dim + self.y_h_dim)
            self.out = nn.Linear(self.x_h_dim + self.y_h_dim, 1)
        if self.sim_type == "general":
            self.w = nn.Linear(self.x_h_dim, self.x_h_dim)

    def forward(self, x, y):
        """
        x and y should be normalized
        x: batch_size_x * batch_size_y * h_dim
        y: 1 * batch_size_y * h_dim
        """
        if self.sim_type == "dot":
            return torch.sum(torch.mul(x, y), dim=-1)
        if self.sim_type == "perceptron":
            hidden = torch.tanh(self.linear(torch.cat((x, y), dim=-1)))
            sim = self.out(hidden).squeeze(-1)
            return sim
        if self.sim_type == "general":
            hidden = self.w(x)
            return torch.sum(torch.mul(x, y), dim=-1)
        return None

class Attention(nn.Module):
    def __init__(self, att_type, x_h_dim=None, y_h_dim=None):
        super(Attention, self).__init__()
        self.att_type = att_type
        self.x_h_dim = x_h_dim
        self.y_h_dim = y_h_dim
        # perceptron attention will oom
        if self.att_type == "general":
            self.w = nn.Linear(self.x_h_dim, self.x_h_dim)

    def forward(self, x, y):
        """
        x and y should be normalized
        x: batch_size_x * 1 * seq * h_dim
        y: batch_size_x * batch_size_y * h_dim
        return: batch_size_x * batch_size_y * seq
        """
        if self.att_type == "general":
            x = self.w(x)
        return torch.matmul(x, y.unsqueeze(-1)).squeeze(-1)

class attnLoss(nn.Module):
    def __init__(self, temperature=1.0, pos_weight=10.0, sim_type="dot", att_type="dot", input_dim=300, cat=True):
        """
        emb -> lstm
        lstm -> lstm_attention
        """
        super(attnLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')
        self.temperature = temperature # 1.0
        self.pos_weight = pos_weight # 1.0
        self.attn_fn = Attention(att_type, x_h_dim=input_dim, y_h_dim=input_dim)
        
        self.input_dim = input_dim
        self.dim_back = self.input_dim
        self.cat = cat
        if self.cat:
            self.dim_back = self.input_dim * 2

        self.sim_fn = Similarity(sim_type, x_h_dim=self.dim_back, y_h_dim=self.input_dim)

    def forward(self, hidden, y_label, input_mask):
        """
        hidden: batch_size_x * (1 + batch_size_y) * max_seq_length * h_dim
        y_label: batch_size_x * batch_size_y
        input_mask: batch_size_x * (1 + batch_size_y) * max_seq_length
        """

        hidden = normalize(hidden)
        hidden = hidden.masked_fill(mask=~input_mask.unsqueeze(-1).expand_as(hidden), value=torch.tensor(0))

        x = hidden[:,0,:,:] # x * seq * dim
        y = hidden[:,1:,:,:] # x * y * seq * dim

        #h[x][0][m][q]
        #h[x][i][n][q]
        #h[x][i][m] = \sum n,q h[x][0][m][q] * h[x][i][n][q]

        """
        a[x][p][q]
        b[y][i][q]
        c[x][y][p] = \sum i,q a_xpq * b_yiq
        """

        x_score = x.unsqueeze(1) # x * 1 * seq * dim
        y_score = torch.sum(y, dim=2) # x * y * dim

        x_mask = input_mask[:,0,:] # x * seq

        score = self.attn_fn(x_score, y_score) / self.temperature # x * y * seq
        score = score.masked_fill(mask=~x_mask.unsqueeze(1).expand_as(score),value=float('-1e8'))
        attn = F.softmax(score, dim=2).unsqueeze(-1) # batch_size_x * batch_size_y * max_seq_length * 1

        x_expand = x.unsqueeze(-1).expand(x.size(0), x.size(1), x.size(2), y.size(1)) # batch_size_x * max_seq_length * h_dim * batch_size_y
        x_perm = x_expand.permute(0, 3, 2, 1) # batch_size_x * batch_size_y * h_dim * max_seq_length

        #print(hidden.shape, x.shape, x_perm.shape, y.shape, attn.shape)

        x_embed = torch.matmul(x_perm, attn).squeeze(-1)

        if not self.cat:
            x_embed = normalize(x_embed) # batch_size_x * batch_size_y * h_dim
        else:
            x_embed = normalize(torch.cat((x_embed, x[:,-1,:].unsqueeze(1).expand_as(x_embed)), dim=-1))

        #y_embed = normalize(torch.sum(y, dim=1)) # batch_size_y * h_dim
        y_embed = y[:,:,-1,:] # x * y * dim
        #y_embed_expand = y_embed.unsqueeze(0).expand(x_embed.size(0), y_embed.size(0), y_embed.size(1)) # batch_size_x * batch_size_y * h_dim

        sim = self.sim_fn(x_embed, y_embed) # x * y
        mse_loss = self.mse_loss(sim, y_label)
        weight = torch.ones_like(sim)
        weight[y_label==1] = self.pos_weight
        mse = torch.mean(torch.mul(mse_loss, weight))
        mse = torch.mean(mse_loss)

        #print(sim.shape, y_label.shape)
        pos_wrong = sum(sim[y_label==1.] < 0).item()
        neg_wrong = sum(sim[y_label==-1.] > 0).item()
        pos_count = len(y_label[y_label==1.])
        neg_count = len(y_label[y_label==-1.])

        return mse, (attn.squeeze(-1), sim, y_label, pos_wrong, neg_wrong, pos_count, neg_count)

class tokenAttnModel(nn.Module):
    def __init__(self, word2vec_path, use_lstm=True, lstm_hidden_dim=768, att_type="general", temperature=1.0, pos_weight=10.0, 
                 sim_type="perceptron", cat=True, device=torch.device("cuda:1")):#, dropout=0.5):
        super(tokenAttnModel, self).__init__()

        #self.dropout = dropout
        #self.embedding_dropout = nn.Dropout(p=self.dropout)
        
        self.device = device
        self.word_vector, self.word2id, self.id2word = load_w2v(word2vec_path)
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(self.word_vector))
        self.embedding.weight.requires_grad_()

        self.use_lstm = use_lstm
        self.lstm_hidden_dim = lstm_hidden_dim

        if use_lstm:
            self.lstm = nn.LSTM(self.word_vector.shape[1], self.lstm_hidden_dim // 2, batch_first=True, num_layers=2, bidirectional=True)#, dropout=self.dropout)
            self.dim_front = self.lstm_hidden_dim
        else:
            self.dim_front = self.word_vector.shape[1]

        self.cat = cat

        self.loss_fn = attnLoss(temperature=temperature, pos_weight=pos_weight,
                                sim_type=sim_type, att_type=att_type, input_dim=self.dim_front, cat=self.cat)

    def normalize(self, tensor):
        return tensor / tensor.norm(dim=-1, keepdim=True)

    def _get_bert_feature_token(self, input_ids):
        emb = self.embedding(input_ids)
        #emb = self.embedding_dropout(emb)

        if not self.use_lstm:
            return emb

        # reshape
        batch_size_x = emb.size(0)
        batch_size_y_1 = emb.size(1)
        seq_length = emb.size(2)
        h_dim = emb.size(3)
        emb_reshape = emb.reshape(-1, seq_length, h_dim)

        h, (_, _) = self.lstm(emb_reshape)
        #print(input_ids.shape, h.shape)
        h_reshape = h.reshape(batch_size_x, batch_size_y_1, seq_length, self.lstm_hidden_dim)
        return h_reshape

    def forward(self, input_ids_x, input_ids_y, y_label, x_mask, y_mask):
        """
        input_ids_x: batch_size_x * max_seq_length
        input_ids_y: batch_size_x * batch_size_y * max_seq_length
        y_label: batch_size_x * batch_size_y
        """
        input_ids_x = input_ids_x.unsqueeze(1)
        x_mask = x_mask.unsqueeze(1)
        input_ids = torch.cat((input_ids_x, input_ids_y), dim=1) # batch_size_x * (1 + batch_size_y) * max_seq_length
        input_mask = torch.cat((x_mask, y_mask), dim=1)

        hidden = self._get_bert_feature_token(input_ids) # batch_size_x * (1 + batch_size_y) * max_seq_length * h_dim

        loss = self.loss_fn(hidden, y_label, input_mask)

        return loss

    # Need modify!!!
    def check_sim_attn(self, input_ids_x, input_ids_y, y_label, x_mask, y_mask):
        # Check similarity and attention
        # should be used under torch.no_grad() and model.eval()

        # with torch.no_grad():
        batch_size_x = input_ids_x.size(0)
        batch_size_y = input_ids_y.size(1)
        loss, other = self.forward(input_ids_x, input_ids_y, y_label, x_mask, y_mask)

        attn_perm = other[0]
        sim = other[1]
        mask = other[2]

        for idx in range(batch_size_x):
            print(self.decode(input_ids_x[idx]))
            for idy in range(batch_size_y):
                print(self.decode(input_ids_y[idx][idy]), sim[idx][idy].item())
                print(attn_perm[idx][idy][x_mask[idx]==1])
            print("-----------")
        return None


    def decode(self, input_id):
        return decode([[id.item() for id in input_id]], self.id2word)
