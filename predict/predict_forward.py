import torch
from torch.nn import functional as F

def normalize(tensor):
    return tensor / torch.clamp(tensor.norm(dim=-1, keepdim=True), min=1e-12)


def attn_sim_lstm(loss_model, hidden, input_mask):
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

    score = loss_model.attn_fn(x_score, y_score) / loss_model.temperature # x * y * seq
    score = score.masked_fill(mask=~x_mask.unsqueeze(1).expand_as(score),value=float('-1e8'))
    attn = F.softmax(score, dim=2).unsqueeze(-1) # batch_size_x * batch_size_y * max_seq_length * 1

    x_expand = x.unsqueeze(-1).expand(x.size(0), x.size(1), x.size(2), y.size(1)) # batch_size_x * max_seq_length * h_dim * batch_size_y
    x_perm = x_expand.permute(0, 3, 2, 1) # batch_size_x * batch_size_y * h_dim * max_seq_length

    #print(hidden.shape, x.shape, x_perm.shape, y.shape, attn.shape)

    x_embed = torch.matmul(x_perm, attn).squeeze(-1)

    if not loss_model.cat:
        x_embed = normalize(x_embed) # batch_size_x * batch_size_y * h_dim
    else:
        x_embed = normalize(torch.cat((x_embed, x[:,-1,:].unsqueeze(1).expand_as(x_embed)), dim=-1))

    #y_embed = normalize(torch.sum(y, dim=1)) # batch_size_y * h_dim
    y_embed = y[:,:,-1,:] # x * y * dim
    #y_embed_expand = y_embed.unsqueeze(0).expand(x_embed.size(0), y_embed.size(0), y_embed.size(1)) # batch_size_x * batch_size_y * h_dim

    sim = loss_model.sim_fn(x_embed, y_embed) # x * y

    return sim



def predict_forward_lstm(model, input_ids_x, input_ids_y, x_mask, y_mask):
    input_ids_x = input_ids_x.unsqueeze(1)
    x_mask = x_mask.unsqueeze(1)
    input_ids = torch.cat((input_ids_x, input_ids_y), dim=1) # batch_size_x * (1 + batch_size_y) * max_seq_length
    input_mask = torch.cat((x_mask, y_mask), dim=1)
    hidden = model._get_bert_feature_token(input_ids)
    sim = attn_sim_lstm(model.loss_fn, hidden, input_mask)
    return sim
