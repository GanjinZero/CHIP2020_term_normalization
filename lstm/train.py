import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from torch import nn
#from data_util import load_train_test
from sampler_util import MyDataset, my_dataloader
from model import tokenAttnModel
from tensorboardX import SummaryWriter
from tqdm import tqdm
import shutil
import ipdb
import sys
sys.path.append("../")
from rule_base.load_icd_v3 import load_train_icd


# Setting Parameters
max_seq_length = 32  # truncate to 48-64 for higher training speed!
# actually longest sentence in train = 103
# longest icd des in train = 36
# longest icd des in icd10 = 71
# 128-56 64-128

#dropout = 0.5

batch_size = 1024
l_rate = 1e-3
epoch_num = 150 # 100
device = torch.device("cuda")#:1")
pos_weight = 5

num_workers = 1

embedding_path = "/media/sdc/GanjinZero/jiangsu_info/word2vec_5_300.model"
#topk_dict_path = "/media/sdd1/Hongyi_Yuan/CHIP2020/Final/result_dict_test/Dice_tfidfjson.pkl"
topk_dict_path = "/media/sdd1/Hongyi_Yuan/CHIP2020/Final/result_dict_test/dice_train_final_v3.pkl"

gradient_accumulation_steps = 1
max_grad_norm = 1.0
check_test_step = 1000
save_step = 5000

# Load data
#x_train, y_train, x_test, y_test = load_train_test(test_size=100, use_icd10=use_icd10, use_additional_icd=use_additional_icd)
x_list, y_clean_list, y_list, y_clean_icd10_list, y_icd10_list = load_train_icd()
x_train = x_list#[0:7000]
y_train = y_clean_icd10_list#[0:7000]

total_train_step = int(len(x_train) / batch_size * epoch_num * 1.1)

my_dataset = MyDataset(x_train, y_train, embedding_path, topk_dict_path=topk_dict_path, max_length=max_seq_length)
my_dataloader = my_dataloader(my_dataset, fixed_length=batch_size, num_workers=num_workers)

"""
for idx, batch in enumerate(my_dataloader):
    print(idx, batch)
    if idx > 5:
        import sys; sys.exit()

import ipdb; ipdb.set_trace()
"""

# Prepare model and optimizer
model = tokenAttnModel(embedding_path, pos_weight=pos_weight, device=device).to(device)

for name, param in model.named_parameters(): #查看可优化的参数有哪些
    print(name, param.requires_grad)
optimizer = torch.optim.Adam(model.parameters(), lr=l_rate)

parameter = ['final', batch_size, epoch_num, l_rate, pos_weight, embedding_path.split("/")[-1].split(".")[0], topk_dict_path.split("/")[-1].split(".")[0], "full"]#, dropout]
output_name = "_".join([str(x) for x in parameter])
try:
    shutil.rmtree(output_name)
except BaseException:
    pass
os.makedirs(os.path.join(".", output_name))

# Training
writer = SummaryWriter(comment='CHIP2020_eval3')

train_step = 0

model.zero_grad()

for epoch_index in range(epoch_num):
    model.train()
    epoch_loss = 0.
    epoch_pos_wrong = 0
    epoch_neg_wrong = 0
    epoch_pos_count = 0
    epoch_neg_count = 0
    epoch_iterator = tqdm(my_dataloader, desc="Iteration")
    for index, batch in enumerate(epoch_iterator):
        optimizer.zero_grad()
        loss, (_, _, _, pos_wrong, neg_wrong, pos_count, neg_count) = model(batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device), batch[4].to(device))

        epoch_loss += loss.item()
        train_step += 1
        
        epoch_pos_wrong += pos_wrong
        epoch_neg_wrong += neg_wrong
        epoch_pos_count += pos_count
        epoch_neg_count += neg_count

        epoch_iterator.set_description("epoch_index: %s, index: %s, batch_loss: %0.4f, epoch_loss: %0.4f, pos_err: %0.4f, neg_err: %0.4f" %
                                        (epoch_index, index, loss.item(), epoch_loss / (index + 1), epoch_pos_wrong/epoch_pos_count, epoch_neg_wrong/epoch_neg_count))
        writer.add_scalar('batch_loss', loss.item(), global_step=train_step)
        writer.add_scalar('epoch_loss', epoch_loss / (index + 1), global_step=train_step)

        #ipdb.set_trace()
        
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        loss.backward()

        if (train_step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm)
            optimizer.step()
            model.zero_grad()

        if train_step % check_test_step == 0:
            model.eval()
            with torch.no_grad():
                model.check_sim_attn(batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device), batch[4].to(device))
            model.train()

        if train_step % save_step == 0:
            torch.save(model, os.path.join(output_name, f"model_{train_step}.pth"))

torch.save(model, os.path.join(output_name, f"model_last.pth"))
