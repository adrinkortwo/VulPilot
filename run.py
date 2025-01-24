from __future__ import absolute_import
import os
import sys
import bleu as bleu
import pickle
import torch
import json
import random
import logging
import argparse
import dgl
import numpy as np
from io import open
from itertools import cycle
import torch.nn as nn
from model import Seq2Seq, HGTClassfication
from collections import defaultdict
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer, BartForConditionalGeneration, BartConfig,
                          BartTokenizer)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}
logging.basicConfig(filename="output/codebert/code/training.log",
                    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
root = os.path.dirname(__file__)
import nltk
from datasets import load_metric

class TrainDataset(Dataset):
    def __init__(self, source_ids, source_mask, train_graphs, target_ids, target_mask):
        self.source_ids = source_ids
        self.source_mask = source_mask
        self.train_graphs = train_graphs  # List of DGLGraph objects
        self.target_ids = target_ids
        self.target_mask = target_mask

    def __len__(self):
        return len(self.source_ids)

    def __getitem__(self, idx):
        return {
            'source_ids': self.source_ids[idx],
            'source_mask': self.source_mask[idx],
            'train_graph': self.train_graphs[idx],  # Returning DGLGraph
            'target_ids': self.target_ids[idx],
            'target_mask': self.target_mask[idx]
        }

def Train_collate_fn(batch):
    source_ids = torch.stack([item['source_ids'] for item in batch]).to("cuda")
    source_mask = torch.stack([item['source_mask'] for item in batch]).to("cuda")
    # 将每个 DGLGraph 移动到 GPU
    train_graph = [item['train_graph'].to("cuda") for item in batch]  # 转移 DGLGraph 到 GPU
    target_ids = torch.stack([item['target_ids'] for item in batch]).to("cuda")
    target_mask = torch.stack([item['target_mask'] for item in batch]).to("cuda")

    return source_ids, source_mask, train_graph, target_ids, target_mask

class TestDataset(Dataset):
    def __init__(self, source_ids, source_mask, test_graphs):
        self.source_ids = source_ids
        self.source_mask = source_mask
        self.test_graphs = test_graphs  # List of DGLGraph objects

    def __len__(self):
        return len(self.source_ids)

    def __getitem__(self, idx):
        return {
            'source_ids': self.source_ids[idx],
            'source_mask': self.source_mask[idx],
            'test_graph': self.test_graphs[idx],  # Returning DGLGraph
        }

def Test_collate_fn(batch):
    source_ids = torch.stack([item['source_ids'] for item in batch]).to("cuda")
    source_mask = torch.stack([item['source_mask'] for item in batch]).to("cuda")
    # 将每个 DGLGraph 移动到 GPU
    test_graph = [item['test_graph'].to("cuda") for item in batch]  # 转移 DGLGraph 到 GPU

    return source_ids, source_mask, test_graph

def GetDataset(commitid):
    path = f"testdata/{commitid}"  # 指定路径

    if not os.path.exists(path):
        print(f"[Error] Folder for commit ID {commitid} not found in testdata.")
        return None

    print('Start loading data!')

    # 寻找后缀为_codebert.npz的文件
    codebert_file = None
    for file in os.listdir(path):
        if file.endswith("_codebert.npz"):
            codebert_file = os.path.join(path, file)
            break

    if not codebert_file:
        print(f"[Error] _codebert.npz file not found in {path}.")
        return None

    # 加载graph数据
    graph = np.load(codebert_file, allow_pickle=True)

    # 异构图Transformer
    edgeIndex = graph['edgeIndex']
    src_node = edgeIndex[0]
    dst_node = edgeIndex[1]
    nodeAttr = graph['nodeAttr']
    edgeAttr = graph['edgeAttr']
    nodeType = graph['nodeType']

    # 节点分类
    nodeDict = {}
    keep_node, del_node, add_node = [], [], []
    keep_index, add_index, del_index = 0, 0, 0
    typeDict = {'-1': 'del', '0': 'keep', '1': 'add'}

    for id, type in enumerate(nodeType):
        if type == '0':
            nodeDict[id] = keep_index
            keep_node.append(torch.tensor(nodeAttr[id]))
            keep_index += 1
        elif type == '1':
            nodeDict[id] = add_index
            add_node.append(torch.tensor(nodeAttr[id]))
            add_index += 1
        elif type == '-1':
            nodeDict[id] = del_index
            del_node.append(torch.tensor(nodeAttr[id]))
            del_index += 1

    edges = defaultdict(lambda: ([], []))
    edge_no = 0
    for edge in edgeAttr:
        src = src_node[edge_no]
        dst = dst_node[edge_no]
        src_type = typeDict[nodeType[src]]
        dst_type = typeDict[nodeType[dst]]

        if np.array_equal(edge[-3:], np.array([1, 0, 0])):
            edges[(src_type, 'CDG', dst_type)][0].append(nodeDict[src])
            edges[(src_type, 'CDG', dst_type)][1].append(nodeDict[dst])
        elif np.array_equal(edge[-3:], np.array([0, 1, 0])):
            edges[(src_type, 'DDG', dst_type)][0].append(nodeDict[src])
            edges[(src_type, 'DDG', dst_type)][1].append(nodeDict[dst])
        elif np.array_equal(edge[-3:], np.array([0, 0, 1])):
            edges[(src_type, 'AST', dst_type)][0].append(nodeDict[src])
            edges[(src_type, 'AST', dst_type)][1].append(nodeDict[dst])

        edge_no += 1

    for edge_type_str, (src_list, dst_list) in edges.items():
        edges[edge_type_str] = (np.array(src_list), np.array(dst_list))

    # 创建异构图
    G = dgl.heterograph(edges)

    # 添加节点属性
    for ntype in G.ntypes:
        if ntype == 'keep':
            nodes = keep_node
        elif ntype == 'del':
            nodes = del_node
        elif ntype == 'add':
            nodes = add_node
        G.nodes[ntype].data['inp'] = torch.stack(nodes, dim=0)

    # 添加边ID
    G.node_dict, G.edge_dict = {}, {}
    for ntype in G.ntypes:
        G.node_dict[ntype] = len(G.node_dict)
    for etype in G.canonical_etypes:
        G.edge_dict[etype] = len(G.edge_dict)
        G.edges[etype].data['id'] = torch.ones(G.number_of_edges(etype), dtype=torch.long) * G.edge_dict[etype]

    print("Graph loaded successfully.")
    return G


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 graph
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.graph = graph


def read_examples(filename):
    """Read examples from filename."""
    examples = []
    count = 0
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            print(count)
            count += 1
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            commit_id = ' '.join(js['commit_id']).replace('\n', ' ')
            contents = ' '.join(js['contents']).replace('\n', ' ')
            nl = ' '.join(js['docstring_tokens']).replace('\n', '')
            nl = ' '.join(nl.strip().split())
            graph = GetDataset(commit_id)
            examples.append(
                Example(
                    idx=idx,
                    source=contents,
                    target=nl,
                    graph=graph
                )
            )
    return examples


class InputFeatures(object):
    """A single training/test feature for an example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,
                 graph):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask
        self.graph = graph  # 添加 train_graph 属性


def convert_examples_to_features(examples, tokenizer, args, stage=None):
    features = []

    for example_index, example in enumerate(examples):
        # source
        source_tokens = tokenizer.tokenize(example.source)[:args.max_source_length - 2]
        source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        source_mask = [1] * (len(source_tokens))
        padding_length = args.max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        source_mask += [0] * padding_length
        # graph
        graph = example.graph
        # target
        if stage == "test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length - 2]
        target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] * len(target_ids)
        padding_length = args.max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length
        target_mask += [0] * padding_length

        if example_index < 5:
            if stage == 'train':
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example.idx))

                logger.info("source_tokens: {}".format([x.replace('\u0120', '_') for x in source_tokens]))
                logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))
                logger.info("source_mask: {}".format(' '.join(map(str, source_mask))))

                logger.info("target_tokens: {}".format([x.replace('\u0120', '_') for x in target_tokens]))
                logger.info("target_ids: {}".format(' '.join(map(str, target_ids))))
                logger.info("target_mask: {}".format(' '.join(map(str, target_mask))))

        features.append(
            InputFeatures(
                example_index,
                source_ids,
                target_ids,
                source_mask,
                target_mask,
                graph  # 添加 train_graph 数据
            )
        )
    return features


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


train_dir = root + '/output/'


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_type", default='roberta', type=str,
                        help="Model type: e.g. roberta")
    # parser.add_argument("--model_name_or_path", default='../codebert', type=str,
    #                    help="Path to pre-trained model: e.g. roberta-base")
    parser.add_argument("--output_dir",
                        default=root + '/output/codebert',
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files")
    ## Other parameters
    parser.add_argument("--train_filename",
                        default='data/code_msg/code_msg_train_VT' + '.jsonl',
                        type=str,
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_filename", default=None, type=str,
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename",
                        default='data/code_msg/code_msg_test_VT' + '.jsonl',
                        type=str,
                        help="The test filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_source_length", default=256, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=128, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true', default=True,
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', default=False,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true', default=True,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true', default=True,
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-6, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=4, type=int,
                        help="beam size for beam search")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=2, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    # print arguments
    args = parser.parse_args()
    logger.info(args)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')

        args.n_gpu = 1
    logger.warning("进程 rank: %s, 设备: %s, GPU 数量: %s, 分布式训练: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
    args.device = device
    # Set seed
    set_seed(args.seed)
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    bart_tokenizer = BartTokenizer.from_pretrained("../bart-base")
    bart_config = BartConfig.from_pretrained("../bart-base")
    bart_pretrain = BartForConditionalGeneration.from_pretrained("../bart-base",
                                                                 config=bart_config)
    bart_pretrain.resize_token_embeddings(len(bart_tokenizer))
    n_out = 2  # 输出类别数
    num_node_types = 3  # 节点类型数
    num_edge_types = 18  # 边类型数
    embedding_dim = 768  # 输入维度
    n_hid = 768  # 隐藏层维度
    n_layers = 1  # 图的层数
    n_heads = 2  # 注意力头的数量
    model = Seq2Seq(n_out, num_node_types, num_edge_types, embedding_dim, n_hid, n_layers, n_heads,
                   bart_model=bart_pretrain, config=bart_config, beam_size=args.beam_size,
                   max_length=args.max_target_length,
                   sos_id=bart_tokenizer.cls_token_id, eos_id=bart_tokenizer.sep_token_id)

    if args.load_model_path is not None:
        logger.info("重新加载模型来自 {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))

    model.to(device)
    if args.local_rank != -1:
        # Distributed training
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)

    if args.do_train:
        metric = load_metric("rouge.py")

        def postprocess_text(preds, labels):
            preds = [pred.strip() for pred in preds]
            labels = [label.strip() for label in labels]

            # rougeLSum expects newline after each sentence
            preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
            labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

            return preds, labels

        def compute_metrics(preds, labels):
            # Some simple post-processing
            decoded_preds, decoded_labels = postprocess_text(preds, labels)

            result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
            # Extract a few results from ROUGE
            result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

            prediction_lens = [np.count_nonzero(pred != bart_tokenizer.pad_token_id) for pred in preds]
            result["gen_len"] = np.mean(prediction_lens)
            result = {k: round(v, 4) for k, v in result.items()}
            return result

        # Prepare training data loader
        train_examples = read_examples(args.train_filename)
        train_features = convert_examples_to_features(train_examples, bart_tokenizer, args, stage='train')
        all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in train_features], dtype=torch.long)
        all_train_graphs = [f.graph for f in train_features]
        all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
        all_target_mask = torch.tensor([f.target_mask for f in train_features], dtype=torch.long)

        train_data = TrainDataset(all_source_ids, all_source_mask, all_train_graphs,all_target_ids, all_target_mask)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)

        train_dataloader = DataLoader(
            train_data,
            sampler=train_sampler,
            batch_size=args.train_batch_size // args.gradient_accumulation_steps,
            collate_fn=Train_collate_fn
        )

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(t_total * 0.1),
                                                    num_training_steps=t_total)

        # Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num epoch = %d", args.num_train_epochs)

        model.train()
        dev_dataset = {}
        nb_tr_examples, nb_tr_steps, tr_loss, global_step, best_bleu, best_loss = 0, 0, 0, 0, 0, 1e6
        best_rouge = 0
        for epoch in range(args.num_train_epochs):
            bar = tqdm(train_dataloader, total=len(train_dataloader))
            for batch in bar:
                source_ids, source_mask, train_graph, target_ids, target_mask = batch
                loss, _, _ = model(source_ids=source_ids, source_mask=source_mask, graphs=train_graph,
                                   target_ids=target_ids, target_mask=target_mask)

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()
                train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
                bar.set_description("epoch {} loss {}".format(epoch, train_loss))
                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()

                if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1

        #     if args.do_eval:
        #         # Eval model with dev dataset
        #         tr_loss = 0
        #         nb_tr_examples, nb_tr_steps = 0, 0
        #         eval_flag = False
        #         if 'dev_loss' in dev_dataset:
        #             eval_examples, eval_data = dev_dataset['dev_loss']
        #         else:
        #             eval_examples = read_examples(args.dev_filename)
        #             eval_features = convert_examples_to_features(eval_examples, bart_tokenizer, args, stage='dev')
        #             all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
        #             all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
        #             all_train_graphs = [f.graph for f in train_features]
        #             all_target_ids = torch.tensor([f.target_ids for f in eval_features], dtype=torch.long)
        #             all_target_mask = torch.tensor([f.target_mask for f in eval_features], dtype=torch.long)
        #             eval_data = TensorDataset(all_source_ids, all_source_mask, all_target_ids, all_target_mask)
        #             dev_dataset['dev_loss'] = eval_examples, eval_data
        #         eval_sampler = SequentialSampler(eval_data)
        #         eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        #
        #         logger.info("\n***** Running evaluation *****")
        #         logger.info("  Num examples = %d", len(eval_examples))
        #         logger.info("  Batch size = %d", args.eval_batch_size)
        #
        #         # Start Evaling model
        #         model.eval()
        #         eval_loss, tokens_num = 0, 0
        #         for batch in eval_dataloader:
        #             batch = tuple(t.to(device) for t in batch)
        #             source_ids, source_mask, target_ids, target_mask = batch
        #
        #             with torch.no_grad():
        #                 _, loss, num = model(source_ids=source_ids, source_mask=source_mask, train_graphs=train_graphs,
        #                                      target_ids=target_ids, target_mask=target_mask)
        #             eval_loss += loss.sum().item()
        #             tokens_num += num.sum().item()
        #         # Pring loss of dev dataset
        #         model.train()
        #         eval_loss = eval_loss / tokens_num
        #         result = {'eval_ppl': round(np.exp(eval_loss), 5),
        #                   'global_step': global_step + 1,
        #                   'train_loss': round(train_loss, 5)}
        #         for key in sorted(result.keys()):
        #             logger.info("  %s = %s", key, str(result[key]))
        #         logger.info("  " + "*" * 20)
        #
        #         # save last checkpoint
        #         last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
        #         if not os.path.exists(last_output_dir):
        #             os.makedirs(last_output_dir)
        #         model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        #         output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
        #         torch.save(model_to_save.state_dict(), output_model_file)
        #         if eval_loss < best_loss:
        #             logger.info("  Best ppl:%s", round(np.exp(eval_loss), 5))
        #             logger.info("  " + "*" * 20)
        #             best_loss = eval_loss
        #             # Save best checkpoint for best ppl
        #             output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
        #             if not os.path.exists(output_dir):
        #                 os.makedirs(output_dir)
        #             model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        #             output_model_file = os.path.join(output_dir, "pytorch_model.bin")
        #             torch.save(model_to_save.state_dict(), output_model_file)
        #
        #             # Calculate bleu
        #         if 'dev_bleu' in dev_dataset:
        #             eval_examples, eval_data = dev_dataset['dev_bleu']
        #         else:
        #             eval_examples = read_examples(args.dev_filename)
        #             eval_examples = random.sample(eval_examples, min(1000, len(eval_examples)))
        #             eval_features = convert_examples_to_features(eval_examples, bart_tokenizer, args, stage='test')
        #             all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
        #             all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
        #             eval_data = TensorDataset(all_source_ids, all_source_mask)
        #             dev_dataset['dev_bleu'] = eval_examples, eval_data
        #
        #         eval_sampler = SequentialSampler(eval_data)
        #         eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        #
        #         model.eval()
        #         p = []
        #         for batch in eval_dataloader:
        #             batch = tuple(t.to(device) for t in batch)
        #             source_ids, source_mask = batch
        #             with torch.no_grad():
        #                 preds = model(source_ids=source_ids, source_mask=source_mask)
        #                 for pred in preds:
        #                     t = pred[0].cpu().numpy()
        #                     t = list(t)
        #                     if 0 in t:
        #                         t = t[:t.index(0)]
        #                     text = bart_tokenizer.decode(t, clean_up_tokenization_spaces=False)
        #                     p.append(text)
        #         model.train()
        #         predictions = []
        #         with open(os.path.join(args.output_dir, "dev.output"), 'w') as f, open(
        #                 os.path.join(args.output_dir, "dev.gold"), 'w') as f1:
        #             for ref, gold in zip(p, eval_examples):
        #                 predictions.append(str(gold.idx) + '\t' + ref)
        #                 f.write(str(gold.idx) + '\t' + ref + '\n')
        #                 f1.write(str(gold.idx) + '\t' + gold.target + '\n')
        #
        #         (goldMap, predictionMap) = bleu.computeMaps(predictions, os.path.join(args.output_dir, "dev.gold"))
        #         dev_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
        #         logger.info("  %s = %s " % ("bleu-4", str(dev_bleu)))
        #         logger.info("  " + "*" * 20)
        #         if dev_bleu > best_bleu:
        #             logger.info("  Best bleu:%s", dev_bleu)
        #             logger.info("  " + "*" * 20)
        #             best_bleu = dev_bleu
        #             # Save best checkpoint for best bleu
        #             output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
        #             if not os.path.exists(output_dir):
        #                 os.makedirs(output_dir)
        #             model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        #             output_model_file = os.path.join(output_dir, "pytorch_model.bin")
        #             torch.save(model_to_save.state_dict(), output_model_file)
        #
        # output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)
        # model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        # output_model_file = os.path.join(output_dir, "pytorch_model.bin")
        # torch.save(model_to_save.state_dict(), output_model_file)

    if args.do_test:
        logger.info("Test file: {}".format(args.test_filename))
        test_examples = read_examples(args.test_filename)

        test_features = convert_examples_to_features(test_examples, bart_tokenizer, args, stage='test')
        all_source_ids = torch.tensor([f.source_ids for f in test_features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in test_features], dtype=torch.long)
        all_test_graphs = [f.graph for f in test_features]
        test_data = TestDataset(all_source_ids, all_source_mask, all_test_graphs)

        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(
                test_data,
                sampler=test_sampler,
                batch_size=args.test_batch_size // args.gradient_accumulation_steps,
                collate_fn=Test_collate_fn
        )
        model.eval()
        p = []
        for batch in tqdm(test_dataloader, total=len(test_dataloader)):
            source_ids, source_mask,test_graph = batch
            with torch.no_grad():
                preds = model(source_ids=source_ids, source_mask=source_mask, graphs=test_graph)
                for pred in preds:
                    t = pred[0].cpu().numpy()
                    t = list(t)
                    if 0 in t:
                        t = t[:t.index(0)]
                    text = bart_tokenizer.decode(t, clean_up_tokenization_spaces=False)
                    p.append(text)
        model.train()
        predictions = []
        rouge_pred, rouge_gold = [], []
        with open(os.path.join(args.output_dir, "test.output"), 'w',encoding='utf-8') as f,\
              open(os.path.join(args.output_dir, "test.gold"), 'w',encoding='utf-8') as f1, \
              open(os.path.join(args.output_dir, "result.output"), 'w',encoding='utf-8') as f2:
            for ref, gold in zip(p, test_examples):
                predictions.append(str(gold.idx) + '\t' + ref)
                f.write(str(gold.idx) + '\t' + ref + '\n')
                f1.write(str(gold.idx) + '\t' + gold.target + '\n')
                rouge_pred.append(ref)
                rouge_gold.append(gold.target)
            matrix = compute_metrics(rouge_pred, rouge_gold)
            f2.write(str(matrix))

if __name__ == "__main__":
    main()

