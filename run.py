import random
import os
import argparse
import numpy as np
import json
import torch
import utils
import collections
from tqdm import tqdm

from preprocess import json2features
from torch.utils.data import TensorDataset, DataLoader

from models.pytorch_modeling import BertConfig, BertForQuestionAnswering
from models.tokenization import BertTokenizer

from optimization import get_optimization, warmup_linear

from evaluate.cmrc2018_output import write_predictions
from evaluate.cmrc2018_evaluate import get_eval



def evaluate(model, 
            args, 
            eval_examples, 
            eval_features, 
            device, 
            global_steps, 
            best_f1, 
            best_em, 
            best_f1_em):
    print("***** Eval *****")
    RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])
    output_prediction_file = 'data/output/predict.json'
    output_nbest_file = output_prediction_file.replace('predict', 'predict_nbest')

    all_input_ids = torch.tensor([f['input_ids'] for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f['input_mask'] for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f['segment_ids'] for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)  # dev_nums

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
    eval_dataloader = DataLoader(eval_data, batch_size=args.n_batch, shuffle=False)

    model.eval()
    all_results = []
    print("Start evaluating")
    for input_ids, input_mask, segment_ids, example_indices in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
            batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask)

        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature['unique_id'])
            all_results.append(RawResult(unique_id=unique_id,
                                         start_logits=start_logits,
                                         end_logits=end_logits))

    write_predictions(eval_examples,
                      eval_features,
                      all_results,
                      n_best_size=args.n_best,
                      max_answer_length=args.max_ans_length,
                      do_lower_case=True,
                      output_prediction_file=output_prediction_file,
                      output_nbest_file=output_nbest_file)

    tmp_result = get_eval(args.dev_file, output_prediction_file)
    tmp_result['STEP'] = global_steps
    with open(args.log_file, 'a') as aw:
        aw.write(json.dumps(tmp_result) + '\n')
    print(tmp_result)

    if float(tmp_result['F1']) > best_f1:
        best_f1 = float(tmp_result['F1'])

    if float(tmp_result['EM']) > best_em:
        best_em = float(tmp_result['EM'])

    if float(tmp_result['F1']) + float(tmp_result['EM']) > best_f1_em:
        best_f1_em = float(tmp_result['F1']) + float(tmp_result['EM'])
        utils.torch_save_model(model, args.checkpoint_dir,
                               {'f1': float(tmp_result['F1']), 'em': float(tmp_result['EM'])}, max_save_num=1)

    model.train()
    return best_f1, best_em, best_f1_em


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0, 1')
    # training parameter
    parser.add_argument('--train_epochs', type=int, default=2)
    parser.add_argument('--n_batch', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--clip_norm', type=float, default=1.0)
    parser.add_argument('--warmup_rate', type=float, default=0.1)
    parser.add_argument("--schedule", default='warmup_linear', type=str, help='schedule')
    parser.add_argument("--weight_decay_rate", default=0.01, type=float, help='weight_decay_rate')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--float16', type=bool, default=False) 
    parser.add_argument('--max_ans_length', type=int, default=50)
    parser.add_argument('--n_best', type=int, default=20)
    parser.add_argument('--eval_epochs', type=float, default=0.5)
    parser.add_argument('--save_best', type=bool, default=True)
    parser.add_argument('--vocab_size', type=int, default=17964)  # 21182

    # data path
    parser.add_argument('--train_file', type=str, default='data/cmrc2018_train.json')
    parser.add_argument('--dev_file', type=str, default='data/cmrc2018_dev.json')
    parser.add_argument('--test_file', type=str, default='data/cmrc2018_trial.json')

    parser.add_argument('--train_example_file', type=str, default='data/output/output_example_train.json')
    parser.add_argument('--dev_example_file', type=str, default='data/output/output_example_dev.json')
    parser.add_argument('--test_example_file', type=str, default='data/output/output_example_test.json')

    parser.add_argument('--train_feature_file', type=str, default='data/output/output_feature_train.json')
    parser.add_argument('--dev_feature_file', type=str, default='data/output/output_feature_dev.json')
    parser.add_argument('--test_feature_file', type=str, default='data/output/output_feature_test.json')

    # pretrain model config file
    parser.add_argument('--bert_config', type=str, default='pretrain_model/bert_config.json')
    parser.add_argument('--vocab', type=str, default='pretrain_model/vocab.txt')
    parser.add_argument('--bert_path', type=str, default='pretrain_model/pytorch_model.bin')

    # save info
    parser.add_argument('--checkpoint_dir', type=str, default='check_points/models/')
    parser.add_argument('--setting_file', type=str, default='setting.txt')
    parser.add_argument('--log_file', type=str, default='log.txt')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()
    bert_config = BertConfig.from_json_file(args.bert_config)
    tokenizer = BertTokenizer(vocab_file=args.vocab, do_lower_case=True)


    if not os.path.exists(args.train_example_file) or not os.path.exists(args.train_feature_file):
        json2features(args.train_file,
                        args.train_example_file,
                        args.train_feature_file,
                        tokenizer,
                        is_training=True,
                        max_seq_length=bert_config.max_position_embeddings)
    if not os.path.exists(args.dev_example_file) or not os.path.exists(args.dev_feature_file):
        json2features(args.dev_file,
                      args.dev_example_file,
                      args.dev_feature_file,
                      tokenizer,
                      is_training=False,
                      max_seq_length=bert_config.max_position_embeddings)
    if not os.path.exists(args.test_example_file) or not os.path.exists(args.test_feature_file):
        json2features(args.test_file,
                      args.test_example_file,
                      args.test_feature_file,
                      tokenizer,
                      is_training=False,
                      max_seq_length=bert_config.max_position_embeddings)


    train_features = json.load(open(args.train_feature_file, 'r'))
    dev_examples = json.load(open(args.dev_example_file, 'r'))
    dev_features = json.load(open(args.dev_feature_file, 'r'))

    steps_per_epoch = len(train_features) // args.n_batch
    eval_steps = int(steps_per_epoch * args.eval_epochs)
    dev_steps_per_epoch = len(dev_features) // args.n_batch

    if len(train_features) % args.n_batch != 0:
        steps_per_epoch += 1
    if len(dev_features) % args.n_batch != 0:
        dev_steps_per_epoch += 1
    total_steps = steps_per_epoch * args.train_epochs


    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    F1s = []
    EMs = []
    best_f1_em = 0
    best_f1, best_em = 0, 0

    model = BertForQuestionAnswering(bert_config)
    utils.torch_init_model(model, args.bert_path)  # 加载模型参数

    if args.float16:
        model.half()
    model.to(device)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    optimizer = get_optimization(model=model,
                                float16=args.float16,
                                learning_rate=args.lr,
                                total_steps=total_steps,
                                schedule=args.schedule,
                                warmup_rate=args.warmup_rate,
                                max_grad_norm=args.clip_norm,
                                weight_decay_rate=args.weight_decay_rate)

    all_input_ids = torch.tensor([f['input_ids'] for f in train_features], dtype=torch.long)  # train_num, 512
    all_input_mask = torch.tensor([f['input_mask'] for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f['segment_ids'] for f in train_features], dtype=torch.long)
    all_start_positions = torch.tensor([f['start_position'] for f in train_features], dtype=torch.long)
    all_end_positions = torch.tensor([f['end_position'] for f in train_features], dtype=torch.long)

    train_data = TensorDataset(all_input_ids,
                               all_input_mask,
                               all_segment_ids,
                               all_start_positions,
                               all_end_positions)
    train_dataloader = DataLoader(train_data, batch_size=args.n_batch, shuffle=True)

    print('***** Training *****')
    model.train()
    global_steps = 1
    best_em = 0
    best_f1 = 0
    for i in range(int(args.train_epochs)):
        print('Starting epoch %d' % (i + 1))
        total_loss = 0
        iteration = 1
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, start_positions, end_positions = batch
            loss = model(input_ids, segment_ids, input_mask, start_positions, end_positions)
            if n_gpu > 1:
                loss = loss.mean()
            total_loss += loss.item()

            if args.float16:
                optimizer.backward(loss)
                lr_this_step = args.lr * warmup_linear(global_steps / total_steps, args.warmup_rate)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
            else:
                loss.backward()

            optimizer.step()
            model.zero_grad()
            global_steps += 1
            iteration += 1

            if global_steps % eval_steps == 0:
                best_f1, best_em, best_f1_em = evaluate(model, args, dev_examples, dev_features, device,
                                                        global_steps, best_f1, best_em, best_f1_em)
    F1s.append(best_f1)
    EMs.append(best_em)

print('Mean F1:', np.mean(F1s), 'Mean EM:', np.mean(EMs))
print('Best F1:', np.max(F1s), 'Best EM:', np.max(EMs))
