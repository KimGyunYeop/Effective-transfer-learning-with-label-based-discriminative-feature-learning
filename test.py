import argparse
import logging
import os

from attrdict import AttrDict
import numpy as np
from torch.utils.data import DataLoader, SequentialSampler
from fastprogress.fastprogress import progress_bar
from datasets import DATASET_LIST, BaseDataset
import pandas as pd

from model import *
import json

from src import (
    CONFIG_CLASSES,
    TOKENIZER_CLASSES,
    init_logger,
    compute_metrics,
    set_seed
)
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    AutoTokenizer,
    AutoConfig
)

logger = logging.getLogger(__name__)


def evaluate(args, model, eval_dataset, mode, global_step=None):
    results = {}
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    if global_step != None:
        logger.info("***** Running Test on {} dataset ({} step) *****".format(mode, global_step))
    else:
        logger.info("***** Running Test on {} dataset *****".format(mode))
    logger.info("  Num examples = {}".format(len(eval_dataset)))
    logger.info("  Eval Batch size = {}".format(args.eval_batch_size))
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    polarity_ids = None
    intensity_ids = None
    out_label_ids = None
    txt_all = []
    ep_loss = []

    for (batch, txt) in progress_bar(eval_dataloader):
        model.eval()
        txt_all = txt_all + list(txt)
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                if len(batch) == 4:
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "token_type_ids": batch[2],
                        "labels": batch[3]
                    }
                else:
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "token_type_ids": None,
                        "labels": batch[2]
                    }

            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            if type(tmp_eval_loss) == tuple:
                # print(list(map(lambda x:x.item(),tmp_eval_loss)))
                ep_loss.append(list(map(lambda x: x.item(), tmp_eval_loss)))
                tmp_eval_loss = sum(tmp_eval_loss)
            else:
                ep_loss.append([tmp_eval_loss.item()])

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)

    result = compute_metrics(out_label_ids, preds)
    results.update(result)

    output_dir = os.path.join(args.output_dir, mode)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_eval_file = os.path.join(output_dir,
                                    "{}-{}.txt".format(mode, global_step) if global_step else "{}.txt".format(mode))
    with open(output_eval_file, "w") as f_w:
        logger.info("***** Eval results on {} dataset *****".format(mode))
        for key in sorted(results.keys()):
            logger.info("  {} = {}".format(key, str(results[key])))
            f_w.write("  {} = {}\n".format(key, str(results[key])))

    return preds, out_label_ids, results, txt_all


def main(cli_args):
    # Read from config file and make args
    max_checkpoint = "checkpoint-best"

    args = torch.load(os.path.join("ckpt", cli_args.result_dir, max_checkpoint, "training_args.bin"))
    with open(os.path.join(cli_args.config_dir, cli_args.config_file)) as f:
        args = AttrDict(json.load(f))
    logger.info("Training/evaluation parameters {}".format(args))
    logger.info("cliargs parameters {}".format(cli_args))

    args.output_dir = os.path.join(args.ckpt_dir, cli_args.result_dir)
    args.model_mode = cli_args.model_mode
    args.device = "cuda:{}".format(cli_args.gpu) if torch.cuda.is_available() and not args.no_cuda else "cpu"

    init_logger()
    set_seed(args)

    model_link = None
    if cli_args.transformer_mode.upper() == "T5":
        model_link = "t5-base"
    elif cli_args.transformer_mode.upper() == "ELECTRA":
        model_link = "google/electra-base-discriminator"
    elif cli_args.transformer_mode.upper() == "ALBERT":
        model_link = "albert-base-v2"

    tokenizer = AutoTokenizer.from_pretrained(model_link)

    args.test_file = os.path.join(cli_args.dataset, args.test_file)
    args.dev_file = os.path.join(cli_args.dataset, args.train_file)
    args.train_file = os.path.join(cli_args.dataset, args.train_file)
    # Load dataset
    train_dataset = DATASET_LIST[cli_args.model_mode](args, tokenizer, mode="train") if args.train_file else None
    dev_dataset = BaseDataset(args, tokenizer, mode="dev") if args.dev_file else None
    test_dataset = BaseDataset(args, tokenizer, mode="test") if args.test_file else None

    if dev_dataset == None:
        args.evaluate_test_during_training = True  # If there is no dev dataset, only use testset

    args.logging_steps = int(len(train_dataset) / args.train_batch_size) + 1
    args.save_steps = args.logging_steps
    labelNumber = train_dataset.getLabelNumber()

    labels = [str(i) for i in range(labelNumber)]
    config = AutoConfig.from_pretrained(model_link)

    args.device = "cuda:{}".format(cli_args.gpu) if torch.cuda.is_available() and not args.no_cuda else "cpu"
    config.device = args.device
    args.model_mode = cli_args.model_mode

    logger.info("Testing model checkpoint to {}".format(max_checkpoint))
    global_step = max_checkpoint.split("-")[-1]

    # GPU or CPU
    args.device = "cuda:{}".format(cli_args.gpu) if torch.cuda.is_available() and not args.no_cuda else "cpu"
    config.device = args.device
    args.model_mode = cli_args.model_mode

    model = MODEL_LIST[cli_args.model_mode](model_link, args.model_type, args.model_name_or_path, config, labelNumber)
    model.load_state_dict(torch.load(os.path.join("ckpt", cli_args.result_dir, max_checkpoint, "training_model.bin")))

    model.to(args.device)

    preds, labels, result, txt_all= evaluate(args, model, test_dataset, mode="test",
                                                                               global_step=global_step)
    pred_and_labels = pd.DataFrame([])
    pred_and_labels["data"] = txt_all
    pred_and_labels["pred"] = preds
    pred_and_labels["label"] = labels
    pred_and_labels["result"] = preds == labels
    decode_result = list(
        pred_and_labels["data"].apply(lambda x: tokenizer.convert_ids_to_tokens(tokenizer(x)["input_ids"])))
    pred_and_labels["tokenizer"] = decode_result

    pred_and_labels.to_csv(os.path.join("ckpt", cli_args.result_dir, "test_result_" + max_checkpoint + ".csv"),
                             encoding="cp949")


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()

    cli_parser.add_argument("--config_dir", type=str, default="config")
    cli_parser.add_argument("--config_file", type=str, default="koelectra-base.json")
    cli_parser.add_argument("--dataset", type=str, required=True)
    cli_parser.add_argument("--result_dir", type=str, required=True)
    cli_parser.add_argument("--model_mode", type=str, required=True, choices=MODEL_LIST.keys())
    cli_parser.add_argument("--transformer_mode", type=str, required=True)
    cli_parser.add_argument("--gpu", type=str, default = 0)

    cli_args = cli_parser.parse_args()

    main(cli_args)
