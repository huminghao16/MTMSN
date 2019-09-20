# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run BERT on DROP."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import json
import random

import numpy as np
import torch

import bert.tokenization as tokenization
from bert.modeling import BertConfig
from bert.modeling_drop import MTMSN
from bert.optimization import BERTAdam
from drop.drop_utils import DropReader, convert_examples_to_features, get_tensors, get_tensors_list, write_predictions, \
    ClusteredBatcher, FixedOrderBatcher, FeatureLenKey, batch_annotate_candidates
from drop.drop_metric import DropEmAndF1

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the parameters optimized on CPU/RAM back to the model on GPU
    """
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        param_model.data.copy_(param_opti.data)


def set_optimizer_params_grad(named_params_optimizer, named_params_model, test_nan=False):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
    """
    is_nan = False
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        if test_nan and torch.isnan(param_model.grad).sum() > 0:
            is_nan = True
        if param_opti.grad is None:
            param_opti.grad = torch.nn.Parameter(param_opti.data.new().resize_(*param_opti.data.size()))
        param_opti.grad.data.copy_(param_model.grad.data)
    return is_nan


def bert_load_state_dict(model, state_dict):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix='' if hasattr(model, 'bert') else 'bert.')

    if len(missing_keys) > 0:
        logger.info("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        logger.info("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    return model


# ["passage_span", "question_span", "addition_subtraction", "counting"]
def read_train_data(args, tokenizer, logger):
    skip_when_all_empty = []
    if args.span_extraction:
        skip_when_all_empty.append("passage_span")
        skip_when_all_empty.append("question_span")
    if args.addition_subtraction:
        skip_when_all_empty.append("addition_subtraction")
    if args.counting:
        skip_when_all_empty.append("counting")
    if args.negation:
        skip_when_all_empty.append("negation")

    drop_reader = DropReader(debug=args.do_debug,
                             include_more_numbers=args.include_more_numbers,
                             skip_when_all_empty=skip_when_all_empty,
                             max_number_of_answer=args.max_answer_number,
                             logger=logger)
    train_examples = drop_reader._read(args.train_file)

    train_features = convert_examples_to_features(
        examples=train_examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        is_train=True,
        answering_abilities=args.answering_abilities,
        logger=logger)

    num_train_steps = int(len(train_features) / args.train_batch_size * args.num_train_epochs)
    logger.info("Num steps = %d", num_train_steps)
    return train_examples, train_features, num_train_steps


def read_eval_data(args, tokenizer, logger):
    drop_reader = DropReader(debug=args.do_debug,
                             include_more_numbers=args.include_more_numbers,
                             skip_when_all_empty=[],
                             logger=logger)
    eval_examples = drop_reader._read(args.predict_file)

    eval_features = convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        is_train=False,
        logger=logger)

    return eval_examples, eval_features


def run_train_epoch(args, global_step, n_gpu, device, model, param_optimizer, optimizer,
                    train_examples, train_features, eval_examples, eval_features,
                    logger, log_path, save_path, best_f1, epoch):
    batching = ClusteredBatcher(args.train_batch_size, FeatureLenKey(), truncate_batches=True)
    running_loss, count = 0.0, 0
    model.train()
    for step, features in enumerate(batching.get_epoch(train_features)):
        batch_features, batch_tensors = get_tensors_list(features, True, args.gradient_accumulation_steps, args.max_seq_length)
        for batch_feature, batch_tensor in zip(batch_features, batch_tensors):
            if n_gpu == 1:
                batch_tensor = tuple(t.to(device) for t in batch_tensor)  # multi-gpu does scattering it-self
            input_ids, input_mask, segment_ids, number_indices, start_indices, end_indices, number_of_answers, \
            input_counts, add_sub_expressions, negations = batch_tensor

            with torch.no_grad():
                output_dict = model("normal", input_ids, segment_ids, input_mask, number_indices)
            if len(args.answering_abilities) >= 1:
                best_answer_ability = output_dict["best_answer_ability"]
            number_sign_logits = output_dict["number_sign_logits"]
            number_mask = output_dict["number_mask"]

            batch_result = []
            for i, feature in enumerate(batch_feature):
                unique_id = int(feature.unique_id)
                result = {}
                result['unique_id'] = unique_id
                if len(args.answering_abilities) >= 1:
                    result['predicted_ability'] = best_answer_ability[i].detach().cpu().numpy()
                result['number_sign_logits'] = number_sign_logits[i].detach().cpu().numpy()
                result['number_mask'] = number_mask[i].detach().cpu().numpy()
                batch_result.append(result)

            number_indices2, sign_indices, sign_labels, _ = \
                batch_annotate_candidates(train_examples, batch_feature, batch_result, args.answering_abilities, True,
                                          args.beam_size, args.max_count)

            number_indices2 = torch.tensor(number_indices2, dtype=torch.long)
            sign_indices = torch.tensor(sign_indices, dtype=torch.long)
            sign_labels = torch.tensor(sign_labels, dtype=torch.long)
            number_indices2 = number_indices2.to(device)
            sign_indices = sign_indices.to(device)
            sign_labels = sign_labels.to(device)

            loss = model("normal", input_ids, segment_ids, input_mask, number_indices,
                         start_indices, end_indices, number_of_answers, input_counts, add_sub_expressions, negations,
                         number_indices2, sign_indices, sign_labels)
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.fp16 and args.loss_scale != 1.0:
                # rescale loss for fp16 training
                # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                loss = loss * args.loss_scale
            if args.gradient_accumulation_steps > 1 and len(batch_tensors) > 1:
                loss = loss / len(batch_tensors)
            loss.backward()
            running_loss += loss.item()

        if args.fp16 or args.optimize_on_cpu:
            if args.fp16 and args.loss_scale != 1.0:
                # scale down gradients for fp16 training
                for param in model.parameters():
                    param.grad.data = param.grad.data / args.loss_scale
            is_nan = set_optimizer_params_grad(param_optimizer, model.named_parameters(), test_nan=True)
            if is_nan:
                logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                args.loss_scale = args.loss_scale / 2
                model.zero_grad()
                continue
            optimizer.step()
            copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
        else:
            optimizer.step()
        model.zero_grad()
        global_step += 1
        count += 1

        if global_step % 1000 == 0 and count != 0:
            logger.info("step: {}, loss: {:.3f}".format(global_step, running_loss / count))
            running_loss, count = 0.0, 0

    logger.info("***** Running evaluation *****")
    model.eval()
    metrics = evaluate(args, model, device, eval_examples, eval_features, logger)
    f = open(log_path, "a")
    print("step: {}, em: {:.3f}, f1: {:.3f}"
          .format(global_step, metrics['em'], metrics['f1']), file=f)
    print(" ", file=f)
    f.close()
    if metrics['f1'] > best_f1:
        logger.info("Save model at {} (step {}, epoch {})".format(save_path, global_step, epoch))
        best_f1 = metrics['f1']
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': global_step,
            'epoch': epoch
        }, save_path)
    return global_step, model, best_f1


def evaluate(args, model, device, eval_examples, eval_features, logger, write_pred=False):
    batching = FixedOrderBatcher(args.predict_batch_size, truncate_batches=True)
    drop_metrics = DropEmAndF1()
    all_results = []
    for step, batch_feature in enumerate(batching.get_epoch(eval_features)):
        if len(all_results) % 1000 == 0 and write_pred:
            logger.info("Processing example: %d" % (len(all_results)))
        batch = get_tensors(batch_feature, is_train=False)
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, number_indices = batch
        with torch.no_grad():
            output_dict = model("normal", input_ids, segment_ids, input_mask, number_indices)

        if len(args.answering_abilities) >= 1:
            best_answer_ability = output_dict["best_answer_ability"]
        span_start_logits = output_dict["span_start_logits"]
        span_end_logits = output_dict["span_end_logits"]
        best_span_number = output_dict["best_span_number"]
        number_sign_logits = output_dict["number_sign_logits"]
        number_mask = output_dict["number_mask"]
        encoded_numbers_output = output_dict["encoded_numbers_output"]
        passage_output = output_dict["passage_output"]
        question_output = output_dict["question_output"]
        pooled_output = output_dict["pooled_output"]
        if "counting" in args.answering_abilities:
            best_count_number = output_dict["best_count_number"]
        if "negation" in args.answering_abilities:
            best_negations_for_numbers = output_dict["best_negations_for_numbers"]

        batch_result = []
        for i, feature in enumerate(batch_feature):
            unique_id = int(feature.unique_id)
            result = {}
            result['unique_id'] = unique_id
            if len(args.answering_abilities) >= 1:
                result['predicted_ability'] = best_answer_ability[i].detach().cpu().numpy()
            result['start_logits'] = span_start_logits[i].detach().cpu().tolist()
            result['end_logits'] = span_end_logits[i].detach().cpu().tolist()
            result['predicted_spans'] = best_span_number[i].detach().cpu().numpy()
            result['number_sign_logits'] = number_sign_logits[i].detach().cpu().numpy()
            result['number_mask'] = number_mask[i].detach().cpu().numpy()
            if "counting" in args.answering_abilities:
                result['predicted_count'] = best_count_number[i].detach().cpu().numpy()
            if "negation" in args.answering_abilities:
                result['predicted_negations'] = best_negations_for_numbers[i].detach().cpu().numpy()
            batch_result.append(result)

        number_indices2, sign_indices, _, sign_scores = \
            batch_annotate_candidates(eval_examples, batch_feature, batch_result, args.answering_abilities,
                                      False, args.beam_size, args.max_count)
        number_indices2 = torch.tensor(number_indices2, dtype=torch.long)
        sign_indices = torch.tensor(sign_indices, dtype=torch.long)
        number_indices2 = number_indices2.to(device)
        sign_indices = sign_indices.to(device)

        with torch.no_grad():
            sign_rerank_logits = model("rerank_inference", input_ids, segment_ids, input_mask, number_indices,
                                       number_indices2=number_indices2, sign_indices=sign_indices,
                                       encoded_numbers_input=encoded_numbers_output, passage_input=passage_output,
                                       question_input=question_output, pooled_input=pooled_output)

        for i, result in enumerate(batch_result):
            result['number_indices2'] = number_indices2[i].detach().cpu().tolist()
            result['sign_indices'] = sign_indices[i].detach().cpu().tolist()
            result['sign_rerank_logits'] = sign_rerank_logits[i].detach().cpu().tolist()
            result['sign_probs'] = sign_scores[i]
            all_results.append(result)

    all_predictions, metrics = write_predictions(eval_examples, eval_features, all_results,
                                                 args.answering_abilities, drop_metrics, args.length_heuristic,
                                                 args.n_best_size, args.max_answer_length,
                                                 args.do_lower_case, args.verbose_logging, logger)

    if write_pred:
        output_prediction_file = os.path.join(args.output_dir, "predictions.json")
        with open(output_prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")
        logger.info("Writing predictions to: %s" % (output_prediction_file))

    return metrics


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_config_file", default=None, type=str, required=True,
                        help="The config json file corresponding to the pre-trained BERT model. "
                             "This specifies the model architecture.")
    parser.add_argument("--vocab_file", default=None, type=str, required=True,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Answering ablities
    parser.add_argument("--span_extraction", default=False, action='store_true', help="Whether to use span extraction.")
    parser.add_argument("--addition_subtraction", default=False, action='store_true', help="Whether to use addition subtraction.")
    parser.add_argument("--counting", default=False, action='store_true', help="Whether to use counting.")
    parser.add_argument("--negation", default=False, action='store_true', help="Whether to use negation.")
    parser.add_argument("--include_more_numbers", default=True, action='store_true', help="Whether to include more numbers.")
    parser.add_argument("--beam_size", default=3, type=int, help="The size of beam search.")
    parser.add_argument("--max_count", default=4, type=int, help="The maximal number of add_sub expressions.")
    parser.add_argument("--max_answer_number", default=8, type=int, help="The maximal number of answers.")

    ## Other parameters
    parser.add_argument("--do_debug", default=False, action='store_true', help="Whether to run in debug mode.")
    parser.add_argument("--train_file", default=None, type=str, help="DROP json for training. E.g., drop_dataset_train.json")
    parser.add_argument("--predict_file", default=None, type=str, help="DROP json for predictions.")
    parser.add_argument("--init_checkpoint", default=None, type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).")
    parser.add_argument("--do_lower_case", default=False, action='store_true',
                        help="Whether to lower case the input text. Should be True for uncased "
                             "models and False for cased models.")
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--do_train", default=False, action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", default=False, action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=32, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.05, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")
    parser.add_argument("--length_heuristic", default=0.05, type=float,
                        help="Weight on length heuristic.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", default=False, action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--data_parallel",
                        default=False,
                        action='store_true',
                        help="Whether not to use data parallel")
    parser.add_argument('--seed', 
                        type=int, 
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--optimize_on_cpu',
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=128,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')

    args = parser.parse_args()

    if not args.span_extraction and not args.addition_subtraction and not args.counting and not args.negation:
        raise ValueError("At least one of `span_extraction` or `addition_subtraction` or `counting` or `negation` must be True.")

    args.answering_abilities = []
    if args.span_extraction:
        args.answering_abilities.append("span_extraction")
    if args.addition_subtraction:
        args.answering_abilities.append("addition_subtraction")
    if args.counting:
        args.answering_abilities.append("counting")
    if args.negation:
        args.answering_abilities.append("negation")
    logger.info("Answering abilities: {}".format(args.answering_abilities))

    assert "span_extraction" in args.answering_abilities and "addition_subtraction" in args.answering_abilities

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logger.info('output_dir: {}'.format(args.output_dir))
    save_path = os.path.join(args.output_dir, 'checkpoint.pth.tar')
    log_path = os.path.join(args.output_dir, 'performance.txt')
    network_path = os.path.join(args.output_dir, 'network.txt')
    parameter_path = os.path.join(args.output_dir, 'parameter.txt')

    f = open(parameter_path, "w")
    for arg in sorted(vars(args)):
        print("{}: {}".format(arg, getattr(args, arg)), file=f)
    f.close()

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.do_train and not args.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
    if args.do_predict and not args.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            logger.info("16-bits training currently not supported in distributed training")
            args.fp16 = False # (see https://github.com/pytorch/pytorch/pull/13496)
    logger.info("torch_version: {} device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        torch.__version__, device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    bert_config = BertConfig.from_json_file(args.bert_config_file)

    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (args.max_seq_length, bert_config.max_position_embeddings))

    # --- Prepare model ---
    logger.info("***** Preparing model *****")
    model = MTMSN(bert_config, args.answering_abilities, args.max_answer_number)
    if args.init_checkpoint is not None and not os.path.isfile(save_path):
        logger.info("Loading model from pretrained checkpoint: {}".format(args.init_checkpoint))
        model = bert_load_state_dict(model, torch.load(args.init_checkpoint, map_location='cpu'))

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1 or args.data_parallel:
        model = torch.nn.DataParallel(model)

    if os.path.isfile(save_path):
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model'])
        logger.info("Loading model from finetuned checkpoint: '{}' (step {}, epoch {})"
                    .format(save_path, checkpoint['step'], checkpoint['epoch']))

    f = open(network_path, "w")
    for n, param in model.named_parameters():
        print("name: {}, size: {}, dtype: {}, requires_grad: {}"
              .format(n, param.size(), param.dtype, param.requires_grad), file=f)
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print("Total trainable parameters: {}".format(total_trainable_params), file=f)
    print("Total parameters: {}".format(total_params), file=f)
    f.close()

    # --- Prepare data ---
    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

    train_examples, train_features, num_train_steps = None, None, None
    eval_examples, eval_features = None, None
    if args.do_train:
        logger.info("***** Preparing training *****")
        train_examples, train_features, num_train_steps = read_train_data(args, tokenizer, logger)
        logger.info("***** Preparing evaluation *****")
        eval_examples, eval_features = read_eval_data(args, tokenizer, logger)
    if args.do_predict and eval_features is None:
        logger.info("***** Preparing prediction *****")
        eval_examples, eval_features = read_eval_data(args, tokenizer, logger)

    # --- Prepare optimizer ---
    logger.info("***** Preparing optimizer *****")
    if args.fp16:
        param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_()) \
                            for n, param in model.named_parameters()]
    elif args.optimize_on_cpu:
        param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) \
                            for n, param in model.named_parameters()]
    else:
        param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if n not in no_decay], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if n in no_decay], 'weight_decay_rate': 0.0}
        ]
    optimizer = BERTAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_steps)

    global_step, global_epoch = 0, 1
    if os.path.isfile(save_path):
        checkpoint = torch.load(save_path)
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("Load optimizer from finetuned checkpoint: '{}' (step {}, epoch {})"
                    .format(save_path, checkpoint['step'], checkpoint['epoch']))
        global_step = checkpoint['step']
        global_epoch = checkpoint['epoch'] + 1

    # --- Run training ---
    if args.do_train and global_epoch < int(args.num_train_epochs)+1:
        logger.info("***** Running training *****")
        best_f1 = 0
        for epoch in range(global_epoch, int(args.num_train_epochs)+1):
            logger.info("***** Epoch: {} *****".format(epoch))
            global_step, model, best_f1 = run_train_epoch(args, global_step, n_gpu, device, model, param_optimizer, optimizer,
                                                          train_examples, train_features, eval_examples, eval_features,
                                                          logger, log_path, save_path, best_f1, epoch)

    # --- Run prediction ---
    if args.do_predict:
        logger.info("***** Running prediction *****")
        # restore from best checkpoint
        if save_path and os.path.isfile(save_path):
            checkpoint = torch.load(save_path)
            model.load_state_dict(checkpoint['model'])
            logger.info("Loading model from finetuned checkpoint: '{}' (step {}, epoch {})"
                        .format(save_path, checkpoint['step'], checkpoint['epoch']))
            global_step = checkpoint['step']

            torch.save({
                'model': model.state_dict(),
                'step': checkpoint['step'],
                'epoch': checkpoint['epoch']
            }, save_path)

        model.eval()
        metrics = evaluate(args, model, device, eval_examples, eval_features, logger, write_pred=True)
        f = open(log_path, "a")
        print("step: {}, test_em: {:.3f}, test_f1: {:.3f}"
              .format(global_step, metrics['em'], metrics['f1']), file=f)
        print(" ", file=f)
        f.close()


if __name__ == "__main__":
    main()
