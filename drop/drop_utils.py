import json
import copy
import string
import itertools
import numpy as np
from random import choice
from decimal import Decimal
from typing import Any, Dict, List, Tuple, Callable
import collections
from collections import defaultdict

from allennlp.common.file_utils import cached_path
from allennlp.tools.squad_eval import metric_max_over_ground_truths
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.dataset_readers.reading_comprehension.util import IGNORED_TOKENS, STRIPPED_CHARACTERS

import torch
import bert.tokenization as tokenization
from squad.squad_utils import _get_best_indexes, get_final_text, _compute_softmax
from squad.squad_evaluate import f1_score as calculate_f1
from drop.w2n import word_to_num
from drop.beam_search import beam_search
from drop.drop_eval import (get_metrics as drop_em_and_f1, answer_json_to_strings)


sign_remap = {0: 0, 1: 1, 2: -1}


class DropExample(object):
    def __init__(self,
                 qas_id,
                 question_tokens,
                 passage_tokens,
                 numbers_in_passage=None,
                 number_indices=None,
                 answer_type=None,
                 number_of_answer=None,
                 passage_spans=None,
                 question_spans=None,
                 add_sub_expressions=None,
                 counts=None,
                 negations=None,
                 answer_annotations=None
                 ):
        self.qas_id = qas_id
        self.question_tokens = question_tokens
        self.passage_tokens = passage_tokens
        self.numbers_in_passage = numbers_in_passage
        self.number_indices = number_indices
        self.answer_type = answer_type
        self.number_of_answer = number_of_answer
        self.passage_spans = passage_spans
        self.question_spans = question_spans
        self.add_sub_expressions = add_sub_expressions
        self.counts = counts
        self.negations = negations
        self.answer_annotations = answer_annotations

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
        s += ", \nquestion: %s" % (" ".join(self.question_tokens))
        s += ", \npassage: %s" % (" ".join(self.passage_tokens))
        if self.numbers_in_passage:
            s += ", \nnumbers_in_passage: {}".format(self.numbers_in_passage)
        if self.number_indices:
            s += ", \nnumber_indices: {}".format(self.number_indices)
        if self.answer_type:
            s += ", \nanswer_type: {}".format(self.answer_type)
        if self.number_of_answer:
            s += ", \nnumber_of_answer: {}".format(self.number_of_answer)
        if self.passage_spans:
            s += ", \npassage_spans: {}".format(self.passage_spans)
        if self.question_spans:
            s += ", \nquestion_spans: {}".format(self.question_spans)
        if self.add_sub_expressions:
            s += ", \nadd_sub_expressions: {}".format(self.add_sub_expressions)
        if self.counts:
            s += ", \ncounts: {}".format(self.counts)
        if self.negations:
            s += ", \nnegations: {}".format(self.negations)
        if self.answer_annotations:
            s += ", \nanswer_annotations: {}".format(self.answer_annotations)
        return s


class InputFeatures(object):
    def __init__(self,
                 unique_id,
                 example_index,
                 tokens,
                 que_token_to_orig_map,
                 doc_token_to_orig_map,
                 input_ids,
                 input_mask,
                 segment_ids,
                 number_indices,
                 start_indices=None,
                 end_indices=None,
                 number_of_answers=None,
                 add_sub_expressions=None,
                 input_counts=None,
                 negations=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.tokens = tokens
        self.que_token_to_orig_map = que_token_to_orig_map
        self.doc_token_to_orig_map = doc_token_to_orig_map
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.number_indices = number_indices
        self.start_indices = start_indices
        self.end_indices = end_indices
        self.number_of_answers = number_of_answers
        self.add_sub_expressions = add_sub_expressions
        self.input_counts = input_counts
        self.negations = negations

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "unique_id: %s" % (self.unique_id)
        s += ", \nnumber_indices: {}".format(self.number_indices)
        if self.start_indices:
            s += ", \nstart_indices: {}".format(self.start_indices)
        if self.end_indices:
            s += ", \nend_indices: {}".format(self.end_indices)
        if self.number_of_answers:
            s += ", \nnumber_of_answers: {}".format(self.number_of_answers)
        if self.add_sub_expressions:
            s += ", \nadd_sub_expressions: {}".format(self.add_sub_expressions)
        if self.input_counts:
            s += ", \ninput_counts: {}".format(self.input_counts)
        if self.negations:
            s += ", \nnegations: {}".format(self.negations)
        return s


WORD_NUMBER_MAP = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
                   "five": 5, "six": 6, "seven": 7, "eight": 8,
                   "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
                   "thirteen": 13, "fourteen": 14, "fifteen": 15,
                   "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}


def split_token_by_delimiter(token: Token, delimiter: str) -> List[Token]:
    split_tokens = []
    char_offset = token.idx
    for sub_str in token.text.split(delimiter):
        if sub_str:
            split_tokens.append(Token(text=sub_str, idx=char_offset))
            char_offset += len(sub_str)
        split_tokens.append(Token(text=delimiter, idx=char_offset))
        char_offset += len(delimiter)
    if split_tokens:
        split_tokens.pop(-1)
        char_offset -= len(delimiter)
        return split_tokens
    else:
        return [token]


def split_tokens_by_hyphen(tokens: List[Token]) -> List[Token]:
    hyphens = ["-", "–", "~"]
    new_tokens: List[Token] = []

    for token in tokens:
        if any(hyphen in token.text for hyphen in hyphens):
            unsplit_tokens = [token]
            split_tokens: List[Token] = []
            for hyphen in hyphens:
                for unsplit_token in unsplit_tokens:
                    if hyphen in token.text:
                        split_tokens += split_token_by_delimiter(unsplit_token, hyphen)
                    else:
                        split_tokens.append(unsplit_token)
                unsplit_tokens, split_tokens = split_tokens, []
            new_tokens += unsplit_tokens
        else:
            new_tokens.append(token)

    return new_tokens


def extend_number_magnitude(number, next_token):
    if next_token == "hundred":
        number *= 100
    elif next_token == "thousand":
        number *= 1000
    elif next_token == "million":
        number *= 1000000
    elif next_token == "billion":
        number *= 1000000000
    elif next_token == "thousand":
        number *= 1000000000000
    return number


class DropReader(object):
    def __init__(self,
                 debug: bool = False,
                 tokenizer: Tokenizer = None,
                 include_more_numbers: bool = False,
                 skip_when_all_empty: List[str] = None,
                 max_number_of_answer: int = 8,
                 max_number_count: int = 10,
                 logger = None) -> None:
        super().__init__()
        self.debug = debug
        self._tokenizer = tokenizer or WordTokenizer()
        self.include_more_numbers = include_more_numbers
        self.max_number_of_answer = max_number_of_answer
        self.max_number_count = max_number_count
        self.skip_when_all_empty = skip_when_all_empty if skip_when_all_empty is not None else []
        for item in self.skip_when_all_empty:
            assert item in ["passage_span", "question_span", "addition_subtraction", "counting", "negation"], \
                f"Unsupported skip type: {item}"
        self.logger = logger

    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        self.logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset = json.load(dataset_file)
        examples, skip_count = [], 0
        for passage_id, passage_info in dataset.items():
            passage_text = passage_info["passage"]
            passage_tokens = self._tokenizer.tokenize(passage_text)
            passage_tokens = split_tokens_by_hyphen(passage_tokens)
            for question_answer in passage_info["qa_pairs"]:
                question_id = question_answer["query_id"]
                question_text = question_answer["question"].strip()
                answer_annotations = []
                if "answer" in question_answer:
                    answer_annotations.append(question_answer["answer"])
                if "validated_answers" in question_answer:
                    answer_annotations += question_answer["validated_answers"]

                example = self.text_to_example(question_text, passage_text, question_id, answer_annotations, passage_tokens)
                if example is not None:
                    examples.append(example)
                else:
                    skip_count += 1
            if self.debug and len(examples) > 100:
                break
        self.logger.info(f"Skipped {skip_count} examples, kept {len(examples)} examples.")
        return examples

    def text_to_example(self,  # type: ignore
                         question_text: str,
                         passage_text: str,
                         question_id: str,
                         answer_annotations: List[Dict] = None,
                         passage_tokens: List[Token] = None):
        if not passage_tokens:
            passage_tokens = self._tokenizer.tokenize(passage_text)
            passage_tokens = split_tokens_by_hyphen(passage_tokens)
        question_tokens = self._tokenizer.tokenize(question_text)
        question_tokens = split_tokens_by_hyphen(question_tokens)

        answer_type: str = None
        answer_texts: List[str] = []
        number_of_answer: int = None
        if answer_annotations:
            # Currently we only use the first annotated answer here, but actually this doesn't affect
            # the training, because we only have one annotation for the train set.
            answer_type, answer_texts = self.extract_answer_info_from_annotation(answer_annotations[0])
            number_of_answer = self.max_number_of_answer if len(answer_texts) > self.max_number_of_answer else len(answer_texts)

        # Tokenize the answer text in order to find the matched span based on token
        tokenized_answer_texts = []
        for answer_text in answer_texts:
            answer_tokens = self._tokenizer.tokenize(answer_text)
            answer_tokens = split_tokens_by_hyphen(answer_tokens)
            tokenized_answer_texts.append(answer_tokens)

        numbers_in_passage = [0]
        number_indices = [-1]
        for token_index, token in enumerate(passage_tokens):
            number = self.convert_word_to_number(token.text, self.include_more_numbers)
            if number is not None:
                numbers_in_passage.append(number)
                number_indices.append(token_index)

        valid_passage_spans = \
            self.find_valid_spans(passage_tokens, tokenized_answer_texts) if tokenized_answer_texts else []
        valid_question_spans = \
            self.find_valid_spans(question_tokens, tokenized_answer_texts) if tokenized_answer_texts else []
        number_of_answer = None if valid_passage_spans == [] and valid_question_spans == [] else number_of_answer

        target_numbers = []
        # `answer_texts` is a list of valid answers.
        for answer_text in answer_texts:
            number = self.convert_word_to_number(answer_text, self.include_more_numbers)
            if number is not None:
                target_numbers.append(number)

        valid_signs_for_add_sub_expressions = self.find_valid_add_sub_expressions(numbers_in_passage,
                                                                                  target_numbers,
                                                                                  max_number_of_numbers_to_consider=3)

        # Currently we only support count number 0 ~ 9
        numbers_for_count = list(range(self.max_number_count))
        valid_counts = self.find_valid_counts(numbers_for_count, target_numbers)

        valid_negations = self.find_valid_negations(numbers_in_passage, target_numbers)

        type_to_answer_map = {"passage_span": valid_passage_spans,
                              "question_span": valid_question_spans,
                              "addition_subtraction": valid_signs_for_add_sub_expressions,
                              "counting": valid_counts,
                              "negation": valid_negations}

        if self.skip_when_all_empty \
                and not any(type_to_answer_map[skip_type] for skip_type in self.skip_when_all_empty):
            return None

        return DropExample(
            qas_id=question_id,
            question_tokens=[token.text for token in question_tokens],
            passage_tokens=[token.text for token in passage_tokens],
            numbers_in_passage=numbers_in_passage,
            number_indices=number_indices,
            answer_type=answer_type,
            number_of_answer=number_of_answer,
            passage_spans=valid_passage_spans,
            question_spans=valid_question_spans,
            add_sub_expressions=valid_signs_for_add_sub_expressions,
            counts=valid_counts,
            negations=valid_negations,
            answer_annotations=answer_annotations)

    @staticmethod
    def extract_answer_info_from_annotation(answer_annotation: Dict[str, Any]) -> Tuple[str, List[str]]:
        answer_type = None
        if answer_annotation["spans"]:
            answer_type = "spans"
        elif answer_annotation["number"]:
            answer_type = "number"
        elif any(answer_annotation["date"].values()):
            answer_type = "date"

        answer_content = answer_annotation[answer_type] if answer_type is not None else None

        answer_texts: List[str] = []
        if answer_type is None:  # No answer
            pass
        elif answer_type == "spans":
            # answer_content is a list of string in this case
            answer_texts = answer_content
        elif answer_type == "date":
            # answer_content is a dict with "month", "day", "year" as the keys
            date_tokens = [answer_content[key]
                           for key in ["month", "day", "year"] if key in answer_content and answer_content[key]]
            answer_texts = date_tokens
        elif answer_type == "number":
            # answer_content is a string of number
            answer_texts = [answer_content]
        return answer_type, answer_texts

    @staticmethod
    def convert_word_to_number(word: str, try_to_include_more_numbers=False, normalized_tokens=None, token_index=None):
        """
        Currently we only support limited types of conversion.
        """
        if try_to_include_more_numbers:
            # strip all punctuations from the sides of the word, except for the negative sign
            punctruations = string.punctuation.replace('-', '')
            word = word.strip(punctruations)
            # some words may contain the comma as deliminator
            word = word.replace(",", "")
            # word2num will convert hundred, thousand ... to number, but we skip it.
            if word in ["hundred", "thousand", "million", "billion", "trillion"]:
                return None
            try:
                number = word_to_num(word)
            except ValueError:
                try:
                    number = int(word)
                except ValueError:
                    try:
                        number = float(word)
                    except ValueError:
                        number = None
            if number is not None and normalized_tokens is not None and token_index is not None:
                if token_index < len(normalized_tokens) - 1:
                    next_token = normalized_tokens[token_index + 1]
                    if next_token in ["hundred", "thousand", "million", "billion", "trillion"]:
                        number = extend_number_magnitude(number, next_token)
            return number
        else:
            no_comma_word = word.replace(",", "")
            if no_comma_word in WORD_NUMBER_MAP:
                number = WORD_NUMBER_MAP[no_comma_word]
            else:
                try:
                    number = int(no_comma_word)
                except ValueError:
                    number = None
            return number

    @staticmethod
    def find_valid_spans(passage_tokens: List[Token],
                         answer_texts: List[List[Token]]) -> List[Tuple[int, int]]:
        normalized_tokens = [token.text.lower().strip(STRIPPED_CHARACTERS) for token in passage_tokens]
        word_positions: Dict[str, List[int]] = defaultdict(list)
        for i, token in enumerate(normalized_tokens):
            word_positions[token].append(i)
        spans = []
        for answer_text in answer_texts:
            answer_tokens = [token.text.lower().strip(STRIPPED_CHARACTERS) for token in answer_text]
            num_answer_tokens = len(answer_tokens)
            if answer_tokens[0] not in word_positions:
                continue
            for span_start in word_positions[answer_tokens[0]]:
                span_end = span_start  # span_end is _inclusive_
                answer_index = 1
                while answer_index < num_answer_tokens and span_end + 1 < len(normalized_tokens):
                    token = normalized_tokens[span_end + 1]
                    if answer_tokens[answer_index].strip(STRIPPED_CHARACTERS) == token:
                        answer_index += 1
                        span_end += 1
                    elif token in IGNORED_TOKENS:
                        span_end += 1
                    else:
                        break
                if num_answer_tokens == answer_index:
                    spans.append((span_start, span_end))
        return spans

    @staticmethod
    def find_valid_add_sub_expressions(numbers: List[int],
                                       targets: List[int],
                                       max_number_of_numbers_to_consider: int = 2) -> List[List[int]]:
        valid_signs_for_add_sub_expressions = []
        decimal_targets = [Decimal(x).quantize(Decimal('0.00')) for x in targets]
        # TODO: Try smaller numbers?'
        for number_of_numbers_to_consider in range(2, max_number_of_numbers_to_consider + 1):
            possible_signs = list(itertools.product((-1, 1), repeat=number_of_numbers_to_consider))
            for number_combination in itertools.combinations(enumerate(numbers), number_of_numbers_to_consider):
                indices = [it[0] for it in number_combination]
                values = [it[1] for it in number_combination]
                for signs in possible_signs:
                    eval_value = sum(sign * value for sign, value in zip(signs, values))
                    decimal_eval_value = Decimal(eval_value).quantize(Decimal('0.00'))
                    if decimal_eval_value in decimal_targets and min(indices) != 0:
                        labels_for_numbers = [0] * len(numbers)  # 0 represents ``not included''.
                        for index, sign in zip(indices, signs):
                            labels_for_numbers[index] = 1 if sign == 1 else 2  # 1 for positive, 2 for negative
                        if labels_for_numbers not in valid_signs_for_add_sub_expressions:
                            valid_signs_for_add_sub_expressions.append(labels_for_numbers)
        return valid_signs_for_add_sub_expressions

    @staticmethod
    def find_valid_negations(numbers: List[int], targets: List[int]) -> List[List[int]]:
        valid_negations = []
        decimal_targets = [Decimal(x).quantize(Decimal('0.00')) for x in targets]
        for index, number in enumerate(numbers):
            decimal_negating_number = Decimal(100 - number).quantize(Decimal('0.00'))
            if number > 0 and number < 100 and decimal_negating_number in decimal_targets:
                labels_for_numbers = [0] * len(numbers)
                labels_for_numbers[index] = 1
                valid_negations.append(labels_for_numbers)
        return valid_negations

    @staticmethod
    def find_valid_counts(count_numbers: List[int], targets: List[int]) -> List[int]:
        valid_indices = []
        for index, number in enumerate(count_numbers):
            if number in targets:
                valid_indices.append(index)
        return valid_indices


def convert_answer_spans(spans, orig_to_tok_index, all_len, all_tokens):
    tok_start_positions, tok_end_positions = [], []
    for span in spans:
        start_position, end_position = span[0], span[1]
        tok_start_position = orig_to_tok_index[start_position]
        if end_position + 1 >= len(orig_to_tok_index):
            tok_end_position = all_len - 1
        else:
            tok_end_position = orig_to_tok_index[end_position + 1] - 1
        if tok_start_position < len(all_tokens) and tok_end_position < len(all_tokens):
            tok_start_positions.append(tok_start_position)
            tok_end_positions.append(tok_end_position)
    return tok_start_positions, tok_end_positions


def convert_examples_to_features(examples, tokenizer, max_seq_length, is_train, answering_abilities=None, logger=None):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000
    skip_count, truncate_count = 0, 0

    features = []
    for (example_index, example) in enumerate(examples):
        que_tok_to_orig_index = []
        que_orig_to_tok_index = []
        all_que_tokens = []
        for (i, token) in enumerate(example.question_tokens):
            que_orig_to_tok_index.append(len(all_que_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                que_tok_to_orig_index.append(i)
                all_que_tokens.append(sub_token)

        doc_tok_to_orig_index = []
        doc_orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.passage_tokens):
            doc_orig_to_tok_index.append(len(all_doc_tokens))
            if i in example.number_indices:
                doc_tok_to_orig_index.append(i)
                all_doc_tokens.append(token)
            else:
                sub_tokens = tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    doc_tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        # Truncate the passage according to the max sequence length
        max_tokens_for_doc = max_seq_length - len(all_que_tokens) - 3
        all_doc_len = len(all_doc_tokens)
        if all_doc_len > max_tokens_for_doc:
            all_doc_tokens = all_doc_tokens[:max_tokens_for_doc]
            truncate_count += 1

        query_tok_start_positions, query_tok_end_positions = \
            convert_answer_spans(example.question_spans, que_orig_to_tok_index, len(all_que_tokens), all_que_tokens)

        passage_tok_start_positions, passage_tok_end_positions = \
            convert_answer_spans(example.passage_spans, doc_orig_to_tok_index, all_doc_len, all_doc_tokens)

        tok_number_indices = []
        for index in example.number_indices:
            if index != -1:
                tok_index = doc_orig_to_tok_index[index]
                if tok_index < len(all_doc_tokens):
                    tok_number_indices.append(tok_index)
            else:
                tok_number_indices.append(-1)

        tokens = []
        que_token_to_orig_map = {}
        doc_token_to_orig_map = {}
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for i in range(len(all_que_tokens)):
            que_token_to_orig_map[len(tokens)] = que_tok_to_orig_index[i]
            tokens.append(all_que_tokens[i])
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        for i in range(len(all_doc_tokens)):
            doc_token_to_orig_map[len(tokens)] = doc_tok_to_orig_index[i]
            tokens.append(all_doc_tokens[i])
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        number_indices = []
        doc_offset = len(all_que_tokens) + 2
        que_offset = 1
        for tok_number_index in tok_number_indices:
            if tok_number_index != -1:
                number_index = tok_number_index + doc_offset
                number_indices.append(number_index)
            else:
                number_indices.append(-1)

        start_indices, end_indices, add_sub_expressions, input_counts, negations, number_of_answers = [], [], [], [], [], []
        if is_train:
            # For distant supervision, we annotate the positions of all answer spans
            if passage_tok_start_positions != [] and passage_tok_end_positions !=[]:
                for tok_start_position, tok_end_position in zip(passage_tok_start_positions, passage_tok_end_positions):
                    start_position = tok_start_position + doc_offset
                    end_position = tok_end_position + doc_offset
                    start_indices.append(start_position)
                    end_indices.append(end_position)
            elif query_tok_start_positions != [] and query_tok_end_positions !=[]:
                for tok_start_position, tok_end_position in zip(query_tok_start_positions, query_tok_end_positions):
                    start_position = tok_start_position + que_offset
                    end_position = tok_end_position + que_offset
                    start_indices.append(start_position)
                    end_indices.append(end_position)

            # Weakly-supervised for addition-subtraction
            if example.add_sub_expressions != []:
                for add_sub_expression in example.add_sub_expressions:
                    # Since we have truncated the passage, the expression should also be truncated
                    if sum(add_sub_expression[:len(number_indices)]) >= 2:
                        assert len(add_sub_expression[:len(number_indices)]) == len(number_indices)
                        add_sub_expressions.append(add_sub_expression[:len(number_indices)])

            # Weakly-supervised for counting
            for count in example.counts:
                input_counts.append(count)

            # Weeakly-supervised for negation
            if example.negations != []:
                for negation in example.negations:
                    if sum(negation[:len(number_indices)]) == 1:
                        assert len(negation[:len(number_indices)]) == len(number_indices)
                        negations.append(negation[:len(number_indices)])

            is_impossible = True
            if "span_extraction" in answering_abilities and start_indices != [] and end_indices != []:
                is_impossible = False
                assert example.number_of_answer is not None
                number_of_answers.append(example.number_of_answer - 1)

            if "negation" in answering_abilities and negations != []:
                is_impossible = False

            if "addition_subtraction" in answering_abilities and add_sub_expressions != []:
                is_impossible = False

            if "counting" in answering_abilities and input_counts != []:
                is_impossible = False

            if start_indices == [] and end_indices == [] and number_of_answers == []:
                start_indices.append(-1)
                end_indices.append(-1)
                number_of_answers.append(-1)

            if negations == []:
                negations.append([-1] * len(number_indices))

            if add_sub_expressions == []:
                add_sub_expressions.append([-1] * len(number_indices))

            if input_counts == []:
                input_counts.append(-1)

            if not is_impossible:
                features.append(InputFeatures(
                        unique_id=unique_id,
                        example_index=example_index,
                        tokens=tokens,
                        que_token_to_orig_map=que_token_to_orig_map,
                        doc_token_to_orig_map=doc_token_to_orig_map,
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        number_indices=number_indices,
                        start_indices=start_indices,
                        end_indices=end_indices,
                        number_of_answers=number_of_answers,
                        add_sub_expressions=add_sub_expressions,
                        input_counts=input_counts,
                        negations=negations))
                unique_id += 1
            else:
                skip_count += 1
        else:
            features.append(InputFeatures(
                unique_id=unique_id,
                example_index=example_index,
                tokens=tokens,
                que_token_to_orig_map=que_token_to_orig_map,
                doc_token_to_orig_map=doc_token_to_orig_map,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                number_indices=number_indices))
            unique_id += 1

        if len(features) % 5000 == 0:
            logger.info("Processing features: %d" % (len(features)))

    logger.info(f"Skipped {skip_count} features, truncated {truncate_count} features, kept {len(features)} features.")
    return features


def wrapped_get_final_text(example, feature, start_index, end_index, do_lower_case, verbose_logging, logger):
    if start_index in feature.doc_token_to_orig_map and end_index in feature.doc_token_to_orig_map:
        orig_doc_start = feature.doc_token_to_orig_map[start_index]
        orig_doc_end = feature.doc_token_to_orig_map[end_index]
        orig_tokens = example.passage_tokens[orig_doc_start:(orig_doc_end + 1)]
    elif start_index in feature.que_token_to_orig_map and end_index in feature.que_token_to_orig_map:
        orig_que_start = feature.que_token_to_orig_map[start_index]
        orig_que_end = feature.que_token_to_orig_map[end_index]
        orig_tokens = example.question_tokens[orig_que_start:(orig_que_end + 1)]
    else:
        return None

    tok_tokens = feature.tokens[start_index:(end_index + 1)]
    tok_text = " ".join(tok_tokens)

    # De-tokenize WordPieces that have been split off.
    tok_text = tok_text.replace(" ##", "")
    tok_text = tok_text.replace("##", "")

    # Clean whitespace
    tok_text = tok_text.strip()
    tok_text = " ".join(tok_text.split())
    orig_text = " ".join(orig_tokens)

    final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging, logger)
    return final_text


def add_sub_beam_search(example, feature, result, is_training, beam_size, max_count):
    number_sign_logits = result['number_sign_logits']   # [L, 3]
    number_mask = result['number_mask'] # [L]
    number_indices_list, sign_indices_list, scores_list = beam_search(number_sign_logits, number_mask, beam_size, max_count)

    number_sign_labels = []
    if is_training:
        if number_indices_list != [] and sign_indices_list != []:
            for number_indices, sign_indices in zip(number_indices_list, sign_indices_list):
                pred_answer = sum([example.numbers_in_passage[number_index] * sign_remap[sign_index]
                                   for number_index, sign_index in zip(number_indices, sign_indices)])
                pred_answer = float(Decimal(pred_answer).quantize(Decimal('0.0000')))
                ground_truth_answer_strings = [answer_json_to_strings(annotation)[0] for annotation in
                                               example.answer_annotations]
                exact_match, _ = metric_max_over_ground_truths(
                    drop_em_and_f1, str(pred_answer), ground_truth_answer_strings)
                number_sign_labels.append(exact_match)

    # Pad to fixed length
    for number_indices, sign_indices in zip(number_indices_list, sign_indices_list):
        while len(number_indices) < max_count:
            number_indices.append(-1)
            sign_indices.append(-1)

    while len(number_indices_list) < beam_size:
        number_indices_list.append([-1] * max_count)
        sign_indices_list.append([-1] * max_count)
        scores_list.append(0)
        if is_training:
            number_sign_labels.append(0)

    # Add ground truth expressions if there is no positive label
    if is_training and max(number_sign_labels) == 0:
        gold_number_indices, gold_sign_indices = [], []
        add_sub_expression = choice(feature.add_sub_expressions)
        for number_index, sign_index in enumerate(add_sub_expression):
            if sign_index > 0 and number_mask[number_index]:
                gold_number_indices.append(number_index)
                gold_sign_indices.append(sign_index)
        while len(gold_number_indices) < max_count:
            gold_number_indices.append(-1)
            gold_sign_indices.append(-1)
        number_indices_list[-1] = gold_number_indices
        sign_indices_list[-1] = gold_sign_indices
        number_sign_labels[-1] = 1

    return number_indices_list, sign_indices_list, number_sign_labels, scores_list


def batch_annotate_candidates(all_examples, batch_features, batch_results, answering_abilities,
                              is_training, beam_size, max_count):
    """Annotate top-k candidate answers into features."""
    unique_id_to_result = {}
    for result in batch_results:
        unique_id_to_result[result['unique_id']] = result

    batch_number_indices, batch_sign_indices, batch_sign_labels, batch_scores = [], [], [], []
    for (feature_index, feature) in enumerate(batch_features):
        example = all_examples[feature.example_index]
        result = unique_id_to_result[feature.unique_id]

        number_indices, sign_indices, sign_labels, scores = None, None, None, None
        if is_training:
            if feature.add_sub_expressions != [[-1] * len(feature.number_indices)]:
                number_indices, sign_indices, sign_labels, scores = add_sub_beam_search(example, feature, result,
                                                                                        is_training, beam_size, max_count)
        else:
            predicted_ability = result['predicted_ability']
            predicted_ability_str = answering_abilities[predicted_ability]
            if predicted_ability_str == "addition_subtraction":
                number_indices, sign_indices, sign_labels, scores = add_sub_beam_search(example, feature, result,
                                                                                        is_training, beam_size, max_count)

        if number_indices is None and sign_indices is None and sign_labels is None and scores is None:
            number_indices, sign_indices, sign_labels, scores = [], [], [], []
            while len(number_indices) < beam_size:
                number_indices.append([-1] * max_count)
                sign_indices.append([-1] * max_count)
                sign_labels.append(0)
                scores.append(0)

        batch_number_indices.append(number_indices)
        batch_sign_indices.append(sign_indices)
        batch_sign_labels.append(sign_labels)
        batch_scores.append(scores)
    return batch_number_indices, batch_sign_indices, batch_sign_labels, batch_scores


def write_predictions(all_examples, all_features, all_results, answering_abilities, drop_metrics, length_heuristic,
                      n_best_size, max_answer_length, do_lower_case, verbose_logging, logger):
    """Write final predictions to the json file."""

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result['unique_id']] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["start_index", "end_index", "start_logit", "end_logit", "rerank_logit", "heuristic_logit"])

    all_nbest_json = collections.OrderedDict()
    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]
        assert len(features) == 1

        feature = features[0]
        result = unique_id_to_result[feature.unique_id]
        predicted_ability = result['predicted_ability']
        predicted_ability_str = answering_abilities[predicted_ability]
        nbest_json, predicted_answers = [], []
        if predicted_ability_str == "addition_subtraction":
            max_prob, best_answer = 0, None
            sign_rerank_probs = _compute_softmax(result['sign_rerank_logits'])
            for number_indices, sign_indices, rerank_prob, prob in zip(result['number_indices2'], result['sign_indices'], sign_rerank_probs, result['sign_probs']):
                pred_answer = sum([sign_remap[sign_index] * example.numbers_in_passage[number_index] for sign_index, number_index in zip(sign_indices, number_indices) if sign_index != -1 and number_index != -1])
                pred_answer = str(float(Decimal(pred_answer).quantize(Decimal('0.0000'))))
                if rerank_prob*prob > max_prob:
                    max_prob = rerank_prob*prob
                    best_answer = pred_answer
            assert best_answer is not None
            predicted_answers.append(best_answer)
            output = collections.OrderedDict()
            output["text"] = str(best_answer)
            output["type"] = "addition_subtraction"
            nbest_json.append(output)
        elif predicted_ability_str == "counting":
            predicted_answers.append(str(result['predicted_count']))
            output = collections.OrderedDict()
            output["text"] = str(result['predicted_count'])
            output["type"] = "counting"
            nbest_json.append(output)
        elif predicted_ability_str == "negation":
            index = np.argmax(result['predicted_negations'])
            pred_answer = 100 - example.numbers_in_passage[index]
            pred_answer = float(Decimal(pred_answer).quantize(Decimal('0.0000')))
            predicted_answers.append(str(pred_answer))
            output = collections.OrderedDict()
            output["text"] = str(pred_answer)
            output["type"] = "negation"
            nbest_json.append(output)
        elif predicted_ability_str == "span_extraction":
            number_of_spans = result['predicted_spans']
            prelim_predictions = []
            start_indexes = _get_best_indexes(result['start_logits'], n_best_size)
            end_indexes = _get_best_indexes(result['end_logits'], n_best_size)
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.que_token_to_orig_map and start_index not in feature.doc_token_to_orig_map:
                        continue
                    if end_index not in feature.que_token_to_orig_map and start_index not in feature.doc_token_to_orig_map:
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue

                    start_logit = result['start_logits'][start_index]
                    end_logit = result['end_logits'][end_index]
                    heuristic_logit = start_logit + end_logit \
                                      - length_heuristic * (end_index - start_index + 1)
                    prelim_predictions.append(
                        _PrelimPrediction(
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=start_logit,
                            end_logit=end_logit,
                            rerank_logit=0,
                            heuristic_logit=heuristic_logit))

            prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.heuristic_logit), reverse=True)

            _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
                "NbestPrediction", ["text", "start_logit", "end_logit", "start_index", "end_index", "rerank_logit", "heuristic_logit"])

            seen_predictions = {}
            nbest = []
            for i, pred_i in enumerate(prelim_predictions):
                if len(nbest) >= n_best_size:
                    break

                final_text = wrapped_get_final_text(example, feature, pred_i.start_index, pred_i.end_index,
                                                    do_lower_case, verbose_logging, logger)
                if final_text in seen_predictions or final_text is None:
                    continue

                seen_predictions[final_text] = True
                nbest.append(
                    _NbestPrediction(
                        text=final_text,
                        start_logit=pred_i.start_logit,
                        end_logit=pred_i.end_logit,
                        start_index=pred_i.start_index,
                        end_index=pred_i.end_index,
                        rerank_logit=pred_i.rerank_logit,
                        heuristic_logit=pred_i.heuristic_logit
                    ))

                # filter out redundant candidates
                if (i + 1) < len(prelim_predictions):
                    indexes = []
                    for j, pred_j in enumerate(prelim_predictions[(i + 1):]):
                        filter_text = wrapped_get_final_text(example, feature, pred_j.start_index, pred_j.end_index,
                                                             do_lower_case, verbose_logging, logger)
                        if filter_text is None:
                            indexes.append(i + j + 1)
                        else:
                            if calculate_f1(final_text, filter_text) > 0:
                                indexes.append(i + j + 1)
                    [prelim_predictions.pop(index - k) for k, index in enumerate(indexes)]

            # In very rare edge cases we could have no valid predictions. So we
            # just create a nonce prediction in this case to avoid failure.
            if not nbest:
                nbest.append(
                    _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0, start_index=0.0, end_index=0.0,
                                     rerank_logit=0., heuristic_logit=0.))

            assert len(nbest) >= 1

            for i, entry in enumerate(nbest):
                if i > number_of_spans:
                    break
                predicted_answers.append(entry.text)
                output = collections.OrderedDict()
                output["text"] = entry.text
                output["type"] = "span_extraction"
                nbest_json.append(output)
        else:
            raise ValueError(f"Unsupported answer ability: {predicted_ability_str}")

        assert len(nbest_json) >= 1 and len(predicted_answers) >= 1
        if example.answer_annotations:
            drop_metrics(predicted_answers, example.answer_annotations)
        all_nbest_json[example.qas_id] = nbest_json

    exact_match, f1_score = drop_metrics.get_metric(reset=True)
    return all_nbest_json, {'em': exact_match, 'f1': f1_score}


class ListBatcher(object):
    def get_epoch(self, data: List):
        raise NotImplementedError()

    def get_batch_size(self):
        """ Return the batch size """
        raise NotImplementedError()

    def epoch_size(self, n_elements):
        raise NotImplementedError()


class ExampleLenKey(object):
    def __call__(self, d: DropExample):
        return len(d.passage_tokens) + len(d.question_tokens)


class FeatureLenKey(object):
    def __call__(self, d: InputFeatures):
        return len(d.input_ids)


class ClusteredBatcher(ListBatcher):
    def __init__(self,
                 batch_size: int,
                 clustering: Callable,
                 truncate_batches=False):
        self.batch_size = batch_size
        self.clustering = clustering
        self.truncate_batches = truncate_batches

    def get_batch_size(self):
        return self.batch_size

    def get_epoch(self, data: List):
        data = sorted(data, key=self.clustering)
        n_batches = len(data) // self.batch_size
        intervals = [(i * self.batch_size, (i + 1) * self.batch_size) for i in range(0, n_batches)]
        remainder = len(data) % self.batch_size
        if self.truncate_batches and remainder > 0:
            intervals.append((len(data) - remainder, len(data)))
        np.random.shuffle(intervals)
        for i, j in intervals:
            yield data[i:j]

    def epoch_size(self, n_elements):
        size = n_elements // self.batch_size
        if self.truncate_batches and (n_elements % self.batch_size) > 0:
            size += 1
        return size


class FixedOrderBatcher(ListBatcher):
    def __init__(self, batch_size: int, truncate_batches=False):
        self.batch_size = batch_size
        self.truncate_batches = truncate_batches

    def get_batch_size(self):
        return self.batch_size

    def get_epoch(self, data: List):
        n_batches = len(data) // self.batch_size
        for i in range(n_batches):
            yield data[i*self.batch_size:(i + 1)*self.batch_size]
        if self.truncate_batches and (len(data) % self.batch_size) > 0:
            yield data[self.batch_size * (len(data) // self.batch_size):]

    def epoch_size(self, n_elements):
        size = n_elements // self.batch_size
        if self.truncate_batches and (n_elements % self.batch_size) > 0:
            size += 1
        return size


def get_tensors_list(batch, is_train, gra_acc_steps, max_seq_length):
    input_len = np.array([len(feature.input_ids) for feature in batch], dtype='int32')
    max_input_len = input_len.max()
    mini_batch_size = int(len(batch) / gra_acc_steps)

    batchs_list, tensors_list = [], []
    if max_input_len > max_seq_length / gra_acc_steps and mini_batch_size > 0:
        mini_batching = ClusteredBatcher(mini_batch_size, FeatureLenKey(), truncate_batches=True)
        for mini_batch in mini_batching.get_epoch(batch):
            tensors_list.append(get_tensors(mini_batch, is_train))
            batchs_list.append(mini_batch)
    else:
        tensors_list.append(get_tensors(batch, is_train))
        batchs_list.append(batch)
    return batchs_list, tensors_list


def get_tensors(batch, is_train):
    input_len = np.array([len(feature.input_ids) for feature in batch], dtype='int32')
    max_input_len = input_len.max()

    number_indices_len = np.array([len(feature.number_indices) for feature in batch], dtype='int32')
    max_number_indices_len = number_indices_len.max()

    if is_train:
        start_indices_len = np.array([len(feature.start_indices) for feature in batch], dtype='int32')
        max_start_indices_len = start_indices_len.max()

        input_counts_len = np.array([len(feature.input_counts) for feature in batch], dtype='int32')
        max_input_counts_len = input_counts_len.max()

        number_of_answers_len = np.array([len(feature.number_of_answers) for feature in batch], dtype='int32')
        max_number_of_answers_len = number_of_answers_len.max()

        add_sub_combination_len, negation_combination_len  = [], []
        for feature in batch:
            add_sub_combination_len.append(len(feature.add_sub_expressions))
            negation_combination_len.append(len(feature.negations))
        max_add_sub_combination_len = np.array(add_sub_combination_len).max()
        max_negation_combination_len = np.array(negation_combination_len).max()

    input_ids_list, input_mask_list, segment_ids_list, number_indices_list = [], [], [], []
    if is_train:
        start_indices_list, end_indices_list, number_of_answers_list, input_counts_list, add_sub_expressions_list, \
        negations_list = [], [], [], [], [], []
    for feature in batch:
        input_ids = copy.deepcopy(feature.input_ids)
        input_mask = copy.deepcopy(feature.input_mask)
        segment_ids = copy.deepcopy(feature.segment_ids)
        # Zero-pad up to the max mini-batch sequence length.
        while len(input_ids) < max_input_len:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        input_ids_list.append(input_ids)
        input_mask_list.append(input_mask)
        segment_ids_list.append(segment_ids)

        number_indices = copy.deepcopy(feature.number_indices)
        while len(number_indices) < max_number_indices_len:
            number_indices.append(-1)

        number_indices_list.append(number_indices)

        if is_train:
            start_indices = copy.deepcopy(feature.start_indices)
            end_indices = copy.deepcopy(feature.end_indices)
            number_of_answers = copy.deepcopy(feature.number_of_answers)
            input_counts = copy.deepcopy(feature.input_counts)
            add_sub_expressions = copy.deepcopy(feature.add_sub_expressions)
            negations = copy.deepcopy(feature.negations)

            while len(start_indices) < max_start_indices_len:
                start_indices.append(-1)
                end_indices.append(-1)

            while len(input_counts) < max_input_counts_len:
                input_counts.append(-1)

            while len(number_of_answers) < max_number_of_answers_len:
                number_of_answers.append(-1)

            new_add_sub_expressions = []
            for add_sub_expression in add_sub_expressions:
                while len(add_sub_expression) < max_number_indices_len:
                    add_sub_expression.append(-1)
                new_add_sub_expressions.append(add_sub_expression)

            while len(new_add_sub_expressions) < max_add_sub_combination_len:
                new_add_sub_expressions.append([-1] * max_number_indices_len)

            new_negations = []
            for negation in negations:
                while len(negation) < max_number_indices_len:
                    negation.append(-1)
                new_negations.append(negation)

            while len(new_negations) < max_negation_combination_len:
                new_negations.append([-1] * max_number_indices_len)

            start_indices_list.append(start_indices)
            end_indices_list.append(end_indices)
            number_of_answers_list.append(number_of_answers)
            input_counts_list.append(input_counts)
            add_sub_expressions_list.append(new_add_sub_expressions)
            negations_list.append(new_negations)

    batch_input_ids = torch.tensor(input_ids_list, dtype=torch.long)
    batch_input_mask = torch.tensor(input_mask_list, dtype=torch.long)
    batch_segment_ids = torch.tensor(segment_ids_list, dtype=torch.long)
    batch_number_indices = torch.tensor(number_indices_list, dtype=torch.long)

    if is_train:
        batch_start_indices = torch.tensor(start_indices_list, dtype=torch.long)
        batch_end_indices = torch.tensor(end_indices_list, dtype=torch.long)
        batch_number_of_answers = torch.tensor(number_of_answers_list, dtype=torch.long)
        batch_input_counts = torch.tensor(input_counts_list, dtype=torch.long)
        batch_add_sub_expressions = torch.tensor(add_sub_expressions_list, dtype=torch.long)
        batch_negations = torch.tensor(negations_list, dtype=torch.long)
        return batch_input_ids, batch_input_mask, batch_segment_ids, batch_number_indices, batch_start_indices, \
               batch_end_indices, batch_number_of_answers, batch_input_counts, batch_add_sub_expressions, batch_negations
    else:
        return batch_input_ids, batch_input_mask, batch_segment_ids, batch_number_indices
