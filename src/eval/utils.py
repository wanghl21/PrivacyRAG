# Utility functions and classes for model evaluation, including stopping criteria and metrics

from transformers import StoppingCriteria
import transformers
from typing import List
import regex
import json
import string
import sacrebleu
import unicodedata
from typing import List
import numpy as np
from collections import Counter
import torch


class MultiTokenEOSCriteria(transformers.StoppingCriteria):
    """
    Custom stopping criteria for text generation that detects multi-token stop sequences.
    
    This class checks if the generated text contains specific sequences (like newlines or punctuation)
    that should trigger the end of generation. It handles the complexity of tokenization differences
    between the input stop sequence string and how it might be tokenized in the model output.
    """

    def __init__(
        self,
        sequence: str,
        tokenizer: transformers.PreTrainedTokenizer,
        initial_decoder_input_length: int,
        batch_size: int,
    ) -> None:
        """
        Initialize the stopping criteria with a sequence to detect.
        
        Args:
            sequence: The string sequence that should trigger stopping when generated
            tokenizer: The tokenizer used by the model for encoding/decoding
            initial_decoder_input_length: Length of the initial input to exclude from checking
            batch_size: Number of sequences being generated in parallel
        """
        self.initial_decoder_input_length = initial_decoder_input_length
        self.done_tracker = [False] * batch_size  # Track which sequences in the batch are done
        self.sequence = sequence
        self.sequence_ids = tokenizer.encode(sequence, add_special_tokens=False)
        
        # Add a buffer of 2 extra tokens in the lookback window
        # This handles tokenization discrepancies - e.g., when a model generates
        # tokens that represent the same string but with different tokenization
        # For example: ['\n', '\n'] vs ['\n\n']
        self.sequence_id_len = len(self.sequence_ids) + 2
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        """
        Check if the generated sequences contain the stop sequence.
        
        This method is called during generation to determine if we should stop.
        It examines only the newly generated tokens, not the original input.
        
        Args:
            input_ids: Tensor containing the generated token IDs (including prompt)
            scores: Tensor containing token prediction scores
            
        Returns:
            bool: True if all sequences in the batch have generated the stop sequence
        """
        # Extract only the generated part, excluding the initial input
        lookback_ids_batch = input_ids[:, self.initial_decoder_input_length :]

        # Further trim to the last n tokens, where n is the length of the stop sequence ID
        lookback_ids_batch = lookback_ids_batch[:, -self.sequence_id_len :]

        # Decode the token IDs to strings for comparison
        lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)

        # Check each sequence in the batch
        for i, done in enumerate(self.done_tracker):
            if not done:
                # Mark as done if the stop sequence is found in the generated text
                self.done_tracker[i] = self.sequence in lookback_tokens_batch[i]
        # If all sequences are done, return True to stop generation
        return False not in self.done_tracker

## copied from https://github.com/EleutherAI/lm-evaluation-harness/blob/cb22e5028a6e40f409a539cbdd87194fd5e2570c/lm_eval/models/utils.py#L248
def stop_sequences_criteria(
    tokenizer: transformers.PreTrainedTokenizer,
    initial_decoder_input_length: int,
    batch_size: int,
    stop_sequences: List[str] = ['\n', '.', ','],
    ) -> transformers.StoppingCriteriaList:
    """
    Create a list of stopping criteria based on multiple sequences.
    
    This function is a helper to create stopping criteria for common punctuation and newline characters.
    
    Args:
        tokenizer: The tokenizer used by the model
        initial_decoder_input_length: Length of the initial input to exclude from checking
        batch_size: Number of sequences being generated in parallel
        stop_sequences: List of string sequences that should trigger stopping when generated
    
    Returns:
        StoppingCriteriaList: A list of stopping criteria objects for use in generation
    """
    return transformers.StoppingCriteriaList(
        [
            *[
                MultiTokenEOSCriteria(
                    sequence, tokenizer, initial_decoder_input_length, batch_size
                )
                for sequence in stop_sequences
            ],
        ]
    )

class SimpleTokenizer(object):
    """
    A simple tokenizer class for splitting text into tokens.
    
    This tokenizer uses regular expressions to identify sequences of alphanumeric characters
    and other non-whitespace characters. It can optionally convert tokens to lowercase.
    """

    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        """
        Initialize the tokenizer, compiling the regular expression used for tokenization.
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        """
        Tokenize the input text into individual tokens.
        
        Args:
            text: The input text to tokenize
            uncased: If True, convert tokens to lowercase
            
        Returns:
            List[str]: A list of tokens extracted from the text
        """
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens


def check_answer(example, tokenizer) -> List[bool]:
    """Search through all the top docs to see if they have any of the answers."""
    answers = example['answers']
    ctxs = example['ctxs']

    hits = []

    for _, doc in enumerate(ctxs):
        text = doc['text']

        if text is None:  # cannot find the document for some reason
            hits.append(False)
            continue

        hits.append(has_answer(answers, text, tokenizer))

    return hits


def has_answer(answers, text, tokenizer=SimpleTokenizer()) -> bool:
    """Check if a document contains an answer string."""
    text = _normalize(text)
    text = tokenizer.tokenize(text, uncased=True)

    for answer in answers:
        answer = _normalize(answer)
        answer = tokenizer.tokenize(answer, uncased=True)
        for i in range(0, len(text) - len(answer) + 1):
            if answer == text[i: i + len(answer)]:
                return True
    return False

def has_answer_count(answers, text, tokenizer=SimpleTokenizer()) -> bool:
    """Check if a document contains an answer string."""
    text = _normalize(text)
    text = tokenizer.tokenize(text, uncased=True)
    tokenized_answers = []
    for answer in answers:
        answer = _normalize(answer)
        tokenized_answers.extend(tokenizer.tokenize(answer, uncased=True))
    count = 0
    for text_entry in text:
        for answer in tokenized_answers:
            if answer == text_entry:
                count += 1
    return count

def _normalize(text):
    return unicodedata.normalize('NFD', text)


def normalize_answer(s):
    """
    Normalize the answer string by removing articles, punctuation, and extra whitespace.
    
    This function prepares the answer string for comparison by removing common English
    articles (a, an, the), punctuation, and normalizing whitespace. It also converts
    the string to lowercase.
    
    Args:
        s: The input string to normalize
    
    Returns:
        str: The normalized string
    """
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    """
    Calculate the exact match score between a prediction and the ground truth.
    
    The exact match score is 1.0 if the normalized prediction and ground truth are identical,
    and 0.0 otherwise.
    
    Args:
        prediction: The predicted answer string
        ground_truth: The ground truth answer string
    
    Returns:
        float: The exact match score (0.0 or 1.0)
    """
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def ems(prediction, ground_truths):
    """
    Calculate the best exact match score for a prediction against multiple ground truths.
    
    This function returns the maximum exact match score obtained when comparing the
    prediction to each ground truth answer.
    
    Args:
        prediction: The predicted answer string
        ground_truths: A list of ground truth answer strings
    
    Returns:
        float: The highest exact match score achieved with any ground truth
    """
    return max([exact_match_score(prediction, gt) for gt in ground_truths])


def f1_score(prediction, ground_truth):
    """
    Calculate the F1 score between a prediction and the ground truth.
    
    The F1 score is the harmonic mean of precision and recall, both calculated
    based on the overlap of tokens between the prediction and ground truth.
    
    Args:
        prediction: The predicted answer string
        ground_truth: The ground truth answer string
    
    Returns:
        float: The F1 score (ranging from 0.0 to 1.0)
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def f1(prediction, ground_truths):
    """
    Calculate the best F1 score for a prediction against multiple ground truths.
    
    This function returns the maximum F1 score obtained when comparing the
    prediction to each ground truth answer.
    
    Args:
        prediction: The predicted answer string
        ground_truths: A list of ground truth answer strings
    
    Returns:
        float: The highest F1 score achieved with any ground truth
    """
    return max([f1_score(prediction, gt) for gt in ground_truths])


def rougel_score(prediction, ground_truth):
    """
    Calculate the ROUGE-L score between a prediction and the ground truth.
    
    The ROUGE-L score measures the longest common subsequence between the prediction
    and ground truth, providing a measure of similarity. This implementation uses the
    `rouge` Python package to calculate the score.
    
    Args:
        prediction: The predicted answer string
        ground_truth: The ground truth answer string
    
    Returns:
        float: The ROUGE-L score (ranging from 0.0 to 1.0)
    """
    from rouge import Rouge
    rouge = Rouge()
    # no normalization
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except ValueError:  # "Hypothesis is empty."
        return 0.0
    return scores["rouge-l"]["f"]


def rl(prediction, ground_truths):
    """
    Calculate the best ROUGE-L score for a prediction against multiple ground truths.
    
    This function returns the maximum ROUGE-L score obtained when comparing the
    prediction to each ground truth answer.
    
    Args:
        prediction: The predicted answer string
        ground_truths: A list of ground truth answer strings
    
    Returns:
        float: The highest ROUGE-L score achieved with any ground truth
    """
    return max([rougel_score(prediction, gt) for gt in ground_truths])


## file-level evaluation ... ### 
def eval_recall(infile):
    """
    Evaluate the recall of model outputs compared to ground truth answers in a file.
    
    This function reads a file of examples, checks if the model output contains the ground truth answer,
    and calculates the recall (percentage of examples where the answer was found). It also reports
    the average length of the answers.
    
    Args:
        infile: The input file containing examples with ground truth answers and model outputs
    
    Returns:
        tuple: A tuple containing the recall and average answer length
    """

    tokenizer = SimpleTokenizer()
    lines = open(infile, 'r').readlines()[1:]

    has_answer_count = 0
    answer_lengths = []
    for line in lines:
        line = json.loads(line)
        answer = line['answer']
        output = ' || '.join(line['output'])

        if has_answer(answer, output, tokenizer):
            has_answer_count += 1

        answer_lengths.append(len(output.split()))

    recall = round(has_answer_count/len(lines), 4)
    lens = round(np.mean(answer_lengths), 4)

    return recall, lens



def eval_fact_checking(outputs,answers):
    """
    Evaluate the accuracy of model outputs for fact-checking tasks.
    
    This function compares model outputs to expected answers (True/False) for fact-checking.
    It returns the accuracy (percentage of correct predictions) and a detailed result list.
    
    Args:
        outputs: The model's predicted answers
        answers: The ground truth answers (True/False)
    
    Returns:
        tuple: A tuple containing the accuracy and a list of individual results
    """

    tokenizer = SimpleTokenizer()

    results = []
    acc_count = 0
    answer_lengths = []
    for output,answer in zip(outputs,answers):

        if answer == "False"  or answer == "no":
            answer = ["refutes", "no", "false"]
        if answer == "True" or answer == "yes":
            answer = ["supports", "yes", "true"]
        assert answer == ["refutes", "no", "false"] or answer == ["supports", "yes", "true"]

        if has_answer(answer, output, tokenizer):
            acc_count += 1
            results.append(1.0)
        else:
            results.append(0.0)
        
        answer_lengths.append(len(output.split()))

    acc = round(sum(results)/len(results),4)
    return acc,results

def eval_fact_checking_vote(outputs,answers):
    """
    Evaluate the accuracy of model outputs for fact-checking tasks with voting.
    
    This function compares model outputs to expected answers (True/False) for fact-checking.
    It uses a voting mechanism where multiple supporting/refuting tokens increase the confidence
    in the prediction. The function returns the accuracy and a detailed result list.
    
    Args:
        outputs: The model's predicted answers
        answers: The ground truth answers (True/False)
    
    Returns:
        tuple: A tuple containing the accuracy and a list of individual results
    """

    tokenizer = SimpleTokenizer()

    results = []
    acc_count = 0
    answer_lengths = []
    for output,answer in zip(outputs,answers):

        if answer == "False" or answer == "no":
            answer = ["refutes", "no", "false"]
            inverse_answer = ["supports", "yes", "true"]
        if answer == "True" or answer == "yes":
            answer = ["supports", "yes", "true"]
            inverse_answer = ["refutes", "no", "false"]
        assert answer == ["refutes", "no", "false"] or answer == ["supports", "yes", "true"]

        if has_answer_count(answer, output, tokenizer) > has_answer_count(inverse_answer, output, tokenizer):
            acc_count += 1
            results.append(1.0)
        else:
            # if has_answer_count(answer, output, tokenizer) > 0:
            #     print("*"*20)
            #     print("Answer: ", answer)
            #     print(output,"\n\n\n")
            results.append(0.0)
        
        answer_lengths.append(len(output.split()))

    acc = round(sum(results)/len(results),4)
    return acc,results

def calculate_sacrebleu(outputs,answers):
    """
    calculate the BLEU score using sacrebleu
    """
    bleu_score = []
    for output,answer in zip(outputs,answers):
        score = sacrebleu.sentence_bleu(output, answer)
        bleu_score.append(score.score)
    return round(np.mean(bleu_score), 4), bleu_score


def eval_truthfulqa(outputs,answers):
    """
    Evaluate model outputs on the TruthfulQA benchmark.
    
    This function calculates the F1 and ROUGE-L scores for model outputs compared to ground truth answers
    in the TruthfulQA dataset. It returns the average F1 and ROUGE-L scores, as well as detailed score lists.
    
    Args:
        outputs: The model's predicted answers
        answers: The ground truth answers
    
    Returns:
        tuple: A tuple containing the average F1 score, average ROUGE-L score, F1 score list, and ROUGE-L score list
    """

    f1_scores = []
    rl_scores = []
    for output,answer in zip(outputs,answers):

        f1_scores.append(f1(output, answer))
        rl_scores.append(rl(output, answer))

    F1 = round(np.mean(f1_scores), 4)
    RL = round(np.mean(rl_scores), 4)

    return F1, RL, f1_scores,rl_scores

def get_exact_match_score(outputs,answers):
    """
    Calculate exact match scores for model outputs compared to ground truth answers.
    
    This function checks if the model output exactly matches any of the ground truth answers
    (after normalization). It returns the exact match rate and a list of individual scores.
    
    Args:
        outputs: The model's predicted answers
        answers: The ground truth answers
    
    Returns:
        tuple: A tuple containing the exact match rate and a list of individual exact match scores
    """
    import numpy as np
    assert len(outputs) == len(answers)
    if not isinstance(answers[0],list):
        answers = [[x] for x in answers]
    exact_match_scores = []
    answer_lengths = []
    for output,answer in zip(outputs,answers):
        if ems(output, answer): # EM evaluation
            exact_match_scores.append(1.0)
        else:
            exact_match_scores.append(0.0)
        
        answer_lengths.append(len(output.split()))

    em = round(sum(exact_match_scores)/len(outputs), 4)
    lens = round(np.mean(answer_lengths), 4)

    return em,exact_match_scores


def get_substring_match_score(outputs,answers):
    """
    Calculate substring match scores for model outputs compared to ground truth answers.
    
    This function checks if any of the ground truth answers are substrings of the model output.
    It returns the substring match rate and a list of individual scores.
    
    Args:
        outputs: The model's predicted answers
        answers: The ground truth answers (can be single strings or lists of strings)
    
    Returns:
        tuple: A tuple containing the substring match rate and a list of individual substring match scores
    """
    import numpy as np
    assert len(outputs) == len(answers)
    if not isinstance(answers[0],list):
        answers = [[x] for x in answers]
    substring_match_scores = []
    answer_lengths = []
    for output,answer in zip(outputs,answers):
        if has_answer(answer,output): # EM evaluation
            substring_match_scores.append(1.0)
        else:
            substring_match_scores.append(0.0)
        
        answer_lengths.append(len(output.split()))

    substring_match = round(sum(substring_match_scores)/len(outputs), 4)
    lens = round(np.mean(answer_lengths), 4)

    return substring_match,substring_match_scores


def eval_multiple_choice(generated_answers,answers):
    """
    Evaluate model performance on multiple-choice questions.
    
    This function calculates the accuracy of model-generated answers to multiple-choice questions
    by comparing them to the correct answers. It returns the accuracy and a list of individual results.
    
    Args:
        generated_answers: The model's predicted answers
        answers: The correct answers
    
    Returns:
        tuple: A tuple containing the accuracy and a list of individual accuracy scores
    """
    ret = []
    assert len(generated_answers) == len(answers)
    for g_answer,answer in zip(generated_answers,answers):
        ret.append(float(g_answer==answer))
    return round(sum(ret)/len(ret),3),ret


def get_unigram_f1(text, answers):
    """Calculate unigram f1 score between the text and reference answers."""
    def _get_unigram_f1(text,answers):
        if isinstance(answers,str):
            answers = [answers]
        norm_pred = normalize_answer(text)
        norm_answers = [normalize_answer(ans) for ans in answers]
        common_tokens = [
            Counter(norm_pred) & Counter(norm_ans) for norm_ans in norm_answers
        ]
        num_same = [sum(common.values()) for common in common_tokens]

        score_list = []
        for i, num in enumerate(num_same):
            if num == 0:
                score_list.append(0.0)
            else:
                p = 1.0 * num / len(norm_pred)
                r = 1.0 * num / len(norm_answers[i])
                f1 = 2 * p * r / (p + r)
                score_list.append(f1)
        return max(score_list)
    unigram_f1 = [_get_unigram_f1(t,a) for t,a in zip(text,answers)]
    
    return sum(unigram_f1)/len(unigram_f1),unigram_f1
