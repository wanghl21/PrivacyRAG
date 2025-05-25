import random
import copy

from .utils import ParaphraseInstructions, PLACEHOLDER_TOKEN
from pysbd import Segmenter


def split_background_sentence(background):
    """
    split a long document into multiple smaller chunks between single_max_len and single_mini_len

    Args:
        background: string

    Return:
        background: a list of string
    """

    segmenter = Segmenter()
    background = segmenter.segment(background)
    # print(background)
    return background


def split_background(background, tokenizer, total_max_len, single_max_len, single_min_len=1):
    """
    split a long document into multiple smaller chunks between single_max_len and single_mini_len

    Args:
        background: string

    Return:
        background: a list of string
    """
    ids = tokenizer(background, add_special_tokens=False,
                    max_length=total_max_len, truncation=True).input_ids
    background = [ids[idx:idx+single_max_len]
                  for idx in range(0, len(ids), single_max_len)]
    assert len(background) >= 1, background
    if len(background[-1]) <= single_min_len and len(background) > 1:
        background = background[:-1]
    background = [tokenizer.decode(x) for x in background]
    return background


def _encode_chat_format(
    messages,
    tokenizer,
    max_seq_length,
    chat_format='mistral',  # tulu
):
    """
    encode messages to input_ids and make non-assistant part

    Args:
        messages (list): list of dict with 'role' and 'content' field
        tokenizer: llm tokenizer
        max_seq_lengh: maximun context length  

    Return:
        input_ids and labels
    """
    # _concat_messages = eval(f"_concat_messages_{chat_format}")

    # example_text = _concat_messages(messages,tokenizer).strip()
    example_text = tokenizer.apply_chat_template(
        messages, tokenize=False).strip()
    tokenized_example = tokenizer(
        example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    # assert tokenizer.eos_token_id in input_ids, (tokenizer("this is good."+tokenizer.eos_token +'\n').input_ids,input_ids)

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    tokenizer.apply_chat_template(messages[:message_idx], tokenize=False), return_tensors='pt', max_length=max_seq_length, truncation=True
                ).input_ids.shape[1]

            # if chat_format in ['mistral','mixtral']:
            #     messages_so_far = tokenizer.apply_chat_template(messages[:message_idx+1],tokenizer)
            # elif chat_format == 'llama':
            #     messages_so_far = _concat_messages(messages[:message_idx+1],tokenizer)
            # else:
            #     raise ValueError(f"Invalid chat_format: {chat_format}")
            messages_so_far = tokenizer.apply_chat_template(
                messages[:message_idx+1], tokenize=False)

            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors='pt',
                max_length=max_seq_length,
                truncation=True
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100

            if message_end_idx >= max_seq_length:
                break

    # assert tokenizer.eos_token_id in input_ids, input_ids
    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
    }


def encode_with_chat_format_pretrain(
        example,
        tokenizer,
        max_seq_length,
        retrieval_embed_length,
        chat_format='mistral',
        use_split=False,
        use_sentence_split=False,
        retrieval_context_single_max_len=180
):
    """
    encode messages into input_ids and labels for paraphrase pretrain

    Args:
        example: data sample with 'text' filed
        tokenizer: llm_tokenizer
        max_seq_length: maximun context length
        retrieval_embed_length: number of tokens for retrieval (typically 1 for dense retrieval model)

    Return:
        input_ids,labels and retriever_input_text
    """

    document = example['text'].strip()

    if use_sentence_split or use_split:
        if use_sentence_split:
            documents = split_background_sentence(
                document, tokenizer, total_max_len=max_seq_length, single_max_len=retrieval_context_single_max_len)
        elif use_split:
            documents = split_background(
                document, tokenizer, total_max_len=max_seq_length, single_max_len=retrieval_context_single_max_len)

        if len(documents) == 0:
            placeholder_token = " ".join(
                [PLACEHOLDER_TOKEN]*retrieval_embed_length)
            documents = document
        else:
            placeholder_token = " ".join(
                [PLACEHOLDER_TOKEN]*retrieval_embed_length*len(documents))
        instruction = random.choice(ParaphraseInstructions).format_map(
            dict(placeholder_token=placeholder_token))
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": document},
        ]
        encoded = _encode_chat_format(
            messages, tokenizer, max_seq_length, chat_format)
        return {
            "placeholder_input_ids": encoded['input_ids'],
            "placeholder_labels": encoded['labels'],
            "retriever_input_text": documents,
        }
    else:
        placeholder_token = " ".join(
            [PLACEHOLDER_TOKEN]*retrieval_embed_length)
        instruction = random.choice(ParaphraseInstructions).format_map(
            dict(placeholder_token=placeholder_token))

        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": document},
        ]

        encoded = _encode_chat_format(
            messages, tokenizer, max_seq_length, chat_format)

        return {
            "placeholder_input_ids": encoded['input_ids'],
            "placeholder_labels": encoded['labels'],
            "retriever_input_text": [document],
        }


def encode_with_chat_format_finetune(
    example,
    tokenizer,
    max_seq_length,
    retrieval_embed_length,
    use_rag_tuning=True,
    use_retriever_embed=False,
    retriever_tokenizer=None,
    chat_format='mistral',
    use_split=False,
    use_sentence_split=False,
    retrieval_context_single_max_len=180,
):

    messages, background = example['messages'], example['background']

    ret = {}

    if use_rag_tuning and use_retriever_embed:
        if use_split:
            if use_sentence_split:
                sharded_background = split_background_sentence(
                    background, retriever_tokenizer, total_max_len=max_seq_length, single_max_len=180)
            else:
                sharded_background = split_background(
                    background, retriever_tokenizer, total_max_len=max_seq_length, single_max_len=retrieval_context_single_max_len)
                num_split = len(sharded_background)
                ret['retriever_input_text'] = sharded_background
                ret['query'] = messages[0]['content']
        else:
            ret['retriever_input_text'] = [background]
            ret['query'] = messages[0]['content']
            num_split = 1

    if use_rag_tuning:

        _messages = copy.deepcopy(messages)
        placeholder_tokens = " ".join(
            [PLACEHOLDER_TOKEN]*retrieval_embed_length * num_split)

        for idx in range(len(_messages)):
            if _messages[idx]['role'] == 'user':
                _messages[idx]['content'] = f"Refer to the background document: {placeholder_tokens}\n\n" + \
                    messages[idx]['content']
                break
        encoded = _encode_chat_format(
            _messages, tokenizer, max_seq_length, chat_format=chat_format)
        ret['placeholder_input_ids'] = encoded['input_ids']
        ret['placeholder_labels'] = encoded['labels']

        # vanilla RAG
        _messages = copy.deepcopy(messages)
        for idx in range(len(_messages)):
            if _messages[idx]['role'] == 'user':
                _messages[idx]['content'] = f"Refer to the background document: {background}\n\n" + \
                    messages[idx]['content']
                break

        encoded = _encode_chat_format(
            _messages, tokenizer, max_seq_length, chat_format=chat_format)

        ret['input_ids'] = encoded['input_ids']
        ret['labels'] = encoded['labels']

    return ret


QA_PROMPT = "Q: {question}?\nA: {answer}"
RAG_QA_PROMPT = "Background: {background}\n\n"+QA_PROMPT
PARAPHRASE_RAG_QA_PROMPT = "Background: {background}\nThe above background document is just a paraphrase of the following: {real_background}\n\n"+QA_PROMPT

FECT_CHECKING_PROPMT = "Claim: {question}\nAnswer: {answer}"
RAG_FECT_CHECKING_PROPMT = "Background: {background}\n\n" + \
    FECT_CHECKING_PROPMT
PARAPHRASE_RAG_FECT_CHECKING_PROPMT = "Background: {background}\nThe above background document is just a paraphrase of the following: {real_background}\n\n" + FECT_CHECKING_PROPMT

MULTIPLE_CHOICE_PROMPT = "Question: {question}\nAnswer: {answer}"
RAG_MULTIPLE_CHOICE_PROMPT = "Background: {background}\n\n" + \
    MULTIPLE_CHOICE_PROMPT
PARAPHRASE_RAG_MULTIPLE_CHOICE_PROMPT = "Background: {background}\nThe above background document is just a paraphrase of the following: {real_background}\n\n" + MULTIPLE_CHOICE_PROMPT


PROMPT_TEMPLATES = {
    "open_qa": {True: {True: PARAPHRASE_RAG_QA_PROMPT, False: RAG_QA_PROMPT}, False: QA_PROMPT},
    'fact_checking': {True: {True: PARAPHRASE_RAG_FECT_CHECKING_PROPMT, False: RAG_FECT_CHECKING_PROPMT}, False: FECT_CHECKING_PROPMT},
    'multiple_choice': {True: {True: PARAPHRASE_RAG_MULTIPLE_CHOICE_PROMPT, False: RAG_MULTIPLE_CHOICE_PROMPT}, False: MULTIPLE_CHOICE_PROMPT},
}
