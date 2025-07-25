# built-in
import argparse
import json
import os
import time
import yaml

# third party
from transformers import (
    MistralForCausalLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    MixtralForCausalLM,
    Qwen2ForCausalLM,
    AddedToken
)
import torch
import datasets
from tqdm import tqdm
import pandas as pd
import random
import math

# own
from src.model import (
    PriMistralForCausalLM,
    PriQwen2ForCausalLM,
    SFR,
    Qwen2,
    BGE_SMALL,
)

from src.training.utils import (
    PLACEHOLDER_TOKEN,
    get_retrieval_embeds,
    get_retrieval_embeds_query_aware,
)
from src.eval.utils import (
    stop_sequences_criteria,
    get_substring_match_score,
    eval_fact_checking,
    eval_truthfulqa,
    calculate_sacrebleu,
)
from src.utils import (
    get_jsonl,
    str2bool
)


def create_prompt_with_chat_format(messages, tokenizer, *args, **kwargs):
    return tokenizer.apply_chat_template(messages, tokenize=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_rag",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--enable_progress_bar",
        type=eval,
        default=True,
    )
    parser.add_argument(
        "--data",
    )
    parser.add_argument(
        "--model_name_or_path",
    )
    parser.add_argument(
        "--eval_metrics",
    )
    parser.add_argument(
        "--n_shot",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--retriever_name_or_path",
    )
    parser.add_argument(
        "--retrieval_topk",
        type=int,
        default=[1],
        nargs="+",
    )
    parser.add_argument(
        "--max_test_samples",
        type=int,
        help="for debug",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=4,
    )
    # Path to pre-computed embeddings to avoid recomputing them during evaluation
    parser.add_argument(
        "--embedding_path",
        default=None,
        help="Directory containing precomputed document embeddings (.pt files)"
    )
    # Directory where evaluation results will be saved
    parser.add_argument(
        "--output_path",
        default="res",
        help="Directory to save evaluation results and generated outputs"
    )
    # Maximum token length for retrieval context documents
    parser.add_argument(
        '--retrieval_context_max',
        type=int,
        default=180,
        help='Maximum sequence length for retrieved documents (tokens)'
    )
    # Controls the probability of using query-aware retrieval during evaluation
    parser.add_argument(
        "--query_aware_rate",
        type=float,
        default=0.0,
        help="Probability (0-1) of using query-aware retrieval for each sample"
    )
    parser.add_argument(
        "--add_dp_noise",
        type=str2bool,
        default=False,
        help="whether to add dp noise to the proxy embedding",
    )
    parser.add_argument(
        "--dp_epsilon",
        type=float,
        default=1.0,
        help="epsilon for gaussian dp",
    )
    parser.add_argument(
        "--dp_delta",
        type=float,
        default=1e-5,
        help="delta for gaussian dp",
    )
    parser.add_argument(
        "--dp_sensitivity",
        type=float,
        default=1,
        help="sensitivity for gaussian dp",
    )

    # Enables target adaptive attack evaluation scenario
    parser.add_argument(
        "--target_adaptive_attack",
        type=str2bool,
        default=False,
        help="Whether to evaluate under a target adaptive attack scenario"
    )
    # Enables untargeted model inversion attack evaluation scenario
    parser.add_argument(
        "--untarget_inversion_attack",
        type=str2bool,
        default=False,
        help="Whether to evaluate the model's resistance to untargeted inversion attacks"
    )
    args = parser.parse_args()

    # Post-processing of arguments based on evaluation scenario
    # Set appropriate task type and metrics based on the evaluation mode
    if args.untarget_inversion_attack:
        # For inversion attacks, use paraphrase task and measure semantic similarity metrics
        args.task_type = "paraphase"
        args.eval_metrics = "f1_rl_bleu"
    else:
        # For standard evaluation, set task type and metrics based on dataset
        if args.data in [
            "nq_open",
            "hotpotqa",
            "triviaqa",
            "webqa",
            "PubMedQA",
            "2WikiMultiHopQA",
            "BioASQ-TaskB",
        ]:
            args.task_type = "open_qa"
            args.eval_metrics = "substring_match"
        elif args.data in ["truthfulqa", "PubMedQA_long_answer", "ms_marco", "emrqa"]:
            args.task_type = "open_qa"
            args.eval_metrics = "f1_rl_bleu"
        elif args.data in ["factkg", "strategyqa"]:
            args.task_type = "fact_checking"
            args.eval_metrics = "fact_checking_acc"
            if args.target_adaptive_attack:
                args.task_type = "open_qa"
                args.eval_metrics = "substring_match"

    # rank starts from 1
    args.retrieval_topk = [x - 1 for x in args.retrieval_topk]

    args.chat_format = eval(f"create_prompt_with_chat_format")

    if args.retriever_name_or_path is not None:
        args.use_rag = True

    return args


# Template definitions for different task types
# These are used to format prompts consistently based on the evaluation task
QA_PROMPT = "Question: {question}?\n"
FECT_CHECKING_PROPMT = "Claim: {question}\n"
BACKGROUND_PROMPT_TEMPLATE = "Background: {background}\n\n"
PARAPHRASE_PROMPT_TEMPLATE = "{question}\n"

# Dictionary mapping task types to their corresponding prompt templates
# This centralizes prompt formatting for different evaluation scenarios
PROMPT_TEMPLATES = {
    "open_qa": QA_PROMPT,
    "fact_checking": FECT_CHECKING_PROPMT,
    "paraphase": PARAPHRASE_PROMPT_TEMPLATE,
}


def get_start_prompt(task_type, use_rag, sample=None):
    """
    Generate the appropriate instruction text based on task type and retrieval mode.

    This function returns task-specific instructions that tell the model what to do,
    with variations depending on whether retrieval-augmented generation is enabled.

    Args:
        task_type: The type of task (open_qa, fact_checking, or paraphase)
        use_rag: Boolean indicating whether retrieval-augmented generation is enabled
        sample: Optional sample data (unused but kept for potential future extensions)

    Returns:
        str: The appropriate instruction text for the specified task and mode
    """
    if task_type == "open_qa":
        return {
            True: "Refer to the background document and answer the questions:",
            False: "Answer the questions:",
        }[use_rag]
    elif task_type == "fact_checking":
        return {
            True: 'Refer to the background document and verify the following claims with "True" or "False":',
            False: 'Verify the following claims with "True" or "False":',
        }[use_rag]
    elif task_type == "paraphase":
        return {
            True: "Refer to the background document and provide a reinterpretation of the preceding background text:",
            False: "Paraphrase the following sentences:",
        }[use_rag]


@torch.no_grad()
def prepare_retrieval_embeds(
    backgrounds, questions, retriever, tokenizer, embedding_path, batch_size=16
):
    """
    Prepare document embeddings for retrieval-augmented generation.

    This function either loads pre-computed embeddings from disk or generates
    new embeddings using the retriever model. Query-aware embeddings can be 
    conditionally generated based on the global query_aware_rate setting.

    Args:
        backgrounds: List of background documents to embed
        questions: List of questions (used for query-aware embedding)
        retriever: The retriever model for generating embeddings
        tokenizer: Tokenizer for the retriever model
        embedding_path: Path to pre-computed embeddings (if available)
        batch_size: Number of documents to process in each batch

    Returns:
        list: Document embeddings for retrieval-augmented generation
    """
    if embedding_path is not None:
        # Load pre-computed embeddings from disk to save computation time
        embeddings = []
        for file in os.listdir(embedding_path):
            if file.endswith(".pt"):
                embeddings.extend(torch.load(
                    os.path.join(embedding_path, file)))
        return embeddings
    else:
        # Process backgrounds in batches to manage memory usage
        backgrounds = [
            backgrounds[idx: idx + batch_size]
            for idx in range(0, len(backgrounds), batch_size)
        ]
        questions = [
            questions[idx: idx + batch_size]
            for idx in range(0, len(questions), batch_size)
        ]
        device = retriever.device
        ret = []
        for background, question in zip(backgrounds, questions):
            tokenized_retrieval_text = tokenizer(
                background,
                max_length=args.retrieval_context_max,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            tokenized_query = tokenizer(
                question,
                max_length=tokenized_retrieval_text["input_ids"].shape[1],
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            # return a torch tensor of shape [batch_size,d_model]
            query_aware_ = random.random()
            if query_aware_ < args.query_aware_rate:
                embeds = get_retrieval_embeds_query_aware(
                    retriever=retriever,
                    input_query_ids=tokenized_query['input_ids'].to(device),
                    query_attention_mask=tokenized_query['attention_mask'].to(
                        device),
                    input_ids=tokenized_retrieval_text["input_ids"].to(device),
                    attention_mask=tokenized_retrieval_text["attention_mask"].to(
                        device),
                ).cpu()
            else:
                embeds = get_retrieval_embeds(
                    model=retriever,
                    input_ids=tokenized_retrieval_text["input_ids"].to(device),
                    attention_mask=tokenized_retrieval_text["attention_mask"].to(
                        device),
                ).cpu()

            embeds = [embeds[idx] for idx in range(embeds.shape[0])]
            ret.extend(embeds)
        return ret


@torch.no_grad()
def llm_for_open_generation(
    llm,
    llm_tokenizer,
    prompts,
    retrieval_embeds,
    batch_size=4,
    enable_progress_bar=True,
):
    generated_answers = []
    total_test_number = len(prompts)
    device = llm.device
    batched_prompts = [
        prompts[idx: idx + batch_size] for idx in range(0, len(prompts), batch_size)
    ]
    if retrieval_embeds is not None:
        batched_retrieval_embeds = [
            retrieval_embeds[idx: idx + batch_size]
            for idx in range(0, len(retrieval_embeds), batch_size)
        ]
        assert len(batched_prompts) == len(batched_retrieval_embeds)

    progress_bar = tqdm(
        range(total_test_number), ncols=60, disable=not enable_progress_bar
    )
    for batch_idx in range(len(batched_prompts)):
        prompt = batched_prompts[batch_idx]
        tokenized_propmt = llm_tokenizer(
            prompt, padding="longest", return_tensors="pt")
        input_ids = tokenized_propmt.input_ids.to(device)
        attention_mask = tokenized_propmt.attention_mask.to(device)
        stopping_criteria = stop_sequences_criteria(
            llm_tokenizer, input_ids.shape[1], input_ids.shape[0]
        )
        retrieval_kwargs = {}
        if retrieval_embeds is not None:
            embeds = batched_retrieval_embeds[batch_idx]
            embeds = [x for y in embeds for x in y]
            embeds = torch.stack(embeds).to(device)
            retrieval_kwargs["retrieval_embeds"] = embeds
            stopping_criteria = stop_sequences_criteria(
                llm_tokenizer, 0, input_ids.shape[0]
            )

        # actual computation
        generated_output = llm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            stopping_criteria=stopping_criteria,
            do_sample=False,
            max_new_tokens=100,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
            **retrieval_kwargs,
        )
        # because HF generate with inputs_embeds would not return prompt
        input_length = 0 if retrieval_kwargs else input_ids.shape[1]
        results = tokenizer.batch_decode(
            generated_output[:, input_length:], skip_special_tokens=False
        )
        generated_answers.extend(results)
        progress_bar.update(batch_size)

    generated_answers = [x.strip() for x in generated_answers]
    return generated_answers


def format_one_example(
    sample,
    include_answer,
    use_rag,
    retrieval_embed_length,
    task_type,
    model,
    tokenizer,
):
    if args.untarget_inversion_attack:
        question = ""
    else:
        question = sample["question"]
    prompt_dict = dict(question=question)
    prompt = PROMPT_TEMPLATES[task_type].format_map(prompt_dict).strip()
    backgrounds = []

    if use_rag:
        backgrounds = sample["background"]  # a list
        background_prompts = ""

        for background in backgrounds:
            if retrieval_embed_length > 0:
                background_prompts += (
                    " ".join([PLACEHOLDER_TOKEN] *
                             retrieval_embed_length) + " "
                )

            else:
                background_prompts += background + " "
        background_prompts = background_prompts.strip()
        prompt = (
            BACKGROUND_PROMPT_TEMPLATE.format_map(
                dict(background=background_prompts))
            + prompt
        )

    return prompt, backgrounds


def get_n_shot_prompt(
    dev_data, n_shot, task_type, use_rag=False, retrieval_embed_length=0, model=None, tokenizer=None
):
    """
    Generate n-shot prompts for few-shot learning scenarios.

    This function creates prompt examples with corresponding background information
    for a few-shot learning setup, where the model is shown a few examples of the task
    before being asked to perform the task on new data.

    Args:
        dev_data: Development data containing examples for n-shot learning
        n_shot: Number of shot (examples) to include
        task_type: The type of task (open_qa, fact_checking, or paraphase)
        use_rag: Boolean indicating whether to use retrieval-augmented generation
        retrieval_embed_length: Length of retrieval embeddings (for placeholder tokens)
        model: The model being used (for tokenization)
        tokenizer: The tokenizer being used

    Returns:
        tuple: A tuple containing two lists - n_shot_prompt and n_shot_background
    """
    assert n_shot >= 0, n_shot
    n_shot_prompt = []
    n_shot_background = []
    if dev_data is not None:
        n_shot_examples = dev_data[:n_shot]
        for example in n_shot_examples:
            prompt, background = format_one_example(
                example,
                include_answer=True,
                use_rag=use_rag,
                retrieval_embed_length=retrieval_embed_length,
                task_type=task_type,
                model=model,
                tokenizer=tokenizer,
            )
            n_shot_prompt.append(prompt)
            n_shot_background.append(background)

    return n_shot_prompt, n_shot_background


def prepare_prompts(
    dev_data,
    test_data,
    task_type,
    tokenizer,
    model,
    n_shot=0,
    use_rag=False,
    retrieval_embed_length=0,
    chat_format=None,
):
    """
    Prepare prompts for evaluation or inference.

    This function constructs the full prompts that will be fed into the language model,
    including any necessary background information and few-shot examples. It also handles
    the formatting required for different task types and retrieval modes.

    Args:
        dev_data: Development data for few-shot examples (if applicable)
        test_data: Test data that requires prompting
        task_type: The type of task (open_qa, fact_checking, or paraphase)
        tokenizer: Tokenizer for the language model
        model: The language model being used
        n_shot: Number of shot (examples) to include from the development data
        use_rag: Boolean indicating whether to use retrieval-augmented generation
        retrieval_embed_length: Length of retrieval embeddings (for placeholder tokens)
        chat_format: Optional chat format function for formatting prompts

    Returns:
        tuple: A tuple containing three lists - prompts, questions, and backgrounds
    """
    splitter = "\n\n"
    prompts = []
    backgrounds = []
    questions = []
    original_n_shot = n_shot
    print("len(test_data)", len(test_data))
    for idx, sample in enumerate(test_data):
        n_shot = original_n_shot
        while True:
            prompt_start = get_start_prompt(
                task_type, use_rag=use_rag, sample=sample)
            question = sample["question"]

            # prompt = PROMPT_TEMPLATES[task_type].format_map(prompt_dict).strip()
            prompt_end, background = format_one_example(
                sample,
                include_answer=False,
                use_rag=use_rag,
                retrieval_embed_length=retrieval_embed_length,
                task_type=task_type,
                model=model,
                tokenizer=tokenizer,
            )
            if "subject" not in sample.keys():
                n_shot_prompt, n_shot_background = get_n_shot_prompt(
                    dev_data,
                    n_shot=n_shot,
                    use_rag=use_rag,
                    retrieval_embed_length=retrieval_embed_length,
                    task_type=task_type,
                    model=model,
                    tokenizer=tokenizer,
                )
            else:
                # select n-shot within the same subjects for MMLU
                dev_data_with_same_subjects = []
                for d in dev_data:
                    if d["subject"] == sample["subject"]:
                        dev_data_with_same_subjects.append(d)
                assert len(dev_data_with_same_subjects) == 5, sample["subject"]
                n_shot_prompt, n_shot_background = get_n_shot_prompt(
                    dev_data_with_same_subjects,
                    n_shot=n_shot,
                    use_rag=use_rag,
                    retrieval_embed_length=retrieval_embed_length,
                    task_type=task_type,
                    model=model,
                    tokenizer=tokenizer,
                )

            if n_shot_prompt:
                prompt = (
                    prompt_start
                    + splitter
                    + splitter.join(n_shot_prompt)
                    + splitter
                    + prompt_end
                )
            else:
                prompt = prompt_start + splitter + prompt_end

            if chat_format is not None:
                messages = [{"role": "user", "content": prompt}]
                if args.untarget_inversion_attack:
                    prompt = chat_format(messages, tokenizer) + \
                        "The paraphrased text is"
                else:
                    prompt = chat_format(
                        messages, tokenizer) + " The answer is:"

            tokenized_prompt = tokenizer(
                prompt, truncation=False, add_special_tokens=False
            ).input_ids

            if len(tokenized_prompt) > 2048 and n_shot >= 1:
                n_shot -= 1
            else:
                break

        prompts.append(prompt)
        questions.append(question)
        backgrounds.append(background + n_shot_background)

    print("**" * 20, "show prompt example", "**" * 20)
    print(prompts[0])
    print("**" * 20, "show backgrounds example", "**" * 20)
    print(backgrounds[0])
    print("**" * 20, "show questions example", "**" * 20)
    print(questions[0])

    return prompts, questions, backgrounds


def load_dataset(data, use_rag, args):
    if args.target_adaptive_attack:
        data = data + "_adapt"
    dev_data = None
    query_path = f"data/eval/{data}/query.jsonl"
    test_data = None
    if os.path.isfile(query_path):
        test_data = get_jsonl(query_path)

    if use_rag:

        test_retrieval_path = f"data/eval/{data}/context.jsonl"
        test_retrieval = get_jsonl(test_retrieval_path)
        assert len(test_retrieval) == len(test_data)
        for idx in range(len(test_data)):
            test_data[idx]["background"] = [
                test_retrieval[idx]["topk"][rank]["text"]
                for rank in args.retrieval_topk
            ]

    return dev_data, test_data


def add_gaussian_noise(tensor, epsilon=1.0, delta=1e-5, sensitivity=1.0):
    """
    添加高斯噪声实现(ϵ,δ)-差分隐私
    """
    sigma = math.sqrt(2 * math.log(1.25/delta)) * sensitivity / epsilon
    noise = torch.randn_like(tensor) * sigma
    return tensor + noise


args = parse_args()
if __name__ == "__main__":

    if args.use_rag:
        if args.retriever_name_or_path is not None:
            args.eval_type = "PriRAG"
        else:
            args.eval_type = "RAG"
    else:
        args.eval_type = "woRAG"

    # load llm
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    MODEL_CLASS = eval(config.architectures[0])
    model = MODEL_CLASS.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )

    model.eval()
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side="left",
        add_eos_token=False,  # import to include this!
        use_fast=False,
    )
    if tokenizer.pad_token:
        pass
    elif tokenizer.unk_token:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    elif tokenizer.eos_token:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if (
        args.retriever_name_or_path is not None
        and args.model_name_or_path.lower() == args.retriever_name_or_path.lower()
    ):
        retriever = model
        retriever_tokenizer = tokenizer
    else:
        # load retriever and retriever_tokenizer
        device = (
            torch.device("cuda") if torch.cuda.is_available(
            ) else torch.device("cpu")
        )
        retriever, retriever_tokenizer = None, None
        if args.retriever_name_or_path is not None:
            if (
                "salesforce/sfr-embedding-mistral"
                in args.retriever_name_or_path.lower()
            ):
                retriever = SFR.from_pretrained(
                    args.retriever_name_or_path, torch_dtype=torch.bfloat16
                )
                retriever_tokenizer = AutoTokenizer.from_pretrained(
                    args.retriever_name_or_path
                )
            elif (
                "bge" in args.retriever_name_or_path.lower()
            ):
                retriever = BGE_SMALL.from_pretrained(
                    args.retriever_name_or_path, torch_dtype=torch.bfloat16
                )
                retriever_tokenizer = AutoTokenizer.from_pretrained(
                    args.retriever_name_or_path
                )
            elif (
                "qwen2" in args.retriever_name_or_path.lower()
            ):
                retriever = Qwen2.from_pretrained(
                    args.retriever_name_or_path, torch_dtype=torch.bfloat16
                )
                retriever_tokenizer = AutoTokenizer.from_pretrained(
                    args.retriever_name_or_path
                )
            retriever.eval()
            retriever = retriever.to(device)
    retrieval_embed_length = retriever.get_embed_length()

    # set Placeholder for model
    # check if the model has a special token for Placeholder
    vocab = tokenizer.get_vocab()
    if PLACEHOLDER_TOKEN not in vocab:
        tokenizer.add_tokens(
            [AddedToken(PLACEHOLDER_TOKEN, lstrip=False, rstrip=False)])
        placeholder_token_id = tokenizer.convert_tokens_to_ids(
            PLACEHOLDER_TOKEN)
        model.resize_token_embeddings(len(tokenizer))

    # prepare prompt
    dev_data, test_data = load_dataset(
        args.data,
        args.use_rag,
        args,
    )

    if args.max_test_samples is not None:
        test_data = test_data[: args.max_test_samples]

    prompts, questions, backgrounds = prepare_prompts(
        dev_data=dev_data,
        test_data=test_data,
        task_type=args.task_type,
        tokenizer=tokenizer,
        model=model,
        n_shot=args.n_shot,
        use_rag=args.use_rag,
        retrieval_embed_length=retrieval_embed_length,
        chat_format=args.chat_format,
    )

    retrieval_embeds = None
    if retriever is not None:
        num_samples = len(backgrounds)
        original_orders = []
        for idx, background in enumerate(backgrounds):
            original_orders.extend([idx] * len(background))

        backgrounds = [x for y in backgrounds for x in y]
        print(
            f"Preparing document embedding with {args.retriever_name_or_path}...")
        _retrieval_embeds = prepare_retrieval_embeds(
            backgrounds,
            questions,
            retriever,
            retriever_tokenizer,
            embedding_path=args.embedding_path,
        )
        retrieval_embeds = [[] for _ in range(num_samples)]
        assert len(_retrieval_embeds) == len(original_orders)
        for id, embeds in zip(original_orders, _retrieval_embeds):
            retrieval_embeds[id].append(embeds)
        print("len of retrieval_embeds", len(retrieval_embeds))
        if args.add_dp_noise:
            for idx in range(len(retrieval_embeds)):
                retrieval_embeds[idx][0] = add_gaussian_noise(
                    retrieval_embeds[idx][0],
                    epsilon=args.dp_epsilon,
                    delta=args.dp_delta,
                    sensitivity=args.dp_sensitivity
                )

    avg_prompt_length = tokenizer(prompts, return_length=True).length
    avg_prompt_length = sum(avg_prompt_length) / len(avg_prompt_length)

    if retriever is not None:
        assert PLACEHOLDER_TOKEN in tokenizer.get_vocab()
        model.set_placeholder_token_id(
            tokenizer.convert_tokens_to_ids(PLACEHOLDER_TOKEN))

    if args.task_type in ["open_qa", "fact_checking", "paraphase"]:
        generated_results = llm_for_open_generation(
            llm=model,
            llm_tokenizer=tokenizer,
            prompts=prompts,
            retrieval_embeds=retrieval_embeds,
            batch_size=args.eval_batch_size,
            enable_progress_bar=args.enable_progress_bar,
        )

    if args.untarget_inversion_attack:
        answers = [x["background"] for x in test_data]
    else:
        answers = [x["answer"] for x in test_data]
    if args.eval_metrics == "substring_match":
        score, score_per_sample = get_substring_match_score(
            generated_results, answers)
    elif args.eval_metrics == "fact_checking_acc":
        score, score_per_sample = eval_fact_checking(
            generated_results, answers)
    elif args.eval_metrics == "f1_rl_bleu":
        f1, rl, f1_scores, rl_scores = eval_truthfulqa(
            generated_results, answers)
        bleu, bleu_scores = calculate_sacrebleu(generated_results, answers)
        score = f"{f1}-{rl}-{bleu}"
        score_per_sample = [
            (f1_score, rl_score, bleu_score) for f1_score, rl_score, bleu_score in zip(f1_scores, rl_scores, bleu_scores)
        ]

    result_dict = {
        "dataset": args.data,
        "batch_size": args.eval_batch_size,
        "include_retrieval": args.use_rag,
        "avg_prompt_length": avg_prompt_length,
        "model": args.model_name_or_path,
        f"{args.eval_metrics}": score,
    }

    if args.retriever_name_or_path is not None:
        result_dict["retriever"] = args.retriever_name_or_path
    print(json.dumps(result_dict, indent=4))

    if args.output_path is not None:
        args.output_path = os.path.join(
            args.output_path,
            args.data, args.eval_type, str(time.strftime("%Y%m%d_%H%M%S")),
        )
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
        with open(os.path.join(args.output_path, args.data+"_results.json"), "w") as f:
            json.dump(result_dict, f, indent=4)
        with open(os.path.join(args.output_path, args.data+"generated_results.jsonl"), "w") as f:
            for idx, (generated, answer) in enumerate(
                zip(
                    generated_results,
                    answers,
                )
            ):
                f.write(
                    json.dumps(
                        {
                            "generated": generated,
                            "answer": answer,
                            "score": score_per_sample[idx],
                            "question": test_data[idx]["question"],
                        }
                    )
                )
                f.write("\n")
        with open(os.path.join(args.output_path, "config.yaml"), "w") as f:
            yaml_config = vars(args)
            yaml.dump(yaml_config, f)

        print(f"Results saved to {args.output_path}")
