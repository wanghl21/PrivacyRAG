import datasets
import random
import json

templates_for_qa = [
    "Question: {question}?\nAnswer:",
    "{question}?",
    "Answer the following question:\n\n{question}",
    "Answer this question:\n\n{question}?",
    "Please answer this question: {question}",
    "Answer the question...{question}?",
    "What is the answer to this question? {question}\n\n",
    "Can you tell me the answer to {question}?",
    "Next question: {question}\n\n",
    "Q: {question} A:",
    "{question}\nWhat is the answer?",
    "Write the answer: {question}",
    "{question}???",
]

if __name__ == "__main__":
    total_data = []
    # prepare squad_v2
    data = datasets.load_dataset("squad_v2")
    print(len(data["train"]))
    print(data["train"][0])
    for idx, sample in enumerate(data["train"]):
        messages = []
        question = sample["question"]
        answer = (
            sample["answers"]["text"][0]
            if len(sample["answers"]["text"]) > 0
            else "I don't know."
        )
        question = random.choice(templates_for_qa).format_map(dict(question=question))

        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})

        total_data.append(
            {
                "id": f"squad_v2_{idx}",
                "messages": messages,
                "task_type": "close_qa",
                "background": sample["context"],
            }
        )
    # prepare pwc
    dataset = datasets.load_dataset("sggetao/PwC", split="train")
    print(len(dataset))
    for idx, sample in enumerate(dataset):
        messages = []
        answer = sample["answer"]
        question = sample["prompt"]
        question = random.choice(templates_for_qa).format_map(dict(question=question))

        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})

        total_data.append(
            {
                "id": f"pwc_{idx}",
                "messages": messages,
                "task_type": "close_qa",
                "background": sample["input"],
            }
        )
    # shuffle the data
    random.shuffle(total_data)
    # save the data
    with open("data/finetune/pwc_squad_v2.jsonl", "w") as f:
        for sample in total_data:
            f.write(json.dumps(sample) + "\n")
