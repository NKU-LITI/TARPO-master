import re
import string

ANSWER_START = "####"

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and "
    "the assistant solves it. The assistant first thinks about the reasoning "
    "process in the mind and then provides the user with the answer. The "
    "final answer is provided after the " + ANSWER_START + " tag, i.e., "
    "{reasoning process} " + ANSWER_START + " {answer}."
)

QA_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a multiple-choice question, and "
    "the assistant solves it. The assistant first thinks about the reasoning "
    "process in the mind and then provides the user with the answer. The "
    "final answer is provided after the " + ANSWER_START + " tag, and MUST be exactly one of the option letters (A, B, C, or D). "
    "i.e., {reasoning process} " + ANSWER_START + " {option letter}."
)

def extract_from_response(text: str) -> str:
    try:
        answer = text.split(ANSWER_START)[-1].strip()
        if answer.endswith("."):
            answer = answer[:-1].strip()
        return answer
    except IndexError:
        return ""


def extract_hash_answer(text: str) -> str | None:
    try:
        return text.split("####")[1].strip()
    except IndexError:
        return None


def extract_boxed_answer(text: str) -> str | None:
    try:  # wrap in boxed for process_math_answer
        return "\\boxed{" + find_box(text).strip() + "}"
    except IndexError:
        return None


def get_reward_func(process_answer_func):
    def reward_func(completions, answer, **kwargs) -> list[float]:
        responses = [completion[0]["content"] for completion in completions]

        ans = [process_answer_func(a) for a in answer]
        extracted = [extract_from_response(r) for r in responses]
        predictions = [process_answer_func(r) for r in extracted]
        accuracy = [True if r == a else False for r, a in zip(predictions, ans)]

        escaped_answer_start = re.escape(ANSWER_START)
        pattern = f"^(?:(?!{escaped_answer_start}).)*{escaped_answer_start}(?:(?!{escaped_answer_start}).)*$"
        matches = [bool(re.search(pattern, r, re.DOTALL)) for r in responses]

        rewards = [1.0 if a and m else 0.0 for a, m in zip(accuracy, matches)]

        print(
            "=" * 50,
            f"\nBatch accuracy: " + "".join("Y" if r > 0 else "N" for r in rewards),
            f"\n1/{len(completions)} responses (answer: {ans[0]}):\n{responses[0]}",
            "\n" + "=" * 50,
        )
        return rewards
    
    return reward_func



def reward_func_rag(completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]

    extracted = [extract_from_response(r) for r in responses]
    predictions = [process_qa_answer(r) for r in extracted]
    # accuracy = [True if r == a else False for r, a in zip(predictions, ans)]
    accuracy = []
    for normalized_pred, answer_list in zip(predictions, answer):
        cur_accuracy = False
        for golden_answer in answer_list:
            normalized_answer = process_qa_answer(golden_answer)
            if normalized_answer == normalized_pred:
                cur_accuracy = True
        accuracy.append(cur_accuracy)

    escaped_answer_start = re.escape(ANSWER_START)
    pattern = f"^(?:(?!{escaped_answer_start}).)*{escaped_answer_start}(?:(?!{escaped_answer_start}).)*$"
    matches = [bool(re.search(pattern, r, re.DOTALL)) for r in responses]

    rewards = [1.0 if a and m else 0.0 for a, m in zip(accuracy, matches)]

    print(
        "=" * 50,
        f"\nBatch accuracy: " + "".join("Y" if r > 0 else "N" for r in rewards),
        f"\n1/{len(completions)} responses (answer: {answer[0]}):\n{responses[0]}",
        "\n" + "=" * 50,
    )
    return rewards


def delete_extra_zero(n):
    try:
        n=float(n)
    except:
        try:
            n = eval(n)
        except:
            print("Conversion to floating number fails: {}".format(n))
            return n
    if isinstance(n, int):
        return str(n)
    if isinstance(n, float):
        n = str(n).rstrip("0")
        n = int(n.rstrip(".")) if n.endswith(".") else float(n)
        n=str(n)
        return n


def find_box(pred_str: str):
    ans = pred_str.split("boxed")[-1]
    if not ans:
        return ""
    if (ans[0] == "{"):
        stack = 1
        a = ""
        for c in ans[1:]:
            if (c == "{"):
                stack += 1
                a += c
            elif (c == "}"):
                stack -= 1
                if (stack == 0): break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()
    return a


def find_latex(pred_str: str):
    pattern = re.compile(
        r"""
        (\\\[(?P<display>[\s\S]+?)\\\])                        # \[ ... \]
        |(\\\((?P<inline>[\s\S]+?)\\\))                        # \( ... \)
        |((?P<dollar>\$\$?)(?P<dcontent>[\s\S]+?)(?P=dollar))  # $...$ or $$...$$
        """,
        re.VERBOSE
    )
    matches = list(pattern.finditer(pred_str))
    if not matches: return ""
    return (matches[-1].group("display") or matches[-1].group("inline") 
            or matches[-1].group("dcontent")).strip()


def _remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        return splits[0]
    else:
        return string


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr:
                if substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except:
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        else:
                            new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split and split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _strip_string(string):
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.strip("$")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    string = _fix_a_slash_b(string)
    return string


def process_gsm8k_answer(pred: str) -> str:
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ")
    pred = [delete_extra_zero(s.replace(",", "")) 
            for s in re.findall(r"-?\d+/?\.?\d*", pred)]

    if len(pred) == 0:
        pred = ""
    else:
        pred = pred[-1].rstrip(".").rstrip("/")
    return pred


def process_math_answer(pred: str) -> str:
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ")
    if "boxed" in pred:
        pred = find_box(pred)
    elif find_latex(pred):
        pred = find_latex(pred)
    else:
        preds = re.findall(r"-?\d*\.?\d+", pred)
        if(len(preds) >= 1):
            pred = preds[-1]
        else:
            pred = ""

    pred = _strip_string(pred).rstrip(".").rstrip("/")
    pred = re.sub(r"\\text\{([^}]*)\}", r"\1", pred).lower()
    return pred


def process_mmlu_answer(pred: str) -> str:
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ")
    tmp = re.findall(r"\b(A|B|C|D|E|F|G|H|I|J)\b", pred.upper())
    if tmp:
        pred = tmp
    else:
        pred = [pred.strip().strip(".")]

    if len(pred) == 0:
        pred = ""
    else:
        pred = pred[-1].rstrip(".").rstrip("/")
    return pred


def process_qa_answer(pred: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(pred))))


def process_gsm8k(batch):
    prompts = [[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": q.strip()},
    ] for q in batch["question"]]

    return {
        "prompt": prompts,
        "answer": [extract_hash_answer(a) for a in batch["answer"]]
    }


def process_math(batch):
    prompts = [[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": q.strip()},
    ] for q in batch["problem"]]

    return {
        "prompt": prompts,
        "answer": [extract_boxed_answer(a) for a in batch["solution"]]
    }


def process_mmlu(batch):
    def get_prompt(question, choices):
        prompt = f"Question: {question}\nOptions:\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65 + i)}. {choice}\n"
        return prompt

    prompts = [[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": get_prompt(q, c).strip()},
    ] for q, c in zip(batch["question"], batch["choices"])]

    return {
        "prompt": prompts,
        "answer": [f"{chr(65 + a)}" for a in batch["answer"]]
    }


def process_rag(batch, topk=3):
    def get_prompt(question, contexts):
        prompt = "Context (which may or may not be relevant):\n"
        for context in contexts[:topk]:
            cur_context = context.split("\n")
            cur_context[0] = cur_context[0].strip('"')
            prompt += "::::".join(cur_context) + "\n"
        prompt += f"\nQuestion: {question}"
        return prompt

    prompts = [[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": get_prompt(q, c).strip()},
    ] for q, c in zip(batch["question"], batch["contexts"])]

    return {
        "prompt": prompts,
        "answer": batch["golden_answers"],
    }


def process_dapo(batch):
    prompts = [[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": q.strip()},
    ] for q in batch["problem"]]

    return {
        "prompt": prompts,
        "answer": ["\\boxed{" + str(a) + "}" for a in batch["solution"]]
    }


def process_gpqa_answer(pred: str) -> str:
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ")
    tmp = re.findall(r"\b(A|B|C|D)\b", pred.upper())
    if tmp:
        return tmp[-1]
    return ""


def extract_gpqa_true_answer(item: dict) -> str:
    return item.get("final_answer", "").strip().upper()

