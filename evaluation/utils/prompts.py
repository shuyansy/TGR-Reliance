"""Prompts. `tga` is the time-span prompt given to the backbones in Table 1; `train` is the prompt of our training data
(EviCe) and of our trained models. New models: pass their own system prompt / final instruction (see README)."""
PROMPTS = {
    "tga": ("You are given a video and a multiple-choice question. First identify the relevant time segment in seconds inside "
            "<think></think> as [start, end]. Then place the correct option letter inside <answer></answer>. "
            "Example: <think>[5.2, 10.4]</think><answer>B</answer>.",
            "Return the relevant time segment in <think></think> and the correct option letter in <answer></answer>."),
    "train": ("You are given a video and a multiple-choice question. First identify the relevant time segment(s) in seconds inside "
              "<think></think> as one or more [start, end] pairs. Then place the correct option letter inside <answer></answer>. "
              "Example: <think>[5.2, 10.4]</think><answer>B</answer> or <think>[5.2, 10.4], [40.0, 52.0]</think><answer>B</answer>.",
              "Return the relevant time segment(s) in <think></think> and the correct option letter in <answer></answer>."),
}


def user_text(question, options, suffix):
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "Question: " + question + "\nOptions:\n" + "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(options)) + "\n" + suffix
