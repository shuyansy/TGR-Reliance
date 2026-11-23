"""Starting model for RL from an SFT output folder: the SFT weights (symlinked) + the non-weight files of the base model (copied).
  python make_view.py --family qwen25|qwen3 <sft_output_dir> <base_model> <view_dir>
Why: the SFT environment (transformers 4.57) saves config / processor / chat template in its own format; the Qwen2.5 RL environment
(transformers 4.52) cannot read the new chat_template.jinja, and for both families RL and evaluation should use exactly the processor,
template and generation_config of the base model. The tensor names must equal those of the base model (checked).
"""
import argparse
import json
import os
import sys

WEIGHT_INDEX = "model.safetensors.index.json"
FILES = {"qwen25": ["chat_template.json", "config.json", "generation_config.json", "merges.txt", "preprocessor_config.json", "tokenizer.json",
                    "tokenizer_config.json", "vocab.json"]}
FILES["qwen3"] = FILES["qwen25"] + ["video_preprocessor_config.json"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["qwen25", "qwen3"])
    ap.add_argument("sft_dir")
    ap.add_argument("base")
    ap.add_argument("view")
    ap.add_argument("--no_load_check", action="store_true", help="skip loading config / processor with transformers")
    a = ap.parse_args()
    os.makedirs(a.view, exist_ok=True)
    weights = [f for f in os.listdir(a.sft_dir) if f.endswith(".safetensors")] + [WEIGHT_INDEX]
    for f in weights:
        src = os.path.join(os.path.abspath(a.sft_dir), f)
        assert os.path.exists(src), f"missing {src}"
        dst = os.path.join(a.view, f)
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(src, dst)
    import shutil
    for f in FILES[a.family]:
        shutil.copy2(os.path.join(a.base, f), os.path.join(a.view, f))
    wv = json.load(open(os.path.join(a.view, WEIGHT_INDEX)))["weight_map"]
    wb = json.load(open(os.path.join(a.base, WEIGHT_INDEX)))["weight_map"]
    assert set(wv) == set(wb), f"tensor names differ from the base model: extra {len(set(wv) - set(wb))} / missing {len(set(wb) - set(wv))}"
    assert all(os.path.exists(os.path.join(a.view, f)) for f in set(wv.values())), "missing weight shards"
    if not a.no_load_check:
        from transformers import AutoConfig, AutoProcessor, AutoTokenizer
        AutoConfig.from_pretrained(a.view)
        AutoProcessor.from_pretrained(a.view)
        s = AutoTokenizer.from_pretrained(a.view).apply_chat_template([{"role": "system", "content": "S"}, {"role": "user", "content": "Q"}],
                                                                       tokenize=False, add_generation_prompt=True)
        assert s.endswith("<|im_start|>assistant\n"), repr(s[-40:])
    print(f"[view] {a.view}: {len(wv)} tensors named as in the base model, weights from {a.sft_dir}, other files from {a.base}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
