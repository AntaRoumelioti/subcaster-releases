"""Converts an OPUS-MT (Marian) translation model to a small ONNX package for Subcaster:
encoder.onnx + decoder.onnx (int8), the SentencePiece pieces as JSON and the vocabulary.
usage: convert_mt.py <hf model id> <out dir>"""
import sys, os, json, subprocess, shutil
from onnxruntime.quantization import quantize_dynamic, QuantType
import sentencepiece as spm
from huggingface_hub import snapshot_download

mid, out = sys.argv[1], sys.argv[2]
tmp = out + "-fp32"
subprocess.check_call(["optimum-cli", "export", "onnx", "--model", mid, "--task", "text2text-generation", tmp])
os.makedirs(out, exist_ok=True)
for a, b in (("encoder_model.onnx", "encoder.onnx"), ("decoder_model.onnx", "decoder.onnx")):
    quantize_dynamic(os.path.join(tmp, a), os.path.join(out, b), weight_type=QuantType.QInt8)
src = snapshot_download(mid, allow_patterns=["*.spm", "vocab.json", "config.json", "generation_config.json", "README.md", "LICENSE*"])
def pieces(f):
    sp = spm.SentencePieceProcessor(model_file=f)
    return [[sp.id_to_piece(i), sp.get_score(i), int(sp.is_unknown(i) or sp.is_control(i))] for i in range(sp.get_piece_size())]
json.dump({"source": pieces(os.path.join(src, "source.spm")), "target": pieces(os.path.join(src, "target.spm"))}, open(os.path.join(out, "spm.json"), "w"), ensure_ascii=False)
shutil.copy(os.path.join(src, "vocab.json"), out)
cfg = json.load(open(os.path.join(src, "config.json")))
gen = json.load(open(os.path.join(src, "generation_config.json"))) if os.path.exists(os.path.join(src, "generation_config.json")) else {}
keep = {k: cfg.get(k) for k in ("decoder_start_token_id", "pad_token_id", "eos_token_id", "max_length", "num_beams", "bad_words_ids", "vocab_size", "d_model")}
keep.update({k: gen[k] for k in ("decoder_start_token_id", "pad_token_id", "eos_token_id", "max_length", "num_beams", "bad_words_ids") if k in gen})
keep["model"] = mid
json.dump(keep, open(os.path.join(out, "config.json"), "w"), indent=1)
for f in os.listdir(src):
    if f.startswith("README") or f.startswith("LICENSE"): shutil.copy(os.path.join(src, f), out)
print("ok", mid, {f: os.path.getsize(os.path.join(out, f)) for f in os.listdir(out)})
