"""Tries TranslateGemma (GGUF, llama.cpp) on caption-like sentences; writes results.txt."""
import json, os, sys, time, subprocess, urllib.request, glob
from huggingface_hub import HfApi, hf_hub_download
from gguf import GGUFReader

SENT = {
    "el": ["Γεια σας φίλοι μου, καλώς ήρθατε στο κανάλι!", "Σήμερα θα ρίξουμε τα ζάρια και θα δούμε τι θα γίνει.",
           "Ο δράκος εμφανίστηκε και όλοι πάγωσαν.", "Ποιος θέλει να παίξει μαζί μας;", "Μην ξεχάσετε να κάνετε like και subscribe!",
           "Έλα ρε, πάλι ένα το έφερα;", "Η μάγισσα ρίχνει ένα ξόρκι φωτιάς στο τέρας."],
    "ko": ["안녕하세요 여러분, 오늘은 주사위를 굴려볼게요.", "드래곤이 나타났어요!", "구독과 좋아요 부탁드려요!", "진짜 대박이다, 이거 봐!"],
    "en": ["Roll for initiative, everyone!", "The dragon breathes fire and the party scatters."],
}
PAIRS = [("el", "ko"), ("el", "en"), ("ko", "el"), ("en", "ko"), ("en", "el")]
out = open("results.txt", "w", encoding="utf-8")
api = HfApi()
for repo in sys.argv[1:]:
    files = [f for f in api.list_repo_files(repo) if f.endswith(".gguf") and "Q4_K_M" in f]
    if not files: out.write(f"no Q4_K_M in {repo}\n"); continue
    path = hf_hub_download(repo, files[0])
    out.write(f"\n===== {repo} / {files[0]} ({os.path.getsize(path)//1048576} MB) =====\n")
    r = GGUFReader(path)
    f = r.fields.get("tokenizer.chat_template")
    if f is not None:
        tmpl = bytes(f.parts[f.data[0]]).decode("utf-8")
        out.write("--- chat_template ---\n" + tmpl[:6000] + "\n---------------------\n")
    srv = subprocess.Popen([glob.glob("llama/**/llama-server", recursive=True)[0], "-m", path, "--jinja", "-c", "4096", "--port", "8091", "-t", "4"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(300):
        try: urllib.request.urlopen("http://127.0.0.1:8091/health", timeout=2).read(); break
        except Exception: time.sleep(1)
    for src, tgt in PAIRS:
        for s in SENT[src]:
            msg = [{"role": "user", "content": [{"type": "text", "source_lang_code": src, "target_lang_code": tgt, "text": s}]}]
            body = json.dumps({"messages": msg, "temperature": 0, "max_tokens": 200}).encode()
            t = time.time()
            try:
                res = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8091/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=600))
                txt = res["choices"][0]["message"]["content"].strip()
            except Exception as e:
                txt = f"ERROR {e}"
            out.write(f"[{src}->{tgt}] {s}\n    => {txt}   ({time.time()-t:.1f}s)\n"); out.flush()
    # the raw prompt as the template renders it, for the app
    try:
        msg = [{"role": "user", "content": [{"type": "text", "source_lang_code": "el", "target_lang_code": "ko", "text": "TEXT"}]}]
        res = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8091/apply-template", json.dumps({"messages": msg}).encode(), {"Content-Type": "application/json"}), timeout=60))
        out.write("--- rendered prompt ---\n" + json.dumps(res, ensure_ascii=False) + "\n")
    except Exception as e:
        out.write(f"apply-template failed {e}\n")
    srv.terminate(); srv.wait()
    os.remove(path)
out.close()
