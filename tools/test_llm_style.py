"""Which local model translates captions most naturally, and follows Korean 해요체? Writes style.txt."""
import json, os, sys, time, subprocess, urllib.request, glob
from huggingface_hub import HfApi, hf_hub_download

EL = ["Γεια σας φίλοι μου, καλώς ήρθατε στο κανάλι!", "Σήμερα θα ρίξουμε τα ζάρια και θα δούμε τι θα γίνει.",
      "Ο δράκος εμφανίστηκε και όλοι πάγωσαν.", "Ποιος θέλει να παίξει μαζί μας;", "Μην ξεχάσετε να κάνετε like και subscribe!",
      "Έλα ρε, πάλι ένα το έφερα;", "Η μάγισσα ρίχνει ένα ξόρκι φωτιάς στο τέρας."]
KO = ["안녕하세요 여러분, 오늘은 주사위를 굴려볼게요.", "드래곤이 나타났어요!", "진짜 대박이다, 이거 봐!"]
NAMES = {"el": "Greek", "ko": "Korean", "en": "English"}
STYLE = {"ko": "Use polite informal Korean, 해요체 (sentence endings -아요/-어요/해요). Never use 합니다체 and never use 반말.",
         "el": "Use natural, everyday spoken Greek, as a Greek YouTuber would say it.", "en": "Use natural, casual spoken English."}
def prompt(src, tgt, text):
    return (f"Translate this {NAMES[src]} video subtitle into {NAMES[tgt]} for a social media video. {STYLE[tgt]} "
            f"Keep it short like a subtitle and keep names unchanged. Output only the translation.\n\n{text}")
out = open("style.txt", "w", encoding="utf-8")
api = HfApi()
for repo in sys.argv[1:]:
    try:
        files = sorted(f for f in api.list_repo_files(repo) if f.endswith(".gguf") and "Q4_K_M" in f and "mmproj" not in f)
    except Exception as e:
        out.write(f"\n===== {repo}: {e}\n"); continue
    if not files: out.write(f"\n===== {repo}: no Q4_K_M\n"); continue
    path = hf_hub_download(repo, files[0])
    out.write(f"\n===== {repo} / {files[0]} ({os.path.getsize(path)//1048576} MB) =====\n"); out.flush()
    srv = subprocess.Popen([glob.glob("llama/**/llama-server", recursive=True)[0], "-m", path, "--jinja", "-c", "4096", "--port", "8092", "-t", "4"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(300):
        try: urllib.request.urlopen("http://127.0.0.1:8092/health", timeout=2).read(); break
        except Exception: time.sleep(1)
    for src, tgt, sents in (("el", "ko", EL), ("ko", "el", KO), ("el", "en", EL[:4])):
        for s in sents:
            msg = [{"role": "user", "content": prompt(src, tgt, s) + (" /no_think" if "Qwen3" in repo else "")}]
            body = json.dumps({"messages": msg, "temperature": 0.2, "max_tokens": 300}).encode()
            t = time.time()
            try:
                res = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8092/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=900))
                txt = res["choices"][0]["message"]["content"].strip()
                if "</think>" in txt: txt = txt.split("</think>")[-1].strip()
            except Exception as e:
                txt = f"ERROR {e}"
            out.write(f"[{src}->{tgt}] {s}\n    => {txt}   ({time.time()-t:.1f}s)\n"); out.flush()
    srv.terminate(); srv.wait(); os.remove(path)
out.close()
