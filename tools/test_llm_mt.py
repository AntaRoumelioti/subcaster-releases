"""TranslateGemma (GGUF, llama.cpp) on caption-like sentences with its exact prompt; writes results.txt.
Variant A = the official prompt, B = the official prompt + a style line (Korean 해요체, natural Greek)."""
import json, os, sys, time, subprocess, urllib.request, glob
from huggingface_hub import HfApi, hf_hub_download

SENT = {
    "el": ["Γεια σας φίλοι μου, καλώς ήρθατε στο κανάλι!", "Σήμερα θα ρίξουμε τα ζάρια και θα δούμε τι θα γίνει.",
           "Ο δράκος εμφανίστηκε και όλοι πάγωσαν.", "Ποιος θέλει να παίξει μαζί μας;", "Μην ξεχάσετε να κάνετε like και subscribe!",
           "Έλα ρε, πάλι ένα το έφερα;", "Η μάγισσα ρίχνει ένα ξόρκι φωτιάς στο τέρας."],
    "ko": ["안녕하세요 여러분, 오늘은 주사위를 굴려볼게요.", "드래곤이 나타났어요!", "구독과 좋아요 부탁드려요!", "진짜 대박이다, 이거 봐!"],
    "en": ["Roll for initiative, everyone!", "The dragon breathes fire and the party scatters."],
}
NAME = {"el": "Greek", "ko": "Korean", "en": "English"}
STYLE = {"ko": "Use polite informal Korean (해요체): end sentences with -아요/-어요/해요, never 합니다체 and never 반말.",
         "el": "Use natural, everyday spoken Greek.", "en": "Use natural, casual spoken English."}
def prompt(src, tgt, text, style):
    S, T = NAME[src], NAME[tgt]
    p = (f"You are a professional {S} ({src}) to {T} ({tgt}) translator. Your goal is to accurately convey the meaning and nuances of the original {S} text "
         f"while adhering to {T} grammar, vocabulary, and cultural sensitivities.\n")
    if style: p += STYLE[tgt] + "\n"
    p += f"Produce only the {T} translation, without any additional explanations or commentary. Please translate the following {S} text into {T}:\n\n\n{text}"
    return f"<start_of_turn>user\n{p}<end_of_turn>\n<start_of_turn>model\n"
PAIRS = [("el", "ko"), ("ko", "el"), ("el", "en"), ("en", "ko"), ("en", "el")]
out = open("results.txt", "w", encoding="utf-8")
api = HfApi()
for repo in sys.argv[1:]:
    files = [f for f in api.list_repo_files(repo) if f.endswith(".gguf") and "Q4_K_M" in f]
    path = hf_hub_download(repo, files[0])
    out.write(f"\n===== {repo} / {files[0]} ({os.path.getsize(path)//1048576} MB) =====\n")
    srv = subprocess.Popen([glob.glob("llama/**/llama-server", recursive=True)[0], "-m", path, "-c", "4096", "--port", "8091", "-t", "4"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(300):
        try: urllib.request.urlopen("http://127.0.0.1:8091/health", timeout=2).read(); break
        except Exception: time.sleep(1)
    for src, tgt in PAIRS:
        for s in SENT[src]:
            for style in (False, True):
                body = json.dumps({"prompt": prompt(src, tgt, s, style), "temperature": 0, "n_predict": 200, "stop": ["<end_of_turn>"]}).encode()
                t = time.time()
                try:
                    res = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8091/completion", body, {"Content-Type": "application/json"}), timeout=900))
                    txt = res["content"].strip()
                except Exception as e:
                    txt = f"ERROR {e}"
                out.write(f"[{src}->{tgt}{' B' if style else ' A'}] {s}\n    => {txt}   ({time.time()-t:.1f}s)\n"); out.flush()
    srv.terminate(); srv.wait(); os.remove(path)
out.close()
