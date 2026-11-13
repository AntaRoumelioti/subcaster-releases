"""Greek homophone spelling from context (σχόλιο/σχολείο…): raw answers of the local model. Writes hom.txt."""
import json, os, sys, time, subprocess, urllib.request, glob
from huggingface_hub import HfApi, hf_hub_download
PROMPT = sys.argv[2]
LINES = ["Γράψτε μου στα σχολεία από κάτω τι θέλετε να δούμε.", "Καλή τείχη σε όλους στο σημερινό παιχνίδι!",
         "Τα παιδιά πήγαν στο σχόλιο το πρωί.", "Ο δράκος κάθεται πάνω στα τύχη του κάστρου.",
         "Νιώθω μεγάλη λίπη που φεύγει.", "Πότε δεν έχω δει τέτοιο πράγμα.", "Κλείνω το βίντεο εδώ, τα λέμε.",
         "Η μάγισσα είπε ότι θα έρθει αύριο.", "Ρίχνω τα ζάρια και φέρνω έξι.", "Μου λύπει πολύ ο φίλος μου από τότε που έφυγε.", "Θα κλίσω την πόρτα να μην μπει ο λύκος.", "Θα σας πω μια ιστορία για έναν ήρωα που έχασε τη φωνή του."]
repo = sys.argv[1]
out = open("hom.txt", "w", encoding="utf-8")
files = sorted(f for f in HfApi().list_repo_files(repo) if f.endswith(".gguf") and "Q4_K_M" in f and "mmproj" not in f)
path = hf_hub_download(repo, files[0])
out.write(f"===== {repo} / {files[0]}\n"); out.flush()
srv = subprocess.Popen([glob.glob("llama/**/llama-server", recursive=True)[0], "-m", path, "--jinja", "-c", "4096", "--port", "8093", "-t", "4"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for _ in range(300):
    try: urllib.request.urlopen("http://127.0.0.1:8093/health", timeout=2).read(); break
    except Exception: time.sleep(1)
for batch in (LINES[:8], LINES[8:]):
    p = PROMPT + "\n\n" + "\n".join(f"{i+1}. {t}" for i, t in enumerate(batch))
    body = json.dumps({"messages": [{"role": "user", "content": p}], "temperature": 0, "max_tokens": 2000, "seed": 7}).encode()
    t = time.time()
    res = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8093/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=1800))
    out.write("---- in\n" + "\n".join(batch) + "\n---- out (%.0fs)\n" % (time.time() - t) + res["choices"][0]["message"]["content"] + "\n"); out.flush()
srv.terminate(); out.close()
