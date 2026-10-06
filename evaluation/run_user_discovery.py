"""Discovery enrollment on the user's real recording (Mix + Nicker/Mond/Sam).

Uses the thresholds tuned on the synthetic dev set, unchanged. There are no
clean references for this recording, so nothing here is an SI-SDR score:
discovered tracks are labelled by ECAPA similarity to the separate
Nicker/Mond/Sam recordings, and live outputs get a rough voice-similarity
check. Listening is the real test. Two variants:

* split: discover on the first half, extract from the second half (the
  product flow; live audio was never heard during discovery);
* full: discover on and extract from the whole clip (more speech for
  discovery, but the live audio overlaps the discovery audio).

Outputs stay in data/user_recordings/discovery/, which Git ignores.
"""

import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf

import discovery_pipeline as dp
from run_user_recordings import PROFILES, read_mono, resample, sha256

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "data/user_recordings"
OUT = INPUTS / "discovery"
CONFIG = ROOT / "data/discovery/results/config_and_tuning.json"
TEMPLATE = ROOT / "discovery_showcase_template.html"
MODELS = ("speakerbeam_ss", "sepformer_selector", "sepformer_competitive")


def load_16k(path):
    wave, rate = read_mono(path)
    return resample(wave, rate, dp.RATE)


def similarity(models, wave, enrolled):
    vector = models.embed("ecapa", wave)
    if vector is None:
        return None
    return {name: float(np.dot(vector, e)) for name, e in enrolled.items()}


def best_match(scores):
    ranked = sorted(scores.items(), key=lambda item: -item[1])
    return {"name": ranked[0][0], "cosine": ranked[0][1], "margin": ranked[0][1] - ranked[1][1]}


def listening_copy(wave, relative):
    path = OUT / "listen" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(wave))) if len(wave) else 0.0
    sf.write(path, wave * (0.9 / peak if peak > 1e-9 else 1.0), dp.RATE, subtype="PCM_16")
    return f"listen/{relative}"


def run_variant(models, name, discovery_audio, live_audio, enrolled, params):
    found, _ = dp.discover(models, discovery_audio, "ecapa", *params["discovery"])
    clusters = []
    for index, speaker in enumerate(found, 1):
        scores = similarity(models, speaker["track"], enrolled["ecapa"])
        clusters.append({
            "cluster": index, "windows": len(speaker["windows"]), "similarity": scores,
            "match": best_match(scores) if scores else None,
            "ecapa": models.embed("ecapa", speaker["track"]),
            "resemblyzer": models.embed("resemblyzer", speaker["track"]),
            "audio": listening_copy(speaker["track"], f"{name}/discovered_{index}.wav")})
    separation = dp.separate_windows(models, live_audio)
    dp.embed_streams(models, separation, "ecapa")
    usable = [c for c in clusters if c["ecapa"] is not None]
    names = list(enrolled["ecapa"])
    runs = [{"key": f"cluster:{c['cluster']}", "enrollment": "discovered", "ecapa": c["ecapa"],
             "resemblyzer": c["resemblyzer"], "candidates": [u["ecapa"] for u in usable],
             "target_index": usable.index(c)} for c in usable]
    runs += [{"key": f"clean:{n}", "enrollment": "clean", "ecapa": enrolled["ecapa"][n],
              "resemblyzer": enrolled["resemblyzer"][n],
              "candidates": [enrolled["ecapa"][m] for m in names], "target_index": names.index(n)}
             for n in names]
    live = {}
    for run in runs:
        live[run["key"]] = {}
        for model in MODELS:
            if model == "speakerbeam_ss":
                output = dp.speakerbeam_extract(models, live_audio, run["resemblyzer"])
            elif model == "sepformer_selector":
                output = dp.select_streams(separation, run["ecapa"],
                                           params[model][run["enrollment"]])
            else:
                output = dp.select_streams_competitive(separation, run["candidates"],
                                                       run["target_index"],
                                                       params[model][run["enrollment"]])
            tag = run["key"].replace(":", "_")
            live[run["key"]][model] = {
                "similarity": similarity(models, output, enrolled["ecapa"]),
                "rms_db": float(10 * np.log10(max(np.mean(output.astype("float64") ** 2), 1e-20))),
                "audio": listening_copy(output, f"{name}/{model}__{tag}.wav")}
    for c in clusters:
        c.pop("ecapa"), c.pop("resemblyzer")
    return {"variant": name, "discovery_seconds": len(discovery_audio) / dp.RATE,
            "live_seconds": len(live_audio) / dp.RATE,
            "live_mix_similarity": similarity(models, live_audio, enrolled["ecapa"]),
            "discovery_mix": listening_copy(discovery_audio, f"{name}/discovery_mix.wav"),
            "live_mix": listening_copy(live_audio, f"{name}/live_mix.wav"),
            "clusters": clusters, "live": live}


PAGE_SCRIPT = r"""
const DATA = /*__DATA__*/null;
const MODEL_NAMES = {speakerbeam_ss: "SpeakerBeam-SS", sepformer_selector: "SepFormer selector", sepformer_competitive: "SepFormer competitive"};
const KEY = "user-discovery-ratings-v1";
let ratings = {}; try { ratings = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
const fmt = (v, d = 2) => v === null || v === undefined ? "–" : Number(v).toFixed(d);
const el = (tag, attrs = {}, html) => { const n = document.createElement(tag); Object.assign(n, attrs); if (html !== undefined) n.innerHTML = html; return n; };
function rate(key, label) {
  const w = el("div", {className: "rating"}, `<span>${label}</span>`);
  for (let v = 1; v <= 5; v++) { const b = el("button", {type: "button", textContent: v}); if (ratings[key] === v) b.classList.add("on");
    b.onclick = () => { ratings[key] = v; try { localStorage.setItem(KEY, JSON.stringify(ratings)); } catch (e) {} w.querySelectorAll("button").forEach(x => x.classList.toggle("on", Number(x.textContent) === v)); };
    w.appendChild(b); }
  return w;
}
function sims(s) { return s ? Object.entries(s).map(([n, v]) => `${n} ${fmt(v)}`).join(" · ") : "no speech found"; }
function card(title, src, truth, keys) {
  const c = el("div", {className: "panel"}, `<h3>${title}</h3>`);
  c.appendChild(el("audio", {controls: true, preload: "none", src}));
  if (truth) c.appendChild(el("div", {className: "truth"}, truth));
  (keys || []).forEach(([k, l]) => c.appendChild(rate(k, l)));
  return c;
}
function render(v) {
  const box = document.getElementById("test"); box.innerHTML = "";
  const s1 = el("div", {}, `<h2><span class="step">1</span>Listen</h2>`); const g1 = el("div", {className: "grid"});
  g1.appendChild(card(`Discovery buffer (${fmt(v.discovery_seconds, 1)} s)`, v.discovery_mix, ""));
  g1.appendChild(card(`Live input (${fmt(v.live_seconds, 1)} s)`, v.live_mix, `Whole-input voice similarity: ${sims(v.live_mix_similarity)}`));
  s1.appendChild(g1); box.appendChild(s1);
  const s2 = el("div", {}, `<h2><span class="step">2</span>Discovered speakers (${v.clusters.length})</h2><p class="lead">Who is each one? Rate how clean each sounds, then reveal the voice match against the Nicker, Mond and Sam recordings.</p>`);
  const g2 = el("div", {className: "grid"}); s2.appendChild(g2); box.appendChild(s2);
  const s3 = el("div", {}, `<h2><span class="step">3</span>Live extraction</h2><p class="lead">Choose a discovered speaker above.</p>`); box.appendChild(s3);
  v.clusters.forEach(c => {
    const m = c.match;
    const truth = m ? `Sounds most like <b>${m.name}</b> (cosine ${fmt(m.cosine)}, margin ${fmt(m.margin)} over the next person). All: ${sims(c.similarity)}. Seen in ${c.windows} windows.` : "No usable speech.";
    const cc = card(`Discovered speaker ${c.cluster}`, c.audio, truth, [[`${v.variant}|disc|${c.cluster}|who`, "One clean voice?"]]);
    if (v.live[`cluster:${c.cluster}`]) { const b = el("button", {type: "button", className: "primary", textContent: "Choose this speaker"});
      b.onclick = () => { box.querySelectorAll("button.chosen").forEach(x => x.classList.remove("chosen")); b.classList.add("chosen"); live(v, c, s3); s3.scrollIntoView({behavior: "smooth"}); }; cc.appendChild(b); }
    g2.appendChild(cc);
  });
}
function live(v, c, s3) {
  const who = c.match ? c.match.name : null;
  s3.innerHTML = `<h2><span class="step">3</span>Live extraction of discovered speaker ${c.cluster}</h2><p class="lead">Each model gets only the embedding of the track you picked. Rate clarity and how much of the others remains (5 = none). Revealed numbers are ECAPA similarity of the output to each person's recording, a rough proxy and not a quality score.</p>`;
  const g = el("div", {className: "grid"});
  for (const m of Object.keys(MODEL_NAMES)) { const o = v.live[`cluster:${c.cluster}`][m];
    g.appendChild(card(`${MODEL_NAMES[m]} · discovered`, o.audio, `Output similarity: ${sims(o.similarity)}`, [[`${v.variant}|live|${m}|cluster${c.cluster}|clarity`, "Clarity"], [`${v.variant}|live|${m}|cluster${c.cluster}|leakage`, "No leakage"]])); }
  s3.appendChild(g);
  if (who && v.live[`clean:${who}`]) {
    const cmp = el("div", {className: "truth"}, `<h3 style="margin-top:14px">Comparison: ${who} with the separate ${who} recording as enrollment</h3>`);
    const g2 = el("div", {className: "grid"});
    for (const m of Object.keys(MODEL_NAMES)) { const o = v.live[`clean:${who}`][m];
      g2.appendChild(card(`${MODEL_NAMES[m]} · clean recording`, o.audio, `Output similarity: ${sims(o.similarity)}`, [[`${v.variant}|live|${m}|clean-${who}|clarity`, "Clarity"], [`${v.variant}|live|${m}|clean-${who}|leakage`, "No leakage"]])); }
    cmp.appendChild(g2); s3.appendChild(cmp);
  }
}
const sel = document.getElementById("variant");
DATA.variants.forEach((v, i) => sel.appendChild(el("option", {value: i, textContent: v.variant === "split" ? "Product flow: discover on first half, extract from second half" : "Whole clip for both (live audio overlaps discovery)"})));
sel.onchange = () => render(DATA.variants[sel.value]); render(DATA.variants[0]);
document.getElementById("reveal").onchange = e => document.body.classList.toggle("reveal", e.target.checked);
document.getElementById("export").onclick = () => { const a = el("a", {href: URL.createObjectURL(new Blob([JSON.stringify({exported: new Date().toISOString(), ratings}, null, 2)], {type: "application/json"})), download: "user_discovery_ratings.json"}); document.body.appendChild(a); a.click(); a.remove(); };
"""


def write_page(record):
    style = re.search(r"<style>.*?</style>", TEMPLATE.read_text(), re.S).group(0)
    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Real Recording Discovery</title>{style}</head>
<body><main>
<h1>Discovery on your recording</h1>
<p class="lead">Your Mix.wav run through the discovery flow with the settings tuned on synthetic conversations, unchanged. There is no isolated track for each person, so nothing here is an SI-SDR score; your ears are the test.</p>
<div class="note">Clips are peak-normalized for listening, so compare voices, clarity and leakage, not loudness. SepFormer only hears 8 kHz audio. Nicker, Mond and Sam's separate recordings are used only to label discovered tracks and for the clean-enrollment comparison.</div>
<div class="panel" style="margin-top:16px"><div class="row">
<label for="variant">Run</label><select id="variant"></select>
<label class="row"><input type="checkbox" id="reveal"> Reveal voice matches and similarity</label>
<button id="export">Export my ratings (JSON)</button></div></div>
<div id="test"></div>
</main><script>{PAGE_SCRIPT.replace("/*__DATA__*/null", json.dumps(record))}</script></body></html>
"""
    (OUT / "index.html").write_text(body)


def main():
    config = json.loads(CONFIG.read_text())
    lt = config["live_tuning"]
    params = {"discovery": (config["discovery_tuning"]["ecapa"]["chosen"]["distance_threshold"],
                            config["discovery_tuning"]["ecapa"]["chosen"]["min_windows"]),
              **{m: {k: lt[m][k]["chosen"]["threshold"] for k in ("clean", "discovered")}
                 for m in ("sepformer_selector", "sepformer_competitive")}}
    mix_path = INPUTS / "mixture/Mix.wav"
    mix = load_16k(mix_path)
    models = dp.Models()
    enrolled = {"ecapa": {}, "resemblyzer": {}}
    for name, relative in PROFILES.items():
        wave = load_16k(INPUTS / relative)
        label = name.capitalize()
        enrolled["ecapa"][label] = models.embed("ecapa", wave)
        enrolled["resemblyzer"][label] = models.embed("resemblyzer", wave)
    half = len(mix) // 2
    variants = [run_variant(models, "split", mix[:half], mix[half:], enrolled, params),
                run_variant(models, "full", mix, mix, enrolled, params)]
    names = list(enrolled["ecapa"])
    record = {
        "scope": "User's real 48 kHz stereo recording, downmixed to 16 kHz mono; offline, noncausal; no clean references",
        "input_sha256": {str(p.relative_to(INPUTS)): sha256(p)
                         for p in [mix_path] + [INPUTS / r for r in PROFILES.values()]},
        "parameters": {k: v for k, v in params.items()},
        "enrollment_cross_similarity": {a: {b: float(np.dot(enrolled["ecapa"][a], enrolled["ecapa"][b]))
                                            for b in names} for a in names},
        "variants": variants,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    write_page(record)
    for v in variants:
        print(f"{v['variant']}: {len(v['clusters'])} discovered", flush=True)
        for c in v["clusters"]:
            print("  cluster", c["cluster"], "windows", c["windows"], "match", c["match"])
        for key, outputs in v["live"].items():
            print("  ", key, {m: {n: round(s, 2) for n, s in o["similarity"].items()} if o["similarity"] else None
                             for m, o in outputs.items()})
    print(f"wrote {OUT / 'index.html'}")


if __name__ == "__main__":
    main()
