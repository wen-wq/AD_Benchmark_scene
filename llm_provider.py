"""Generate or validate five emergency specs with a session file or any OpenAI-compatible API.

Examples: python llm_provider.py --provider session
          LLM_API_KEY=... python llm_provider.py --provider openai-compatible \
            --base-url https://.../v1 --model MODEL
          python llm_provider.py --provider openai-compatible \
            --base-url http://localhost:11434/v1 --model qwen2.5:14b
"""
import argparse
import json
import os
import urllib.error
import urllib.request

from pku_pipeline import ROOT, route_model, save_json, validate_events

PROMPT = """Create five synthetic, physically plausible autonomous-driving emergencies on a 970 m public-road route outside PKU East Gate, south on Zhongguancun North Street and west on Haidian Road to outside South Gate. Return only a JSON object with schema_version=2, author, intent, events. Events have IDs E1..E5, sorted distance_m 100..880, spacing at least 130 m. Include one lead_vehicle_hard_brake at about 140 m and four cut-ins from left_cutin/right_cutin. E2 near 300 m is left, E3 near 465 m is right, E4 near 720 m is right with subtype=curbside_pullout, and E5 near 870 m is a slower right-side merge. Each has reason, ego_reaction_s 0.35..0.65 and ego_brake_duration_s 1.2..1.6. A lead brake additionally has lead_decel_mps2 -6.2..-5 and lead_brake_duration_s 1.2..1.6. A cut-in additionally has initial_gap_m 22..30, actor_speed_mps 5..9, cutin_duration_s 2.3..3.3. For a curbside pullout add yield_speed_mps 3..5 and yield_duration_s 8..10. For E5 add hold_speed_mps 5..6, hold_duration_s 4..6, pre_event_visibility_s 2..3 to enter through a clear curbside gap. Do not claim invented events happened in reality."""


def call_api(base_url,model):
    url=base_url.rstrip("/")+"/chat/completions"
    payload={"model":model,"temperature":.3,"response_format":{"type":"json_object"},
             "messages":[{"role":"system","content":"Return valid JSON only."},
                         {"role":"user","content":PROMPT}]}
    headers={"Content-Type":"application/json"}
    if os.environ.get("LLM_API_KEY"):
        headers["Authorization"]="Bearer "+os.environ["LLM_API_KEY"]
    try:
        req=urllib.request.Request(url,data=json.dumps(payload).encode(),headers=headers,method="POST")
        with urllib.request.urlopen(req,timeout=90) as r:out=json.load(r)
    except urllib.error.HTTPError as exc:
        if exc.code not in (400,422):raise
        payload.pop("response_format")  # Some compatible local servers omit this feature.
        req=urllib.request.Request(url,data=json.dumps(payload).encode(),headers=headers,method="POST")
        with urllib.request.urlopen(req,timeout=90) as r:out=json.load(r)
    content=out["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        content=content.split("\n",1)[1].rsplit("```",1)[0].strip()
    return json.loads(content)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--provider",choices=["session","openai-compatible"],default="session")
    p.add_argument("--base-url")
    p.add_argument("--model")
    a=p.parse_args()
    if a.provider=="session":
        spec=json.load(open(ROOT/"llm_events.json",encoding="utf-8"))
    else:
        if not a.base_url or not a.model:raise SystemExit("--base-url and --model required")
        spec=call_api(a.base_url,a.model)
        spec["author"]=f"{a.provider}:{a.model}"
    _,_,cum,_=route_model()
    validate_events(spec,cum[-1])
    save_json(ROOT/"llm_events.json",spec)
    print("Validated and saved exactly five events")


if __name__=="__main__":main()
