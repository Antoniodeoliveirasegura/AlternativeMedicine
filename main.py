"""
Drug alternatives lookup using RxNav API
Helps find generic/brand equivalents for medications
"""
import argparse
import asyncio
import re
import sys
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, Query
from pydantic import BaseModel

app = FastAPI(title="Drug Alternatives API")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this to your domain in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RXNAV_BASE = "https://rxnav.nlm.nih.gov/REST"
DESIRED_TTYS = {"SCD", "SBD", "GPCK", "BPCK"}  # generic/brand clinical dose forms + packs


def log_debug(msg: str) -> None:
    """Simple debug logging"""
    print(f"[DEBUG] {msg}")


# ------------------ Text helpers ------------------
def levenshtein(a: str, b: str) -> int:
    a, b = a or "", b or ""
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[n]


def sim(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    denom = max(len(a or ""), len(b or ""), 1)
    return 1.0 - (levenshtein((a or "").lower(), (b or "").lower()) / denom)


# Common misspellings I've noticed
SPELLING_FIXES = {
    "tylenal": "tylenol",
    "amoxycillin": "amoxicillin", 
    "liptor": "lipitor",
    "aderrall": "adderall",
    "metmorfin": "metformin",
    "prednizone": "prednisone",
}

STRENGTH_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(mcg|mg|g|ml)\b", re.I)
FILLER_WORDS_RE = re.compile(
    r"\b(tablet|tab|tabs|cap|caps|capsule|extended release|er|xr|sr|ir|oral|po|solution|suspension|cream|ointment|patch|injection|spray)\b",
    re.I,
)


def extract_base_name(text: str) -> str:
    """Remove strength and common filler words to get the base drug name."""
    if not text:
        return ""
    t = STRENGTH_RE.sub(" ", text)
    t = FILLER_WORDS_RE.sub(" ", t)
    t = re.sub(r"[^a-zA-Z0-9\- ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def parse_strength(text: str) -> Optional[Dict[str, str]]:
    m = STRENGTH_RE.search(text or "")
    if not m:
        return None
    return {"value": m.group(1), "unit": m.group(2).lower()}


# ------------------ Models ------------------
class NormalizedOut(BaseModel):
    status: str  # 'ok' | 'ambiguous' | 'not_found'
    query: str
    rxcui: Optional[str] = None
    strength_value: Optional[str] = None
    strength_unit: Optional[str] = None
    suggestions: Optional[List[str]] = None
    confidence: Optional[float] = None


class Concept(BaseModel):
    rxcui: str
    name: str
    tty: Optional[str] = None
    source: Optional[str] = "RxNav"


class AlternativesOut(BaseModel):
    query: str
    normalized: NormalizedOut
    variants: List[Concept] = []
    ndcs: List[str] = []
    note: Optional[str] = None


# ------------------ RxNav helpers ------------------
async def call_rxnav_api(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Make API call to RxNav service"""
    url = f"{RXNAV_BASE}/{path}"
    log_debug(f"API call: {url} {params or ''}")
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(url, params=params)
        log_debug(f"Response: {response.status_code}")
        response.raise_for_status()
        return response.json()


async def get_spelling_suggestions(name: str) -> List[str]:
    try:
        data = await call_rxnav_api("spellingsuggestions.json", {"name": name})
        suggestions = data.get("suggestionGroup", {}).get("suggestionList", {}).get("suggestion", [])
        return suggestions or []
    except Exception as e:
        log_debug(f"Spell check failed: {e}")
        return []


async def find_approximate_matches(term: str, max_entries: int = 3) -> List[Dict[str, Any]]:
    try:
        data = await call_rxnav_api("approximateTerm.json", {"term": term, "maxEntries": max_entries})
        candidates = data.get("approximateGroup", {}).get("candidate", [])
        return candidates or []
    except Exception as e:
        log_debug(f"Approximate search failed: {e}")
        return []


async def get_drug_variants(rxcui: str) -> List[Dict[str, Any]]:
    """
    Find related drug forms (generic/brand variants)
    Tries comprehensive lookup first, falls back to basic if needed
    """
    # Try comprehensive related lookup first
    try:
        data = await call_rxnav_api(f"rxcui/{rxcui}/allrelated.json")
        groups = data.get("allRelatedGroup", {}).get("conceptGroup", []) or []
        results: List[Dict[str, Any]] = []
        for group in groups:
            tty = group.get("tty")
            if tty and tty.upper() in DESIRED_TTYS:
                for concept in group.get("conceptProperties", []) or []:
                    results.append({"rxcui": concept["rxcui"], "name": concept["name"], "tty": tty})
        if results:
            return results
    except Exception as e:
        log_debug(f"Comprehensive lookup failed: {e}")

    # Fallback to basic related lookup
    try:
        data = await call_rxnav_api(f"rxcui/{rxcui}/related.json")
        groups = data.get("relatedGroup", {}).get("conceptGroup", []) or []
        results = []
        for group in groups:
            tty = (group.get("tty") or "").upper()
            if tty in DESIRED_TTYS:
                for concept in group.get("conceptProperties", []) or []:
                    results.append({"rxcui": concept["rxcui"], "name": concept["name"], "tty": tty})
        return results
    except Exception as e:
        log_debug(f"Basic lookup also failed: {e}")
        return []


async def get_ndc_codes(rxcui: str) -> List[str]:
    """Get NDC codes for a given RxCUI"""
    try:
        data = await call_rxnav_api(f"rxcui/{rxcui}/ndcs.json")
        ndcs = data.get("ndcGroup", {}).get("ndcList", {}).get("ndc", [])
        return ndcs or []
    except Exception as e:
        log_debug(f"NDC lookup failed: {e}")
        return []


def _matches_strength(name: str, value: Optional[str], unit: Optional[str]) -> bool:
    if not (value and unit):
        return True
    pattern = re.compile(rf"\b{re.escape(str(value))}\s*{re.escape(unit)}\b", re.I)
    return bool(pattern.search(name or ""))


# ------------------ Core normalization ------------------
async def normalize_raw(raw: str) -> NormalizedOut:
    raw = (raw or "").strip()
    if not raw:
        return NormalizedOut(status="not_found", query=raw)

    strength = parse_strength(raw)
    cleaned = re.sub(r"\s+", " ", raw)
    base = extract_base_name(cleaned)
    log_debug(f"Processing: '{cleaned}', extracted base: '{base}'")

    # Check for common misspellings
    hint_full = SPELLING_FIXES.get(cleaned.lower())
    hint_base = SPELLING_FIXES.get(base.lower()) if base else None

    # Get spelling suggestions
    sugs_full = await get_spelling_suggestions(cleaned)
    sugs_base = await get_spelling_suggestions(base) if base and base.lower() != cleaned.lower() else []

    # Build candidate pool (ordered, unique)
    candidates_text: List[str] = [cleaned]
    if base and base.lower() != cleaned.lower():
        candidates_text.append(base)
    for h in (hint_full, hint_base):
        if h:
            candidates_text.append(h)
    candidates_text.extend((sugs_base or [])[:5])
    candidates_text.extend((sugs_full or [])[:5])

    seen, deduped = set(), []
    for t in candidates_text:
        if t and t.lower() not in seen:
            deduped.append(t)
            seen.add(t.lower())
    candidates_text = deduped
    log_debug(f"Search candidates: {candidates_text}")

    # Search RxNav for each candidate
    all_cands: List[Dict[str, Any]] = []
    for t in candidates_text:
        approx = await find_approximate_matches(t, max_entries=3)
        for c in approx:
            score = float(c.get("score", 0))
            all_cands.append(
                {
                    "input": t,
                    "rxcui": c.get("rxcui"),
                    "score": score,
                    "sim": sim(cleaned, t),
                }
            )

    if not all_cands:
        log_debug("No matches found in RxNav")
        return NormalizedOut(
            status="not_found",
            query=raw,
            strength_value=strength["value"] if strength else None,
            strength_unit=strength["unit"] if strength else None,
            suggestions=(sugs_base or [])[:5] or (sugs_full or [])[:5] or None,
        )

    # Score candidates - favor text similarity over RxNav score
    for c in all_cands:
        c["conf"] = 0.35 * (c["score"] / 100.0) + 0.65 * c["sim"]

    best = max(all_cands, key=lambda x: x["conf"])
    log_debug(f"Top match: {best}")

    # Be more lenient if we have good text match + strength info
    confidence_threshold = 0.70
    if strength and best.get("sim", 0) >= 0.90:
        confidence_threshold = 0.60

    status = "ok" if best["conf"] >= confidence_threshold else "ambiguous"

    return NormalizedOut(
        status=status,
        query=raw,
        rxcui=best["rxcui"] if status == "ok" else None,
        strength_value=strength["value"] if strength else None,
        strength_unit=strength["unit"] if strength else None,
        suggestions=((sugs_base or [])[:5] or (sugs_full or [])[:5]) if status != "ok" else None,
        confidence=round(best["conf"], 3),
    )


# ------------------ HTTP endpoints ------------------
@app.get("/normalize", response_model=NormalizedOut)
async def normalize(q: str = Query(..., description="Drug (e.g., 'liptor 20 mg')")) -> NormalizedOut:
    # Basic input validation
    if not q or len(q.strip()) > 200:
        return NormalizedOut(status="not_found", query=q or "")
    
    # Sanitize input - remove potentially harmful characters
    sanitized = re.sub(r'[<>"\';\\]', '', q.strip())
    return await normalize_raw(sanitized)


@app.get("/alternatives", response_model=AlternativesOut)
async def alternatives(q: str = Query(..., description="Drug (free text, e.g., 'liptor 20 mg')")) -> AlternativesOut:
    # Basic input validation
    if not q or len(q.strip()) > 200:
        return AlternativesOut(
            query=q or "", 
            normalized=NormalizedOut(status="not_found", query=q or ""),
            variants=[], 
            ndcs=[], 
            note="Invalid input"
        )
    
    # Sanitize input - remove potentially harmful characters
    sanitized = re.sub(r'[<>"\';\\]', '', q.strip())
    norm = await normalize_raw(sanitized)
    if norm.status != "ok" or not norm.rxcui:
        note = "Ambiguous or not found."
        return AlternativesOut(query=sanitized, normalized=norm, variants=[], ndcs=[], note=note)

    variants = await get_drug_variants(norm.rxcui)
    if norm.strength_value and norm.strength_unit:
        variants = [
            v
            for v in variants
            if _matches_strength(v.get("name", ""), norm.strength_value, norm.strength_unit)
        ]

    ndcs = await get_ndc_codes(norm.rxcui)
    concepts = [Concept(**v) for v in variants]
    return AlternativesOut(
        query=q,
        normalized=norm,
        variants=concepts,
        ndcs=ndcs,
        note="Data from RxNav/NIH. For informational purposes only - always check with your doctor.",
    )


@app.get("/")
def root() -> Dict[str, Any]:
    return {"status": "ok", "endpoints": ["/normalize", "/alternatives"]}


# ------------------ CLI helpers ------------------
async def cli_run_query(q: str, show_ndcs: bool = True, limit: int = 25) -> int:
    print(f"\n[QUERY] {q}")
    norm = await normalize_raw(q)

    if norm.status == "not_found":
        print("No match found. Try another spelling.")
        return 2

    if norm.status == "ambiguous":
        print("Input ambiguous.")
        if norm.suggestions:
            print("Did you mean:")
            for s in norm.suggestions:
                print(f"  - {s}")
        return 3

    print(f"Normalized: RXCUI {norm.rxcui} (confidence={norm.confidence})")
    if norm.strength_value and norm.strength_unit:
        print(f"Strength parsed: {norm.strength_value} {norm.strength_unit}")

    variants = await get_drug_variants(norm.rxcui)
    if norm.strength_value and norm.strength_unit:
        variants = [
            v
            for v in variants
            if _matches_strength(v.get("name", ""), norm.strength_value, norm.strength_unit)
        ]

    if not variants:
        print("No brand/generic variants found with that strength; showing all related:")
        variants = await get_drug_variants(norm.rxcui)

    print("\nAlternatives (brand/generic variants):")
    for i, v in enumerate(variants[:limit], 1):
        tty = v.get("tty", "")
        print(f"{i:>2}. [{tty}] {v.get('name')} (rxcui={v.get('rxcui')})")

    if show_ndcs:
        ndcs = await get_ndc_codes(norm.rxcui)
        if ndcs:
            print(f"\nNDCs for normalized concept (first 10): {', '.join(ndcs[:10])}")

    print("\n(Info only; consult your prescriber before switching medicines.)\n")
    return 0


async def cli_loop() -> None:
    print("Drug Alternatives – Terminal Mode")
    print("Type a drug (e.g., 'liptor 20 mg'). Type 'quit' to exit.\n")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return
        if not q:
            continue
        if q.lower() in {"q", "quit", "exit"}:
            print("Bye.")
            return
        try:
            await cli_run_query(q)
        except Exception as e:
            print(f"[ERROR] {e}")


# ------------------ Entry point ------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Drug Alternatives (API + CLI)")
    parser.add_argument("--cli", action="store_true", help="Run interactive terminal mode")
    parser.add_argument("--once", type=str, help="Run a single query and exit (e.g., --once 'liptor 20 mg')")
    parser.add_argument("--no-ndcs", action="store_true", help="Do not print NDCs in CLI output")
    parser.add_argument("--limit", type=int, default=25, help="Max alternatives to print (default 25)")
    args = parser.parse_args()

    if args.cli or args.once:
        if sys.platform.startswith("win"):
            try:
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            except Exception:
                pass

        if args.once:
            code = asyncio.run(
                cli_run_query(args.once, show_ndcs=not args.no_ndcs, limit=args.limit)
            )
            raise SystemExit(code)
        else:
            asyncio.run(cli_loop())
    else:
        print("API mode. Start the server with:\n  uvicorn main:app --reload --port 8000")


if __name__ == "__main__":
    main()
