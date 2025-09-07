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
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ------------------ App setup ------------------
app = FastAPI(title="Drug Alternatives API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your domain in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RXNAV_BASE = "https://rxnav.nlm.nih.gov/REST"
DESIRED_TTYS = {"SCD", "SBD", "GPCK", "BPCK"}  # generic/brand clinical dose forms + packs
OPENFDA_NDC = "https://api.fda.gov/drug/ndc.json"


def debug(msg: str) -> None:
    """Simple debug logger."""
    print(f"[DEBUG] {msg}")


# ------------------ NDC helpers (openFDA + RxNav fallback) ------------------
def hyphenate_ndc(ndc: str) -> str:
    """
    Convert 11-digit NDC like '00071015623' -> '00071-0156-23' (5-4-2 split).
    If we get 10 digits, we can't know the split w/o metadata; we try
    the common layouts in fetch_openfda_ndc().
    """
    ndc = re.sub(r"\D", "", ndc or "")
    if len(ndc) == 11:  # standardized 11-digit
        return f"{ndc[:5]}-{ndc[5:9]}-{ndc[9:]}"
    return ndc  # leave 10-digit as-is; we'll try multiple layouts


async def fetch_openfda_ndc(package_or_product_ndc: str) -> Optional[Dict[str, Any]]:
    """
    Try to fetch a product by package_ndc first; if not found, try product_ndc.
    Accepts 11-digit (we'll hyphenate) or 10-digit (we'll try common splits).
    Returns the first matching result (dict) or None.
    """
    ndc_raw = re.sub(r"\D", "", package_or_product_ndc or "")
    if not ndc_raw:
        return None

    async with httpx.AsyncClient(timeout=20) as cx:
        # 1) If 11-digit, use 5-4-2 hyphenation and try package_ndc directly
        if len(ndc_raw) == 11:
            pkg = hyphenate_ndc(ndc_raw)
            for field in ("package_ndc", "product_ndc"):
                try:
                    url = f'{OPENFDA_NDC}?search={field}:"{pkg}"&limit=1'
                    debug(f"openFDA lookup: {url}")
                    r = await cx.get(url)
                    if r.status_code == 200:
                        js = r.json()
                        results = js.get("results", [])
                        if results:
                            return results[0]
                except Exception as e:
                    debug(f"openFDA {field} lookup failed: {e}")
            return None

        # 2) If 10-digit, try the 3 common layouts for both fields
        layouts = [(4, 4, 2), (5, 3, 2), (5, 4, 1)]
        for a, b, c in layouts:
            if len(ndc_raw) != (a + b + c):
                continue
            h = f"{ndc_raw[:a]}-{ndc_raw[a:a+b]}-{ndc_raw[a+b:]}"
            for field in ("package_ndc", "product_ndc"):
                try:
                    url = f'{OPENFDA_NDC}?search={field}:"{h}"&limit=1'
                    debug(f"openFDA lookup: {url}")
                    r = await cx.get(url)
                    if r.status_code == 200:
                        js = r.json()
                        results = js.get("results", [])
                        if results:
                            return results[0]
                except Exception as e:
                    debug(f"openFDA {field} lookup failed: {e}")

        return None


async def rxnav_ndc_properties(ndc: str) -> Optional[Dict[str, Any]]:
    """Use RxNav to get labeler/product/package info for an NDC."""
    ndc_clean = re.sub(r"\D", "", ndc or "")
    if not ndc_clean:
        return None

    attempts = [ndc_clean]
    # try 11-digit hyphenation
    if len(ndc_clean) == 11:
        attempts.append(hyphenate_ndc(ndc_clean))
    # try common hyphenations for 10-digit
    if len(ndc_clean) == 10:
        for a, b, c in [(4, 4, 2), (5, 3, 2), (5, 4, 1)]:
            if len(ndc_clean) == a + b + c:
                attempts.append(f"{ndc_clean[:a]}-{ndc_clean[a:a+b]}-{ndc_clean[a+b:]}")

    for candidate in attempts:
        try:
            data = await call_rxnav_api("ndcproperties.json", {"ndc": candidate})
            props = data.get("ndcPropertyList", {}).get("ndcProperty", [])
            if props:
                return props[0]  # first hit is fine
        except Exception as e:
            debug(f"RxNav ndcproperties failed for {candidate}: {e}")
    return None


def summarize_openfda_product(p: Dict[str, Any]) -> Dict[str, Any]:
    """Return a compact, friendly summary from an openFDA NDC result row."""
    packaging = p.get("packaging", []) or []
    pack_desc = packaging[0].get("description") if packaging else None
    return {
        "labeler_name": p.get("labeler_name"),
        "brand_name": p.get("brand_name"),
        "generic_name": p.get("generic_name"),
        "dosage_form": p.get("dosage_form"),
        "route": p.get("route"),
        "product_ndc": p.get("product_ndc"),
        "package_example": pack_desc,
        "marketing_start_date": p.get("marketing_start_date"),
        "marketing_end_date": p.get("marketing_end_date"),
        "active_ingredients": p.get("active_ingredients"),
    }


def summarize_rxnav_ndc(props: Dict[str, Any]) -> Dict[str, Any]:
    """Compact summary from RxNav ndcproperties."""
    return {
        "ndc": props.get("ndc"),
        "name": props.get("name"),
        "labeler": props.get("labeler"),
        "startDate": props.get("startDate"),
        "endDate": props.get("endDate"),
        "rxcui": props.get("rxcui"),
        "tty": props.get("tty"),
        "status": props.get("status"),
    }


@app.get("/ndcinfo")
async def ndcinfo(ndc: str) -> Dict[str, Any]:
    """
    Look up one NDC (11- or 10-digit or hyphenated).
    Returns a friendly summary from openFDA, else from RxNav.
    """
    ndc_clean = re.sub(r"\D", "", ndc or "")
    if not ndc_clean:
        return {"input": ndc, "error": "Invalid NDC"}

    # Try openFDA first
    row = await fetch_openfda_ndc(ndc_clean)
    if row:
        return {"input": ndc, "source": "openFDA", "summary": summarize_openfda_product(row), "raw": row}

    # Fallback to RxNav ndcproperties
    props = await rxnav_ndc_properties(ndc_clean)
    if props:
        return {"input": ndc, "source": "RxNav", "summary": summarize_rxnav_ndc(props), "raw": props}

    return {"input": ndc, "error": "No match found in openFDA or RxNav"}


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
    """Make API call to RxNav service."""
    url = f"{RXNAV_BASE}/{path}"
    debug(f"API call: {url} {params or ''}")
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(url, params=params)
        debug(f"Response: {response.status_code}")
        response.raise_for_status()
        return response.json()


async def get_spelling_suggestions(name: str) -> List[str]:
    try:
        data = await call_rxnav_api("spellingsuggestions.json", {"name": name})
        suggestions = data.get("suggestionGroup", {}).get("suggestionList", {}).get("suggestion", [])
        return suggestions or []
    except Exception as e:
        debug(f"Spell check failed: {e}")
        return []


async def find_approximate_matches(term: str, max_entries: int = 3) -> List[Dict[str, Any]]:
    try:
        data = await call_rxnav_api("approximateTerm.json", {"term": term, "maxEntries": max_entries})
        candidates = data.get("approximateGroup", {}).get("candidate", [])
        return candidates or []
    except Exception as e:
        debug(f"Approximate search failed: {e}")
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
        debug(f"Comprehensive lookup failed: {e}")

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
        debug(f"Basic lookup also failed: {e}")
        return []


async def get_ndc_codes(rxcui: str) -> List[str]:
    """Get NDC codes for a given RxCUI."""
    try:
        data = await call_rxnav_api(f"rxcui/{rxcui}/ndcs.json")
        ndcs = data.get("ndcGroup", {}).get("ndcList", {}).get("ndc", [])
        return ndcs or []
    except Exception as e:
        debug(f"NDC lookup failed: {e}")
        return []


def _matches_strength(name: str, value: Optional[str], unit: Optional[str]) -> bool:
    if not (value and unit):
        return True
    pattern = re.compile(rf"\b{re.escape(str(value))}\s*{re.escape(unit)}\b", re.I)
    return bool(pattern.search(name or ""))


async def rxnav_name_to_rxcui(name: str) -> Optional[str]:
    """Direct name -> RXCUI fallback using RxNav."""
    try:
        data = await call_rxnav_api("rxcui.json", {"name": name, "search": "1"})
        ids = data.get("idGroup", {}).get("rxnormId", [])
        return ids[0] if ids else None
    except Exception as e:
        debug(f"rxcui name lookup failed: {e}")
        return None


# ------------------ Core normalization ------------------
async def normalize_raw(raw: str) -> NormalizedOut:
    raw = (raw or "").strip()
    if not raw:
        return NormalizedOut(status="not_found", query=raw)

    strength = parse_strength(raw)
    cleaned = re.sub(r"\s+", " ", raw)
    base = extract_base_name(cleaned)
    debug(f"Processing: '{cleaned}', extracted base: '{base}'")

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
    debug(f"Search candidates: {candidates_text}")

    # Search RxNav for each candidate
    all_cands: List[Dict[str, Any]] = []
    for t in candidates_text:
        approx = await find_approximate_matches(t, max_entries=3)
        for c in approx:
            score = float(c.get("score", 0))
            all_cands.append({"input": t, "rxcui": c.get("rxcui"), "score": score, "sim": sim(cleaned, t)})

    if not all_cands:
        debug("No matches found in RxNav")
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
    debug(f"Top match: {best}")

    # Be more lenient if we have good text match + strength info
    confidence_threshold = 0.70
    if best.get("sim", 0) >= 0.98:
        confidence_threshold = 0.65
    if strength and best.get("sim", 0) >= 0.90:
        confidence_threshold = 0.60

    status = "ok" if best["conf"] >= confidence_threshold else "ambiguous"

    # Last-resort: direct name -> rxcui lookup when ambiguous
    if status != "ok":
        direct = await rxnav_name_to_rxcui(base or cleaned)
        if direct:
            return NormalizedOut(
                status="ok",
                query=raw,
                rxcui=direct,
                strength_value=strength["value"] if strength else None,
                strength_unit=strength["unit"] if strength else None,
                suggestions=None,
                confidence=round(max(best.get("conf", 0), 0.66), 3),
            )

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
    sanitized = re.sub(r'[<>"\';\\]', "", q.strip())
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
            note="Invalid input",
        )

    # Sanitize input - remove potentially harmful characters
    sanitized = re.sub(r'[<>"\';\\]', "", q.strip())
    norm = await normalize_raw(sanitized)
    if norm.status != "ok" or not norm.rxcui:
        note = "Ambiguous or not found."
        return AlternativesOut(query=sanitized, normalized=norm, variants=[], ndcs=[], note=note)

    variants = await get_drug_variants(norm.rxcui)
    if norm.strength_value and norm.strength_unit:
        variants = [v for v in variants if _matches_strength(v.get("name", ""), norm.strength_value, norm.strength_unit)]

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
    return {"status": "ok", "endpoints": ["/normalize", "/alternatives", "/ndcinfo"]}


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
        variants = [v for v in variants if _matches_strength(v.get("name", ""), norm.strength_value, norm.strength_unit)]

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
    parser.add_argument("--ndc", type=str, help="Look up one NDC (e.g., --ndc 00071015623)")
    args = parser.parse_args()

    # One-off NDC decode path
    if args.ndc:
        row = asyncio.run(fetch_openfda_ndc(args.ndc))
        if row:
            from pprint import pprint

            print("Source: openFDA")
            print("Summary:")
            pprint(summarize_openfda_product(row))
            raise SystemExit(0)

        props = asyncio.run(rxnav_ndc_properties(args.ndc))
        if props:
            from pprint import pprint

            print("Source: RxNav")
            print("Summary:")
            pprint(summarize_rxnav_ndc(props))
            raise SystemExit(0)

        print(f"No match for NDC: {args.ndc}")
        raise SystemExit(2)

    if args.cli or args.once:
        if sys.platform.startswith("win"):
            try:
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            except Exception:
                pass

        if args.once:
            code = asyncio.run(cli_run_query(args.once, show_ndcs=not args.no_ndcs, limit=args.limit))
            raise SystemExit(code)
        else:
            asyncio.run(cli_loop())
    else:
        print("API mode. Start the server with:\n  uvicorn main:app --reload --port 8000")


if __name__ == "__main__":
    main()