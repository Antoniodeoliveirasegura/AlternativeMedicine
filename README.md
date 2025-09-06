# Drug Alternatives API

A FastAPI service that helps find generic and brand alternatives for medications using the RxNav API from the National Library of Medicine.

## What it does

Ever wondered if there's a generic version of your prescription? Or what the brand name is for that generic drug? This tool helps you find medication alternatives by:

- **Smart drug name matching** - Handles common misspellings and variations
- **Strength-aware searching** - Finds alternatives with the same dosage
- **Brand/generic lookup** - Shows both branded and generic equivalents
- **NDC code retrieval** - Gets National Drug Codes for pharmacy systems

## Quick Start

### API Mode (recommended)
```bash
# Install dependencies
pip install fastapi httpx uvicorn

# Start the server
uvicorn main:app --reload --port 8000
```

Then visit `http://localhost:8000/docs` for the interactive API documentation.

### CLI Mode
```bash
# Interactive mode
python main.py --cli

# Single query
python main.py --once "lipitor 20mg"
```

## API Endpoints

### GET /normalize
Normalizes a drug name and finds the best RxNav match.

**Example:**
```bash
curl "http://localhost:8000/normalize?q=liptor%2020mg"
```

**Response:**
```json
{
  "status": "ok",
  "query": "liptor 20mg",
  "rxcui": "617318",
  "strength_value": "20",
  "strength_unit": "mg",
  "confidence": 0.695
}
```

### GET /alternatives
Gets brand/generic alternatives for a medication.

**Example:**
```bash
curl "http://localhost:8000/alternatives?q=liptor%2020mg"
```

**Response:**
```json
{
  "query": "liptor 20mg",
  "normalized": { ... },
  "variants": [
    {
      "rxcui": "617318",
      "name": "atorvastatin 20 MG Oral Tablet [Lipitor]",
      "tty": "SBD"
    },
    {
      "rxcui": "617310", 
      "name": "atorvastatin 20 MG Oral Tablet",
      "tty": "SCD"
    }
  ],
  "ndcs": ["00071015623", "00071015640", ...],
  "note": "Data from RxNav/NIH. For informational purposes only - always check with your doctor."
}
```

## Features

### Fuzzy Matching
Handles common misspellings:
- "liptor" → "lipitor"
- "tylenal" → "tylenol" 
- "metmorfin" → "metformin"

### Strength Parsing
Automatically extracts dosage information:
- "lipitor 20mg" → 20 mg
- "tylenol 500 mg" → 500 mg
- "amoxicillin 250mg caps" → 250 mg

### Smart Confidence Scoring
Uses a blend of RxNav similarity scores and text matching to determine the best match. More lenient thresholds when strength information is available.

## CLI Examples

```bash
# Interactive mode
python main.py --cli
> lipitor 20mg
> tylenol 500mg
> quit

# Single queries
python main.py --once "advil 200mg"
python main.py --once "generic prozac" --no-ndcs
python main.py --once "metformin" --limit 10
```

## Development

The code is structured with clear separation:
- **Text processing** - Handles name normalization and strength extraction
- **RxNav integration** - API calls with fallback mechanisms  
- **Confidence scoring** - Weighted similarity matching
- **FastAPI endpoints** - Clean REST interface
- **CLI interface** - Terminal-friendly interaction

### Key Functions
- `normalize_raw()` - Core drug name normalization
- `get_drug_variants()` - Find brand/generic alternatives
- `get_spelling_suggestions()` - RxNav spell checking
- `find_approximate_matches()` - Fuzzy name matching

## Data Source

All drug data comes from [RxNav](https://rxnav.nlm.nih.gov/), the National Library of Medicine's drug information service. This includes:
- Drug names and identifiers (RxCUI)
- Brand/generic relationships
- NDC (National Drug Code) mappings
- Spelling suggestions

## Important Notes

⚠️ **This tool is for informational purposes only**

- Always consult your healthcare provider before switching medications
- Generic equivalents may have different inactive ingredients
- Dosing and administration may vary between formulations
- Not a substitute for professional medical advice

## License

MIT License - feel free to use and modify as needed.