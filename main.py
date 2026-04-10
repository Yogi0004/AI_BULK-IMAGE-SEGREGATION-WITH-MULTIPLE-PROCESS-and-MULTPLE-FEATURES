"""
Real Estate Scraper Dashboard - AI-POWERED API Integration
===========================================================
Fast loading + Ground truth comparison + AI-powered mapping + Database API posting
"""

import streamlit as st
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import pandas as pd
import subprocess
import threading
import sys
import time
import queue
import re 
import requests
from openai import OpenAI
from dotenv import load_dotenv
import os
import functools
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from datetime import datetime
import platform
import select
import gzip
from datetime import datetime

# ===== SHARED FILE FOR BOTH DASHBOARDS =====
SHARED_FILE = "shared_data.json"

def send_to_image_dashboard(data):

    with open(SHARED_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print("✅ Data sent to Image Dashboard")

print("🔥 STARTING AISCRAP DASHBOARD FROM:", __file__)
print("🔥 CURRENT WORKING DIR:", os.getcwd())

from groundtruth import calculate_similarity  # For Linux non-blocking I/O

# Load environment variables FIRST
# load_dotenv('/home/aiscrap.homes247.in/public_html/.env')
load_dotenv('E:\abijith\Aiscrap 17-01-26\.env')
print("OPENAI_API_KEY loaded:", bool(os.getenv("OPENAI_API_KEY")))
print("MODEL_NAME loaded:", os.getenv("MODEL_NAME"))
# ============================================================================
# CROSS-PLATFORM PATH CONFIGURATION
# ============================================================================

if platform.system() == 'Linux':
    BASE_DIR = Path('/home/aiscrap.homes247.in/public_html')
    OUTPUT_DIR = BASE_DIR / 'output'
    LOG_DIR = BASE_DIR / 'logs'
else:
    OUTPUT_DIR = Path('output')
    LOG_DIR = Path('logs')
    load_dotenv('.env')

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

def ensure_file_permissions(file_path):
    """Ensure files have proper permissions on Linux"""
    try:
        if platform.system() == 'Linux':
            os.chmod(file_path, 0o666)
    except Exception as e:
        print(f"Warning: Could not set permissions for {file_path}: {e}")

# ============================================================================
# STREAMLIT CONFIG
# ============================================================================

st.set_page_config(page_title="Real Estate Scraper", page_icon="🏢", layout="wide")

# ============================================================================
# CONFIG
# ============================================================================

MAJOR_CITIES = ['Bangalore', 'Mumbai', 'Pune', 'Delhi', 'Hyderabad', 'Chennai']

SCRAPERS = {
    'magicbricks': {'name': 'MagicBricks', 'file': 'magicbricks_scraper.py', 'output': '{city}magic_bricks_properties.json', 'icon': '🥇'},
    'housing': {'name': 'Housing.com', 'file': 'housing_scraper.py', 'output': 'housing_{city}_properties.json', 'icon': '🥈'},
    'nobroker': {'name': 'NoBroker', 'file': 'nobroker_scraper.py', 'output': 'nobroker_{city}_properties.json', 'icon': '🥉'},
    'proptiger': {'name': 'PropTiger', 'file': 'proptiger_scraper.py', 'output': 'proptiger_final_output.json', 'icon': '4️⃣'},
    'commonfloor': {'name': 'CommonFloor', 'file': 'commonfloor_scraper.py', 'output': '{city}_commonfloor_projects.json', 'icon': '5️⃣'},
    'roofandfloor': {'name': 'RoofAndFloor', 'file': 'roofandfloor_scraper.py', 'output': '{city}_roofandfloor_projects.json', 'icon': '6️⃣'},
    'homznspace': {'name': 'HomznSpace', 'file': 'homznspace_scraper.py', 'output': 'homznspace_properties.json', 'icon': '7️⃣'},
    'propsoch': {'name': 'Propsoch', 'file': 'propsoch_scraper.py', 'output': 'propsoch_enhanced.json', 'icon': '8️⃣'},
    'squareyards': {'name': 'SquareYards', 'file': 'squareyards_scraper.py', 'output': '{city}_squareyards.json', 'icon': '9️⃣'}
}

INFO_JSON = OUTPUT_DIR / 'info.json'
COMPARISON_LOG = LOG_DIR / 'comparison_log.json'
API_ENDPOINT = os.getenv('API')


OPENAI_KEY = os.getenv("OPENAI_API_KEY")

AI_ENABLED = False
client = None

if OPENAI_KEY:
    try:
        client = OpenAI(api_key=OPENAI_KEY)
        MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
        AI_ENABLED = True
        print("✅ OpenAI client initialized successfully")
    except Exception as e:
        print("❌ OpenAI initialization failed:")
        import traceback
        traceback.print_exc()
        AI_ENABLED = False
else:
    print("❌ OPENAI_API_KEY not found in environment")
    AI_ENABLED = False


# ... rest of your code continues ...

# ============================================================================
# LOOKUP TABLES
# ============================================================================

# ============================================================================
# LOOKUP TABLES - UPDATED WITH BETTER LOCALITY LOADING
# ============================================================================

def load_lookup_tables():
    lookups = {
        'cities': {}, 
        'property_types': {}, 
        'property_status': {}, 
        'bhk': {}, 
        'amenities': {}, 
        'localities': []
    }
    id_folder = Path('id')
    configs = [
        ('city.json', 'city', 'city_name', 'city_IDPK', 'cities'),
        ('propertytype.json', 'propertytype', 'propertyType_name', 'propertyType_IDPK', 'property_types'),
        ('propertystatus.json', 'propertystatus', 'propertyStatus_name', 'propertyStatus_IDPK', 'property_status'),
        ('bhk.json', 'bhk', 'bhk', 'bhk_IDPK', 'bhk'),
        ('amenities.json', 'amenities', 'amenities_name', 'amenities_IDPK', 'amenities')
    ]
    try:
        for filename, table_name, name_key, id_key, target in configs:
            file_path = id_folder / filename
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    for item in content:
                        if item.get('type') == 'table' and item.get('name') == table_name:
                            for row in item.get('data', []):
                                name = str(row.get(name_key, '')).strip().lower()
                                lookups[target][name] = row.get(id_key)
        
        # Load localities as list - IMPROVED LOADING
        locality_file = id_folder / 'locality.json'
        if locality_file.exists():
            with open(locality_file, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                
                # Check if it's wrapped in table structure
                if isinstance(raw_data, list):
                    for item in raw_data:
                        if item.get('type') == 'table' and item.get('name') == 'locality':
                            lookups['localities'] = item.get('data', [])
                            print(f"✅ Loaded {len(lookups['localities'])} localities from table structure")
                            break
                    else:
                        # If no table structure, assume it's direct data
                        lookups['localities'] = raw_data
                        print(f"✅ Loaded {len(lookups['localities'])} localities (direct array)")
                elif isinstance(raw_data, dict) and 'data' in raw_data:
                    lookups['localities'] = raw_data['data']
                    print(f"✅ Loaded {len(lookups['localities'])} localities from data key")
                
                # Debug: Show sample locality
                if lookups['localities']:
                    sample = lookups['localities'][0]
                    print(f"📍 Sample locality: {sample.get('locality_name')} (ID: {sample.get('locality_IDPK')}, CityFK: {sample.get('locality_cityIDFK')})")
        else:
            print("⚠️ locality.json file not found!")
        
        return lookups
    except Exception as e:
        print(f"❌ Error loading lookups: {e}")
        import traceback
        traceback.print_exc()
        return lookups


# ============================================================================
# DIAGNOSTIC FUNCTION
# ============================================================================

def diagnose_locality_data(lookups):
    """Diagnose locality data structure and city mapping"""
    print("\n" + "="*60)
    print("🔍 LOCALITY DATA DIAGNOSTICS")
    print("="*60)
    
    localities = lookups.get('localities', [])
    print(f"Total localities loaded: {len(localities)}")
    
    if not localities:
        print("❌ No localities found!")
        return
    
    # Check structure
    print(f"\n📋 Sample locality structure:")
    sample = localities[0]
    for key, value in sample.items():
        print(f"   {key}: {value} (type: {type(value).__name__})")
    
    # Group by city
    by_city = {}
    for loc in localities:
        city_fk = str(loc.get('locality_cityIDFK', 'unknown'))
        by_city[city_fk] = by_city.get(city_fk, 0) + 1
    
    print(f"\n🏙️ Localities by City ID:")
    for city_id, count in sorted(by_city.items(), key=lambda x: x[1], reverse=True)[:10]:
        city_name = [k for k, v in lookups['cities'].items() if str(v) == city_id]
        city_name = city_name[0] if city_name else 'Unknown'
        print(f"   City ID {city_id} ({city_name}): {count} localities")
    
    # Show sample localities for city_id = 1
    print(f"\n📍 Sample localities for city_id = 1 (Bangalore):")
    bangalore_locs = [loc for loc in localities if str(loc.get('locality_cityIDFK')) == '1'][:5]
    if bangalore_locs:
        for loc in bangalore_locs:
            print(f"   - {loc.get('locality_name')} (ID: {loc.get('locality_IDPK')})")
    else:
        print("   ❌ No localities found for city_id = 1!")
        print(f"   Available city IDs: {list(by_city.keys())[:10]}")
    
    print("="*60 + "\n")


# ============================================================================
# UPDATED SESSION STATE INITIALIZATION
# ============================================================================

# ============================================================================
# UPDATED SESSION STATE INITIALIZATION
# ============================================================================

if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.stop_requested = False
    st.session_state.scraper_status = {k: {'status': 'idle', 'count': 0, 'log': []} for k in SCRAPERS}
    st.session_state.msg_queue = queue.Queue()
    st.session_state.api_stats = {'posted': 0, 'failed': 0, 'skipped': 0, 'log': []}
    st.session_state.all_props = []
    st.session_state.unique_props = []
    st.session_state.by_city = {}
    st.session_state.data_loaded = False
    st.session_state.load_time = 0
    st.session_state.lookups = load_lookup_tables()
    st.session_state.ai_enabled = AI_ENABLED
    st.session_state.posted_new_amenities = set()
    st.session_state.ai_cache = {}
    st.session_state.mapping_log = {}
    st.session_state.batch_size = 10
    st.session_state.posted_property_hashes = set()
    
    # Load previously posted properties
    posted_log_file = LOG_DIR / 'posted_properties.json'
    if posted_log_file.exists():
        try:
            with open(posted_log_file, 'r') as f:
                posted_data = json.load(f)
                st.session_state.posted_property_hashes = set(posted_data.get('posted_hashes', []))
        except:
            pass
    
    # DO NOT run diagnostics here - it blocks UI
    # User can trigger manually with button in sidebar
    
    # COMMENT OUT DIAGNOSTICS - It's too heavy for startup
    # diagnose_locality_data(st.session_state.lookups)
    # RUN DIAGNOSTICS
    #diagnose_locality_data(st.session_state.lookups)
# ============================================================================
# UTILITIES
# ============================================================================
def load_json_file(path):
    """Load single JSON file safely"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None
def normalize_text(text):
    if not text: return ""
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', str(text).lower())).strip()

def normalize_for_comparison(text):
    """Advanced normalization for comparison"""
    if not text:
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove common suffixes/prefixes
    text = re.sub(r'\b(phase|tower|block|wing)\s*[0-9ivx]+\b', '', text)
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove common words that don't help matching
    stop_words = {'the', 'by', 'at', 'in', 'on', 'near', 'apartment', 'apartments', 
                  'villa', 'villas', 'project', 'residential', 'commercial'}
    words = [w for w in text.split() if w not in stop_words]
    
    return ' '.join(words)

# ============================================================================
# AI-POWERED MAPPING FUNCTIONS
def generate_property_hash(prop):
    """
    FIXED: Generate unique hash with better fallback strategy
    """
    # Strategy 1: URL (most reliable)
    url = (prop.get('url') or prop.get('link') or 
           prop.get('refer_links') or prop.get('property_url', '')).strip()
    if url:
        url_clean = url.lower().split('?')[0].rstrip('/')
        return hashlib.md5(f"URL:{url_clean}".encode()).hexdigest()
    
    # Strategy 2: Name + City + Builder
    project_name = normalize_for_comparison(
        prop.get('project_name') or prop.get('name') or prop.get('title', '')
    )

    city = normalize_text(prop.get('city_name', ''))

    builder = normalize_for_comparison(
        prop.get('builder') or prop.get('builder_name', '')
    )
    
    if project_name and city and builder and builder != 'unknown':
        return hashlib.md5(f"PCB:{project_name}|{city}|{builder}".encode()).hexdigest()
    
    # Strategy 3: Name + Location + RERA (if available)
    location = normalize_for_comparison(
        prop.get('location') or prop.get('locality') or prop.get('address', '')
    )
    rera = str(prop.get('rera_id') or prop.get('rera_number', '')).strip()
    
    if project_name and location:
        if rera:
            return hashlib.md5(f"PLR:{project_name}|{location}|{rera}".encode()).hexdigest()
        return hashlib.md5(f"PL:{project_name}|{location}".encode()).hexdigest()
    
    # Strategy 4: Last resort - use all available fields
    # This is LESS strict to catch edge cases
    hash_parts = [
        f"name:{project_name}" if project_name else "",
        f"city:{city}" if city else "",
        f"loc:{location}" if location else "",
        f"builder:{builder}" if builder else "",
        f"rera:{rera}" if rera else ""
    ]
    hash_string = "|".join([p for p in hash_parts if p])
    
    if hash_string:
        return hashlib.md5(f"FALLBACK:{hash_string}".encode()).hexdigest()
    
    # Absolute fallback - very rare
    return hashlib.md5(f"UNKNOWN:{str(prop)[:100]}".encode()).hexdigest()


# ============================================================================
def save_posted_hashes(new_hashes):
    """Save posted property hashes with proper permissions"""
    try:
        posted_log_file = LOG_DIR / 'posted_properties.json'
        existing_hashes = set()
        
        if posted_log_file.exists():
            try:
                with open(posted_log_file, 'r') as f:
                    content = f.read().strip()
                    if content:
                        data = json.loads(content)
                        existing_hashes = set(data.get('posted_hashes', []))
            except json.JSONDecodeError:
                print("⚠️ Corrupted posted_properties.json - resetting")
                existing_hashes = set()
        
        combined_hashes = existing_hashes | new_hashes | st.session_state.posted_property_hashes
        
        with open(posted_log_file, 'w') as f:
            json.dump({
                'last_updated': datetime.now().isoformat(),
                'total_posted': len(combined_hashes),
                'posted_hashes': list(combined_hashes)
            }, f, indent=2)
        
        ensure_file_permissions(posted_log_file)  # ADD THIS
        
        print(f"✅ Saved {len(combined_hashes)} posted property hashes")
    except Exception as e:
        print(f"❌ Failed to save posted hashes: {e}")

def ai_map_property_complete(prop, lookups, use_cache=True):
    """
    Enhanced AI mapping - extracts from ALL available fields
    """
    if not st.session_state.ai_enabled:
        return create_basic_mapping(prop)
    
    prop_hash = generate_property_hash(prop)
    
    if use_cache and prop_hash in st.session_state.ai_cache:
        return st.session_state.ai_cache[prop_hash]
    
    display_name = (prop.get('project_name') or prop.get('name') or 
                   prop.get('title') or prop.get('address', 'Unnamed Property'))
    
    print(f"\n{'='*60}")
    print(f"🏠 AI Mapping: {display_name}")
    
    # Check if property already has amenities field
    raw_amenities = prop.get('amenities', [])
    print(f"📋 Raw amenities found: {raw_amenities}")
    
    # EXTRACT EVERYTHING - Send full property data to AI
    full_prop_json = json.dumps(prop, indent=2)
    
    available_options = {
        'cities': list(lookups['cities'].keys()),
        'property_types': list(lookups['property_types'].keys()),
        'property_status': list(lookups['property_status'].keys()),
        'bhk_types': list(lookups['bhk'].keys()),
        'amenities': list(lookups['amenities'].keys())
    }
    
    # Show sample of available amenities for better matching
    amenities_sample = ', '.join(available_options['amenities'][:100])
    
    prompt = f"""You are a property data extraction expert. Extract ALL possible data from this JSON.

FULL PROPERTY DATA:
{full_prop_json}

DATABASE OPTIONS:
- Cities: {', '.join(available_options['cities'])}
- Property Types: {', '.join(available_options['property_types'])}
- Status: {', '.join(available_options['property_status'])}
- BHK: {', '.join(available_options['bhk_types'])}
- Amenities (first 100): {amenities_sample}

EXTRACTION RULES (SEARCH EVERYWHERE):

1. **BHK** - Search in: bhk, bedrooms, title, name, project_name, description, configuration
   - Extract bedroom COUNT only: "2BHK" → "2 bhk", "3 BR" → "3 bhk", "Studio" → "1 bhk"
   - If range "2-3 BHK" → use minimum "2 bhk" if there is more than one option pass it comma separated "2 bhk, 3 bhk"

2. **Area (sqft)** - Search in: area, super_area, carpet_area, built_up_area, plot_area, size
   - Extract ONLY numeric value: "1200 sqft" → "1200", "1200-1500" → "1350"
   - Convert: "100 sqm" → "1076", "1 acre" → "43560"

3. **Prices** - Search in: price, min_price, max_price, price_range, starting_price, cost
   - Convert to integers:
     * "1.5 Cr" → 15000000
     * "85 Lac" → 8500000
     * "₹50L-₹75L" → min=5000000, max=7500000
   - If single price, use for both min and max

4. **City** - Match: bangalore, mumbai, pune, delhi, hyderabad, chennai

5. **Locality** - Extract from: location, locality, address, area_name

6. **Property Type** - Match: apartment, villa, plot, independent house, penthouse

7. **Status** - Match: under construction, ready to move, upcoming, new launch

8. **Source** - Identify: MagicBricks, Housing.com, NoBroker, PropTiger, CommonFloor, RoofAndFloor, HomznSpace, Propsoch

9. **AMENITIES (CRITICAL)** - Search in: amenities, features, facilities,Project Key USPs, specifications or in full dict of that property json and identify all amenities
   - Look for: gym, swimming pool, park, clubhouse, security, parking, power backup, lift, garden, playground
   - Match EXACTLY to database list above
   - Common variations:
     * "swimming pool" / "pool" → match to "swimming pool"
     * "gym" / "gymnasium" / "fitness center" → match to "gym"
     * "car parking" / "parking" → match to "parking"
     * "children play area" / "playground" → match to "play area"
     * "24/7 security" / "security" → match to "security"
     * "club house" / "clubhouse" → match to "club house"
   - If amenity NOT in database, add to "amenities_new"

10. **POSSESSION DATE** - Search in: possession_date, possession, ready_to_move_date, completion_date, handover_date, delivery_date, expected_completion
   - Extract date in format: "YYYY-MM-DD" or "YYYY-MM" or "YYYY"
   - Examples:
     * "Dec 2025" → "2025-12"
     * "Q4 2024" → "2024-10"
     * "March 2026" → "2026-03"
     * "Ready to Move" → "2024-12" (current date)
     * "Immediate" → "2024-12" (current date)
     * "2025" → "2025-01"
   - If range "2024-2025" → use earlier date "2024-01"
   - If text like "Ready to Move" or "Immediate" → use current date "2024-12"
11. **RERA ID** - Search in: rera_id, rera_number, rera_registration, registration_number, rera
    - Extract alphanumeric code: "RERA123456", "PRM/KA/RERA/1251/446/PR/171120/002345"

Return ONLY this JSON (no markdown):
{{
  "city_name": "bangalore",
  "locality_name": "whitefield",
  "property_type_name": "apartment",
  "status_name": "under construction",
  "bhk_name": "2 bhk, 3 bhk",
  "area_sqft": "1200",
  "min_price": "5000000",
  "max_price": "7500000",
  "possession_date": "2025-12",
  "rera_id": "PRM/KA/RERA/1251/446",
  "source_name": "MagicBricks",
  "amenities_matched": ["gym", "swimming pool", "parking"],
  "amenities_new": ["robot parking"]
}}

IMPORTANT: 
- Extract amenities even if listed as array, comma-separated, or bullet points
- Match amenities to database names (ignore case, spaces, punctuation)
- BHK must be format "X bhk" where X is a number
- Mandatory :If any field not found, return empty string or empty list
- Return valid JSON only, no extra text"""


    try:
        response = client.responses.create(
            model=MODEL_NAME,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a property data extraction expert. Return only valid JSON."}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }
            ],
            temperature=0.1,
            max_output_tokens=1200
        )

        result = response.output_text.strip()
        result = result.replace("```json", "").replace("```", "").strip()
        ai_mapping = json.loads(result)

        # Convert to API format
        mapped_data = convert_names_to_ids(ai_mapping, prop, lookups)

        # Cache
        st.session_state.ai_cache[prop_hash] = mapped_data

        # Debug
        print(f"✅ Extracted:")
        print(f"   BHK: {ai_mapping.get('bhk_name')} → ID: {mapped_data.get('bhk_id')}")
        print(f"   Area: {mapped_data.get('area')} sqft")
        print(f"   Price: {mapped_data.get('min_price')}-{mapped_data.get('max_price')}")
        print(f"   Amenities matched: {ai_mapping.get('amenities_matched', [])}")
        print(f"   Amenities IDs: {mapped_data.get('amenities_id')}")
        print(f"   New amenities: {mapped_data.get('new_amenities')}")

        return mapped_data

    except Exception as e:
        print("❌ AI mapping failed:")
        import traceback
        traceback.print_exc()
        return create_basic_mapping(prop)

def extract_rera_id(prop):
    """Extract first RERA ID from array or fallback to rera_id field"""
    rera_ids = prop.get('rera_ids', [])
    if rera_ids and isinstance(rera_ids, list) and len(rera_ids) > 0:
        return str(rera_ids[0])
    return str(prop.get('rera_id', ''))

def extract_amenity_ids(prop):
    """Extract amenity_id values from amenities array"""
    amenities = prop.get('amenities', [])
    if amenities and isinstance(amenities, list):
        amenity_ids = [str(item.get('amenity_id', '')) for item in amenities if item.get('amenity_id')]
        return ','.join(amenity_ids)
    return ''

def extract_new_amenities(prop):
    """Extract amenity names from new_amenities array"""
    new_amenities = prop.get('new_amenities', [])
    if new_amenities and isinstance(new_amenities, list):
        amenity_names = [str(item.get('amenity', '')) for item in new_amenities if item.get('amenity')]
        return ','.join(amenity_names)
    return ''

def extract_date_from_details(prop, date_type):
    """
    Extract launch_date or possession_date from details object
    date_type: 'Launch Date' or 'Possession Date'
    Returns: date in 'YYYY-MM-DD' format or empty string
    """
    details = prop.get('details', {})
    if not details:
        return ''
    
    # Search for key containing the date_type
    for key, value in details.items():
        if date_type in key and isinstance(value, str):
            try:
                # Parse date formats like "01 May 2025" or "01 December 2029"
                date_obj = datetime.strptime(value.strip(), '%d %B %Y')
                return date_obj.strftime('%Y-%m-%d')
            except ValueError:
                try:
                    # Try alternative format "01 May 25"
                    date_obj = datetime.strptime(value.strip(), '%d %b %Y')
                    return date_obj.strftime('%Y-%m-%d')
                except ValueError:
                    continue
    return ''

def extract_floor_plans_json(prop):
    """Extract floor plans and return as JSON string for API"""
    floor_plans = prop.get('floor_plans', [])
    
    if not floor_plans:
        return ''
    
    # Clean and format floor plans for API
    api_floor_plans = []
    
    for fp in floor_plans:
        # Only include if has required data
        if fp.get('image_url') and fp.get('bhk_id'):
            api_floor_plans.append({
                'image_url': fp.get('image_url', ''),
                'bhk_id': str(fp.get('bhk_id', '')),
                'bhk_area_type': str(fp.get('bhk_area_type', '')),
                'size': fp.get('size') or 0,  # Default to 0 if None
                'price': fp.get('price') or 0  # Default to 0 if None
            })
    
    if not api_floor_plans:
        return ''
    
    # Convert to JSON string
    try:
        return json.dumps(api_floor_plans, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️ Error converting floor plans to JSON: {e}")
        return ''


def convert_names_to_ids(ai_mapping, prop, lookups):
    """Convert AI-returned names to database IDs - FIXED amenity matching"""
    
    city_name = ai_mapping.get('city_name', '').lower().strip()
    city_id = str(lookups['cities'].get(city_name, ''))
    
    locality_name = ai_mapping.get('locality_name', '')
    locality_id = ''
    if locality_name and city_id:
        locality_id = find_locality_id(locality_name, city_id, lookups)
    
    ptype_name = ai_mapping.get('property_type_name', '').lower().strip()
    proptype_id = str(lookups['property_types'].get(ptype_name, ''))
    
    status_name = ai_mapping.get('status_name', '').lower().strip()
    propstatus_id = str(lookups['property_status'].get(status_name, ''))
    
    bhk_name = ai_mapping.get('bhk_name', '').lower().strip()
    bhk_id = str(lookups['bhk'].get(bhk_name, ''))
     
    
    # URL extraction
    url = (prop.get('url') or prop.get('link') or 
           prop.get('refer_links') or prop.get('property_url') or 
           prop.get('detail_url') or prop.get('source_url') or '')

    # # Extract RERA ID
    # rera_ids_array = prop.get('rera_ids', [])
    # if rera_ids_array and isinstance(rera_ids_array, list) and len(rera_ids_array) > 0:
    #     rera_id = str(rera_ids_array[0])
    # else:
    #     rera_id = str(prop.get('rera_id', ''))
    
    # Build API payload
    api_data = {
        'city_id': str(prop.get('city_id', '')),
        'propname': str(prop.get('project_name') or prop.get('name') or prop.get('title', 'Unknown')),
        'refer_links': str(prop.get('url') or prop.get('link', '') or prop.get('refer_links', '')),
        'prop_source': str(prop.get('_source', 'Unknown')),
        'builder_name': str(prop.get('builder') or prop.get('builder_name', '')),
        'builder_id': str(prop.get('builder_id')),
        'builder_exp': str(prop.get('builder_exp')),
        'builder_address': str(prop.get('builder_address')),
        'proptype_id': str(prop.get('type_id', '')),
        'locality_id': str(prop.get('locality_id', '')),
        'locality_name': str(prop.get('location', '')),
        'address': str(prop.get('location') or prop.get('address', '')),
        'latitude': str(prop.get('latitude', '')),
        'longitude': str(prop.get('longitude', '')),
        'area': str(prop.get('acre', '')),
        'total_units': str(prop.get('units', '')),
        'min_price': str(prop.get('price_min', '')),
        'max_price': str(prop.get('price_max', '')),
        'possession_date': extract_date_from_details(prop, 'Possession Date'),
        'launch_date': extract_date_from_details(prop, 'Launch Date'),
        'rera_id': extract_rera_id(prop),
        'num_towers': str(prop.get('towers', '')),
        'amenities_id': extract_amenity_ids(prop),
        'new_amenities': extract_new_amenities(prop),
        'floor_plan': extract_floor_plans_json(prop)
    }
    
    return api_data

def extract_new_amenities_from_main_json():
    """Extract all new amenities from main.json into separate file"""
    main_json_path = OUTPUT_DIR / 'main.json'
    
    if not main_json_path.exists():
        st.error("❌ main.json not found!")
        return None
    
    try:
        with open(main_json_path, 'r') as f:
            properties = json.load(f)
        
        all_new_amenities = set()
        amenity_frequency = {}
        
        for prop in properties:
            new_amenities_str = prop.get('new_amenities', '')
            if new_amenities_str:
                amenities_list = [a.strip() for a in new_amenities_str.split(',') if a.strip()]
                for amenity in amenities_list:
                    all_new_amenities.add(amenity)
                    amenity_frequency[amenity] = amenity_frequency.get(amenity, 0) + 1
        
        # Sort by frequency
        sorted_amenities = sorted(amenity_frequency.items(), key=lambda x: x[1], reverse=True)
        
        amenities_data = {
            'total_unique_amenities': len(all_new_amenities),
            'timestamp': datetime.now().isoformat(),
            'amenities': [
                {
                    'name': amenity,
                    'frequency': count
                }
                for amenity, count in sorted_amenities
            ]
        }
        
        # Save to file
        new_amenities_file = OUTPUT_DIR / 'new_amenities.json'
        with open(new_amenities_file, 'w', encoding='utf-8') as f:
            json.dump(amenities_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Extracted {len(all_new_amenities)} unique new amenities")
        return amenities_data
        
    except Exception as e:
        st.error(f"❌ Error extracting amenities: {e}")
        return None


def post_to_api_from_file(log_placeholder, limit=None):
    """Post with FORM DATA - Fixed to prevent duplicate posting"""
    file_path = OUTPUT_DIR / 'main.json'
    
    if not file_path.exists():
        st.error("❌ main.json not found!")
        return

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            all_properties = json.load(f)
        
        # DEDUPLICATE BEFORE POSTING - Filter out already posted properties
        properties_to_post = []
        skipped_already_posted = 0
        
        for prop in all_properties:
            prop_hash = prop.get('_hash') or generate_property_hash(prop)
            
            # Skip if already posted
            if prop_hash in st.session_state.posted_property_hashes:
                skipped_already_posted += 1
                continue
            
            properties_to_post.append(prop)
        
        if skipped_already_posted > 0:
            st.info(f"ℹ️ Skipped {skipped_already_posted} already posted properties")
        
        if not properties_to_post:
            st.warning("⚠️ No new properties to post - all already posted!")
            return
        
        # Apply limit to new properties only
        if limit:
            properties_to_post = properties_to_post[:limit]
        
        live_logs = []
        total = len(properties_to_post)
        progress_bar = st.progress(0, text=f"Processing 0/{total}")
        
        st.info(f"📤 Posting {total} NEW properties (out of {len(all_properties)} total)")
        
        for i, prop in enumerate(properties_to_post):
            if st.session_state.stop_requested:
                live_logs.append("🛑 Stopped by user")
                break
            
            prop_hash = prop.get('_hash') or generate_property_hash(prop)
            prop_name = prop.get('propname', 'Unknown')
            
            # Build payload
            payload = {}
            for key, value in prop.items():
                if key.startswith('_'):
                    continue
                if value is not None and str(value).strip():
                    payload[key] = str(value).strip()
            
            # Ensure required fields
            if 'city_id' not in payload or not payload['city_id']:
                payload['city_id'] = '1'
            if 'propname' not in payload:
                payload['propname'] = 'Unknown Property'
            if 'refer_links' not in payload:
                payload['refer_links'] = ''
            
            try:
                print(f"\n{'='*60}")
                print(f"[{i+1}/{total}] Posting NEW: {prop_name}")
                print(f"Hash: {prop_hash[:16]}...")
                
                response = requests.post(
                    API_ENDPOINT, 
                    data=payload,
                    timeout=15
                )
                
                # Handle non-JSON responses
                response_text = response.text.strip()
                response_json = {}
                
                try:
                    if response_text and response_text.startswith('{'):
                        response_json = json.loads(response_text)
                except json.JSONDecodeError:
                    response_json = {'raw_response': response_text[:200]}
                
                if response.status_code in [200, 201]:
                    status_msg = (f"✅ [{i+1}/{total}] SUCCESS: {prop_name}\n"
                                f"   Response: {response_json}\n"
                                f"   BHK: {payload.get('bhk_id','N/A')} | "
                                f"Area: {payload.get('area','N/A')} sqft\n")
                    st.session_state.api_stats['posted'] += 1
                    
                    # Mark as posted IMMEDIATELY
                    st.session_state.posted_property_hashes.add(prop_hash)
                    save_posted_hashes({prop_hash})
                else:
                    error_detail = response_text[:600]
                    status_msg = (f"❌ [{i+1}/{total}] FAILED: {prop_name}\n"
                                f"   Status: {response.status_code}\n"
                                f"   Response: {error_detail}\n")
                    st.session_state.api_stats['failed'] += 1
                    
                    # Save failure log
                    with open(LOG_DIR / 'failed_properties.json', 'a') as f:
                        json.dump({
                            'index': i,
                            'property_name': prop_name,
                            'hash': prop_hash,
                            'payload': payload,
                            'status_code': response.status_code,
                            'response': error_detail,
                            'timestamp': datetime.now().isoformat()
                        }, f, indent=2)
                        f.write('\n')
            
            except Exception as e:
                status_msg = f"⚠️ [{i+1}/{total}] Error: {prop_name} -> {str(e)[:200]}\n"
                st.session_state.api_stats['failed'] += 1
                import traceback
                traceback.print_exc()
            
            live_logs.append(status_msg)
            st.session_state.api_stats['log'] = live_logs
            log_display = "\n".join(live_logs[-20:])
            log_placeholder.text_area("Live Logs", log_display, height=400, key=f"log_{i}")
            progress_bar.progress((i + 1) / total, text=f"Processing {i+1}/{total}")
        
        # Summary
        summary = (f"\n{'='*60}\n"
                  f"🎉 POSTING COMPLETE!\n"
                  f"✅ Posted: {st.session_state.api_stats['posted']}\n"
                  f"❌ Failed: {st.session_state.api_stats['failed']}\n"
                  f"⏭️ Skipped (already posted): {skipped_already_posted}\n"
                  f"{'='*60}\n")
        live_logs.append(summary)
        log_placeholder.text_area("Final Results", "\n".join(live_logs[-25:]), height=400)
        
    except Exception as e:
        st.error(f"🚨 Error: {e}")
        import traceback
        st.code(traceback.format_exc())
...
def post_new_amenities_to_api():
    """Post new amenities to API endpoint"""
    new_amenities_file = OUTPUT_DIR / 'new_amenities.json'
    
    if not new_amenities_file.exists():
        st.error("❌ new_amenities.json not found! Extract amenities first.")
        return
    
    try:
        with open(new_amenities_file, 'r') as f:
            amenities_data = json.load(f)
        
        amenities_list = amenities_data.get('amenities', [])
        
        if not amenities_list:
            st.warning("⚠️  No new amenities to post")
            return
        
        # API endpoint for amenities (adjust as needed)
        AMENITIES_API_ENDPOINT = "API_ENDPOINT"
        
        posted_count = 0
        failed_count = 0
        
        progress_bar = st.progress(0, text=f"Posting amenities...")
        log_area = st.empty()
        logs = []
        
        for i, amenity_obj in enumerate(amenities_list):
            amenity_name = amenity_obj['name']
            
            try:
                response = requests.post(
                    AMENITIES_API_ENDPOINT,
                    data={'amenity_name': amenity_name},
                    timeout=10
                )
                
                if response.status_code in [200, 201]:
                    logs.append(f"✅ Posted: {amenity_name} (freq: {amenity_obj['frequency']})")
                    posted_count += 1
                else:
                    logs.append(f"❌ Failed: {amenity_name} (Code: {response.status_code})")
                    failed_count += 1
            except Exception as e:
                logs.append(f"⚠️  Error: {amenity_name} -> {str(e)[:100]}")
                failed_count += 1
            
            progress_bar.progress((i + 1) / len(amenities_list), 
                                 text=f"Posting {i+1}/{len(amenities_list)}")
            log_area.text_area("Amenity Posting Log", "\n".join(logs[-15:]), height=300)
        
        st.success(f"✅ Posted: {posted_count} | ❌ Failed: {failed_count}")
        
    except Exception as e:
        st.error(f"🚨 Error: {e}")

def find_locality_id(locality_name, city_id, lookups):
    """Find locality ID filtered by city"""
    localities = lookups.get('localities', [])
    
    # Filter by city
    city_localities = [
        loc for loc in localities 
        if str(loc.get('locality_cityIDFK')) == str(city_id)
    ]
    
    locality_name_lower = locality_name.lower().strip()
    
    # Exact match
    for loc in city_localities:
        if loc.get('locality_name', '').lower().strip() == locality_name_lower:
            return str(loc.get('locality_IDPK', ''))
    
    # Partial match
    for loc in city_localities:
        db_name = loc.get('locality_name', '').lower().strip()
        if locality_name_lower in db_name or db_name in locality_name_lower:
            return str(loc.get('locality_IDPK', ''))
    
    return ''
# ADD THIS NEW FUNCTION - Test what PHP receives

def test_api_connection():
    """Test API connection WITHOUT posting data"""
    st.write("### 🧪 Testing API Connection")
    st.info("This test will CHECK the API endpoint without posting any data")
    
    try:
        st.write("**Test 1: Endpoint Reachability**")
        response = requests.get(API_ENDPOINT.replace('/buy_postProperty', ''), timeout=5)
        st.success(f"✅ Server reachable - Status: {response.status_code}")
    except Exception as e:
        st.error(f"❌ Server unreachable: {e}")
        return
    
    try:
        st.write("**Test 2: Endpoint Validation**")
        from urllib.parse import urlparse
        parsed = urlparse(API_ENDPOINT)
        st.code(f"Protocol: {parsed.scheme}\nHost: {parsed.netloc}\nPath: {parsed.path}")
        
        if parsed.scheme not in ['http', 'https']:
            st.warning("⚠️ Invalid protocol")
            return
        
        st.success("✅ Endpoint structure valid")
    except Exception as e:
        st.error(f"❌ Invalid endpoint: {e}")
        return
    
    st.write("**Test 3: POST Request Preview (Not Sent)**")
    
    sample_payload = {
        'city_id': '1',
        'propname': 'TEST_CONNECTION_CHECK',
        'refer_links': 'https://test.com'
    }
    
    st.json(sample_payload)
    st.info("ℹ️ Above payload would be sent as form-data")
    
    st.write("**Test 4: Equivalent cURL Command**")
    curl_params = ' '.join([f"-d '{k}={v}'" for k, v in sample_payload.items()])
    curl_cmd = f"curl -X POST {API_ENDPOINT} {curl_params}"
    st.code(curl_cmd, language='bash')
    
    st.success("✅ All validation tests passed!")
    st.warning("⚠️ No test data was posted to the API")


def create_basic_mapping(prop):
    """Fallback basic mapping when AI is disabled"""

    return {
        'city_id': str(prop.get('city_id', '')),
        'propname': str(prop.get('project_name') or prop.get('name') or prop.get('title', 'Unknown')),
        'refer_links': str(prop.get('url') or prop.get('link', '') or prop.get('refer_links', '')),
        'prop_source': str(prop.get('_source', 'Unknown')),
        'builder_name': str(prop.get('builder') or prop.get('builder_name', '')),
        'builder_id': str(prop.get('builder_id')),
        'builder_exp': str(prop.get('builder_exp')),
        'builder_address': str(prop.get('builder_address')),
        'proptype_id': str(prop.get('type_id', '')),
        'locality_id': str(prop.get('locality_id', '')),
        'locality_name': str(prop.get('location', '')),
        'address': str(prop.get('location') or prop.get('address', '')),
        'latitude': str(prop.get('latitude', '')),
        'longitude': str(prop.get('longitude', '')),
        'area': str(prop.get('acre', '')),
        'total_units': str(prop.get('units', '')),
        'min_price': str(prop.get('price_min', '')),
        'max_price': str(prop.get('price_max', '')),
        'possession_date': extract_date_from_details(prop, 'Possession Date'),
        'launch_date': extract_date_from_details(prop, 'Launch Date'),
        'rera_id': extract_rera_id(prop),
        'num_towers': str(prop.get('towers', '')),
        'amenities_id': extract_amenity_ids(prop),
        'new_amenities': extract_new_amenities(prop),
        'floor_plan': extract_floor_plans_json(prop)
    }
# ============================================================================
# POSTED AMENITIES TRACKING (to prevent re-posting)
# ============================================================================

# Add this to session state initialization
def filter_already_posted_amenities(new_amenities):
    """Filter out amenities that have already been posted to API across ALL properties"""
    if not new_amenities:
        return []
    
    filtered_amenities = []
    
    for amenity in new_amenities:
        amenity_normalized = normalize_text(amenity)
        
        if amenity_normalized in st.session_state.posted_new_amenities:
            print(f"   🔄 Skipping already posted: '{amenity}'")
        else:
            filtered_amenities.append(amenity)
            st.session_state.posted_new_amenities.add(amenity_normalized)
            print(f"   ✅ Marking '{amenity}' for posting (first time)")
    
    return filtered_amenities

# ============================================================================
# AI-POWERED TRANSFORMATION
# ============================================================================

# ============================================================================
# GROUND TRUTH COMPARISON
# ============================================================================

def calculate_similarity(prop1, prop2):
    """
    EXISTING FUNCTION - Keep as is, but here's the complete version
    Enhanced similarity calculation with multiple factors
    """
    score = 0
    max_score = 100
    
    # 1. PROJECT NAME MATCHING (40 points)
    name1 = normalize_for_comparison(
        prop1.get('project_name') or prop1.get('name') or prop1.get('title', '')
    )
    name2 = normalize_for_comparison(
        prop2.get('project_name') or prop2.get('name') or prop2.get('title', '')
    )
    
    if name1 and name2:
        if name1 == name2:
            score += 40
        elif name1 in name2 or name2 in name1:
            score += 35
        else:
            # Jaccard similarity (word overlap)
            words1 = set(name1.split())
            words2 = set(name2.split())
            if words1 and words2:
                intersection = len(words1 & words2)
                union = len(words1 | words2)
                jaccard = intersection / union if union > 0 else 0
                score += int(40 * jaccard)
    
    # 2. CITY MATCHING (20 points)
    city1 = normalize_text(prop1.get('city', ''))
    city2 = normalize_text(prop2.get('city', ''))
    if city1 and city2:
        if city1 == city2:
            score += 20
        elif city1 in city2 or city2 in city1:
            score += 15
    
    # 3. BUILDER MATCHING (20 points)
    builder1 = normalize_for_comparison(prop1.get('builder') or prop1.get('builder_name', ''))
    builder2 = normalize_for_comparison(prop2.get('builder') or prop2.get('builder_name', ''))
    if builder1 and builder2 and builder1 != 'unknown' and builder2 != 'unknown':
        if builder1 == builder2:
            score += 20
        elif builder1 in builder2 or builder2 in builder1:
            score += 15
    
    # 4. LOCATION/LOCALITY MATCHING (10 points)
    loc1 = normalize_for_comparison(prop1.get('location') or prop1.get('locality', ''))
    loc2 = normalize_for_comparison(prop2.get('location') or prop2.get('locality', ''))
    if loc1 and loc2:
        if loc1 == loc2:
            score += 10
        elif loc1 in loc2 or loc2 in loc1:
            score += 7
    
    # 5. PRICE RANGE OVERLAP (10 points)
    try:
        price1_min = prop1.get('min_price') or prop1.get('price')
        price1_max = prop1.get('max_price') or prop1.get('price')
        price2_min = prop2.get('min_price') or prop2.get('price')
        price2_max = prop2.get('max_price') or prop2.get('price')
        
        if price1_min and price2_min:
            p1_min = float(str(price1_min).replace(',', ''))
            p1_max = float(str(price1_max or price1_min).replace(',', ''))
            p2_min = float(str(price2_min).replace(',', ''))
            p2_max = float(str(price2_max or price2_min).replace(',', ''))
            
            # Check overlap
            overlap_start = max(p1_min, p2_min)
            overlap_end = min(p1_max, p2_max)
            
            if overlap_start <= overlap_end:
                score += 10
            else:
                # Within 20% tolerance
                avg1 = (p1_min + p1_max) / 2
                avg2 = (p2_min + p2_max) / 2
                diff_pct = abs(avg1 - avg2) / max(avg1, avg2) if max(avg1, avg2) > 0 else 1
                if diff_pct < 0.2:
                    score += 5
    except:
        pass
    
    return min(score, max_score)



def compare_ground_truth(threshold=60):
    """Compare scraped data with ground truth"""
    if not INFO_JSON.exists():
        st.warning("⚠️ info.json not found!")
        return None
    
    try:
        with open(INFO_JSON, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
        
        if not isinstance(ground_truth, list):
            ground_truth = [ground_truth] if isinstance(ground_truth, dict) else []
        
        print(f"\n{'='*60}")
        print(f"GROUND TRUTH COMPARISON (Threshold: {threshold}%)")
        print(f"{'='*60}")
        
        gt_index = {}
        for idx, gt_prop in enumerate(ground_truth):
            title = normalize_text(gt_prop.get('title', ''))
            city = gt_prop.get('city', '')
            if title:
                key = f"{city}_{' '.join(title.split()[:3])}"
                if key not in gt_index:
                    gt_index[key] = []
                gt_index[key].append(idx)
        
        output_files = [f for f in OUTPUT_DIR.glob('*.json') 
                       if f.name not in ['info.json', 'main.json', 'comparison_log.json']]
        
        results = {}
        total_matched_indices = set()
        
        for file_path in output_files:
            scraped_data = load_json_file(file_path)
            if not scraped_data:
                continue
            
            if isinstance(scraped_data, dict):
                props = []
                for v in scraped_data.values():
                    if isinstance(v, list):
                        props.extend(v)
                scraped_data = props
            
            matches_count = 0
            matched_indices = set()
            
            for scraped_prop in scraped_data:
                s_title = normalize_text(scraped_prop.get('project_name') or 
                                        scraped_prop.get('name') or 
                                        scraped_prop.get('title', ''))
                s_city = scraped_prop.get('city', '')
                
                if not s_title:
                    continue
                
                key = f"{s_city}_{' '.join(s_title.split()[:3])}"
                candidate_indices = gt_index.get(key, [])
                
                best_match_idx = None
                max_score = 0
                
                for gt_idx in candidate_indices:
                    if gt_idx in matched_indices:
                        continue
                    
                    gt_prop = ground_truth[gt_idx]
                    score = calculate_similarity(gt_prop, scraped_prop)
                    
                    if score > max_score:
                        max_score = score
                        best_match_idx = gt_idx
                
                if max_score >= threshold and best_match_idx is not None:
                    matched_indices.add(best_match_idx)
                    total_matched_indices.add(best_match_idx)
                    matches_count += 1
            
            match_rate = (matches_count / len(ground_truth) * 100) if ground_truth else 0
            results[file_path.name] = {
                'total_scraped': len(scraped_data),
                'matches': matches_count,
                'match_rate': round(match_rate, 2)
            }
        
        overall_rate = (len(total_matched_indices) / len(ground_truth) * 100) if ground_truth else 0
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'ground_truth_count': len(ground_truth),
            'overall_matched': len(total_matched_indices),
            'overall_match_rate': round(overall_rate, 2),
            'file_results': results
        }
        
        with open(COMPARISON_LOG, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    except Exception as e:
        print(f"❌ Error in ground truth comparison: {e}")
        return None

# ============================================================================
# SCRAPER EXECUTION
# ============================================================================
# ============================================================================
# UI
# ============================================================================
# Test version - add at the very start of main():
# ============================================================================
# ALL UPDATED FUNCTIONS - Real Estate Scraper Dashboard
# ============================================================================

def run_scraper(key, msg_queue):
    """Run scraper with ENHANCED output handling - FIXED for NoBroker Selenium"""
    import threading
    import io
    
    scraper = SCRAPERS[key]
    msg_queue.put({'scraper': key, 'type': 'status', 'value': 'running'})
    msg_queue.put({'scraper': key, 'type': 'log', 'value': f"Starting {scraper['name']}..."})
    
    python_cmd = sys.executable
    cmd = [python_cmd, '-u', scraper['file']]
    
    msg_queue.put({'scraper': key, 'type': 'log', 'value': f"Python: {python_cmd}"})
    msg_queue.put({'scraper': key, 'type': 'log', 'value': f"File: {scraper['file']}"})
    
    if not os.path.exists(scraper['file']):
        msg_queue.put({'scraper': key, 'type': 'status', 'value': 'error'})
        msg_queue.put({'scraper': key, 'type': 'log', 'value': "ERROR: File not found"})
        return
    
    def safe_decode(text):
        """Safely decode text with emoji and unicode support"""
        if text is None:
            return ""
        try:
            # Handle bytes
            if isinstance(text, bytes):
                return text.decode('utf-8', errors='replace')
            # Handle string with emojis
            return str(text)
        except Exception:
            return ''.join(char if ord(char) < 128 else '?' for char in str(text))
    
    def read_output_improved(pipe, queue, scraper_key, pipe_name):
        """Enhanced thread to read output - handles Selenium live updates"""
        try:
            buffer = ""
            while True:
                chunk = pipe.read(1)  # Read character by character for real-time updates
                if not chunk:
                    break
                
                buffer += chunk
                
                # Process complete lines
                if '\n' in buffer or '\r' in buffer:
                    lines = buffer.replace('\r\n', '\n').replace('\r', '\n').split('\n')
                    buffer = lines[-1]  # Keep incomplete line
                    
                    for line in lines[:-1]:
                        clean_line = line.strip()
                        if clean_line:
                            # Skip noise
                            skip_patterns = [
                                'downloading', 'extracting', 'driver [', 'checking',
                                'get driverversion', 'get latest', 'there is no'
                            ]
                            if any(skip.lower() in clean_line.lower() for skip in skip_patterns):
                                continue
                            
                            safe_line = safe_decode(clean_line)
                            queue.put({'scraper': scraper_key, 'type': 'log', 'value': safe_line})
                            
                            # Extract property count
                            if 'Property' in safe_line and '/' in safe_line:
                                try:
                                    count = int(safe_line.split('/')[0].split()[-1])
                                    queue.put({'scraper': scraper_key, 'type': 'count', 'value': count})
                                except:
                                    pass
            
            # Process remaining buffer
            if buffer.strip():
                safe_line = safe_decode(buffer.strip())
                queue.put({'scraper': scraper_key, 'type': 'log', 'value': safe_line})
                
        except Exception as e:
            queue.put({'scraper': scraper_key, 'type': 'log', 'value': f"[{pipe_name}] Read error: {safe_decode(str(e))}"})
        finally:
            try:
                pipe.close()
            except:
                pass
    
    try:
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUNBUFFERED'] = '1'
        env['WDM_LOG_LEVEL'] = '0'
        env['WDM_PROGRESS_BAR'] = '0'
        env['TERM'] = 'dumb'  # Disable terminal control sequences
        
        msg_queue.put({'scraper': key, 'type': 'log', 'value': "🚀 Launching scraper..."})
        
        # Use text mode with utf-8 encoding
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding='utf-8',
            errors='replace',
            bufsize=0,  # Unbuffered for real-time output
            universal_newlines=True,
            cwd=os.getcwd(),
            env=env
        )
        
        msg_queue.put({'scraper': key, 'type': 'log', 'value': f"✅ Process started (PID: {process.pid})"})
        
        # Start output readers
        stdout_thread = threading.Thread(
            target=read_output_improved,
            args=(process.stdout, msg_queue, key, 'OUT'),
            daemon=True
        )
        stderr_thread = threading.Thread(
            target=read_output_improved,
            args=(process.stderr, msg_queue, key, 'ERR'),
            daemon=True
        )
        
        stdout_thread.start()
        stderr_thread.start()
        
        # Wait for completion with timeout
        try:
            return_code = process.wait(timeout=3600)  # 60 min timeout for NoBroker
        except subprocess.TimeoutExpired:
            msg_queue.put({'scraper': key, 'type': 'log', 'value': '⏰ TIMEOUT: Scraper took too long (60 min)'})
            process.kill()
            msg_queue.put({'scraper': key, 'type': 'status', 'value': 'error'})
            return
        
        # Wait for threads to finish
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)
        
        if return_code == 0:
            msg_queue.put({'scraper': key, 'type': 'status', 'value': 'completed'})
            msg_queue.put({'scraper': key, 'type': 'log', 'value': '🎉 === COMPLETED SUCCESSFULLY ==='})
        else:
            msg_queue.put({'scraper': key, 'type': 'status', 'value': 'error'})
            msg_queue.put({'scraper': key, 'type': 'log', 'value': f'❌ EXIT CODE: {return_code}'})
            
            if return_code == 1:
                msg_queue.put({'scraper': key, 'type': 'log', 'value': '💡 TIP: Check Chrome installation and network connection'})
            
    except FileNotFoundError:
        msg_queue.put({'scraper': key, 'type': 'status', 'value': 'error'})
        msg_queue.put({'scraper': key, 'type': 'log', 'value': f"❌ ERROR: Python not found at {python_cmd}"})
    except Exception as e:
        msg_queue.put({'scraper': key, 'type': 'status', 'value': 'error'})
        msg_queue.put({'scraper': key, 'type': 'log', 'value': f"❌ Exception: {safe_decode(str(e))}"})


def process_messages():
    """Process ALL messages from queue with count updates"""
    processed = 0
    try:
        while not st.session_state.msg_queue.empty():
            msg = st.session_state.msg_queue.get_nowait()
            key = msg.get('scraper')
            
            if key in st.session_state.scraper_status:
                if msg['type'] == 'log':
                    st.session_state.scraper_status[key]['log'].append(msg['value'])
                    processed += 1
                elif msg['type'] == 'status':
                    st.session_state.scraper_status[key]['status'] = msg['value']
                    processed += 1
                elif msg['type'] == 'count':
                    # Update property count from scraper
                    st.session_state.scraper_status[key]['count'] = msg['value']
                    processed += 1
    except:
        pass
    
    return processed


def get_scraper_statistics():
        """Get detailed statistics for each scraper - NEW FOLDER STRUCTURE"""
        stats = {}
        
        for key in SCRAPERS.keys():
            scraper_name = SCRAPERS[key]['name']
            scraper_folder = OUTPUT_DIR / key
            
            # Initialize stats
            total_current = 0
            total_posted = 0
            total_failed = 0
            total_extracted = 0  # NEW: Track total ever scraped
            
            # Check if scraper folder exists
            if not scraper_folder.exists():
                stats[key] = {
                    'extracted': 0,
                    'posted': 0,
                    'failed': 0,
                    'pending': 0
                }
                continue
            
            # Aggregate from all city folders OR direct files
            # Pattern 1: City subfolders (e.g., magicbricks/bangalore/)
            city_folders = [f for f in scraper_folder.iterdir() if f.is_dir() and f.name != 'archive']
            
            if city_folders:
                # Multi-city structure
                for city_folder in city_folders:
                    metadata_file = city_folder / "metadata.json"
                    
                    if metadata_file.exists():
                        try:
                            with open(metadata_file, 'r') as f:
                                meta = json.load(f)
                            
                            # FIXED: Calculate total extracted
                            current_count = meta.get('total_current', 0)
                            posted_count = meta.get('posted', 0)
                            failed_count = meta.get('failed', 0)
                            
                            # Total extracted = current + posted + failed
                            extracted_count = current_count + posted_count + failed_count
                            
                            total_current += current_count
                            total_posted += posted_count
                            total_failed += failed_count
                            total_extracted += extracted_count
                            
                        except:
                            pass
            else:
                # Single file structure (for scrapers without city breakdown)
                metadata_file = scraper_folder / "metadata.json"
                
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            meta = json.load(f)
                        
                        # FIXED: Calculate total extracted
                        current_count = meta.get('total_current', 0)
                        posted_count = meta.get('posted', 0)
                        failed_count = meta.get('failed', 0)
                        
                        # Total extracted = current + posted + failed
                        total_extracted = current_count + posted_count + failed_count
                        total_current = current_count
                        total_posted = posted_count
                        total_failed = failed_count
                        
                    except:
                        pass
            
            stats[key] = {
                'extracted': total_extracted,  # FIXED: Now shows total ever scraped
                'posted': total_posted,
                'failed': total_failed,
                'pending': total_current  # Pending = what's still in current.json
            }
        
        return stats

def post_source_to_api(scraper_key, batch_size, log_placeholder):
    """Post properties from a specific scraper source to API"""
    
    scraper_folder = OUTPUT_DIR / scraper_key
    
    if not scraper_folder.exists():
        st.error(f"❌ No data folder found for {SCRAPERS[scraper_key]['name']}")
        return
    
    # Collect all current.json files (from city subfolders or direct)
    current_files = []
    city_folders = [f for f in scraper_folder.iterdir() if f.is_dir() and f.name != 'archive']
    
    if city_folders:
        # Multi-city structure
        for city_folder in city_folders:
            current_file = city_folder / "current.json"
            posted_file = city_folder / "posted.json"
            failed_file = city_folder / "failed.json"
            metadata_file = city_folder / "metadata.json"
            
            if current_file.exists():
                current_files.append({
                    'current': current_file,
                    'posted': posted_file,
                    'failed': failed_file,
                    'metadata': metadata_file,
                    'city_name': city_folder.name
                })
    else:
        # Single file structure
        current_file = scraper_folder / "current.json"
        posted_file = scraper_folder / "posted.json"
        failed_file = scraper_folder / "failed.json"
        metadata_file = scraper_folder / "metadata.json"
        
        if current_file.exists():
            current_files.append({
                'current': current_file,
                'posted': posted_file,
                'failed': failed_file,
                'metadata': metadata_file,
                'city_name': 'all'
            })
    
    if not current_files:
        st.warning(f"⚠️ No properties found to post for {SCRAPERS[scraper_key]['name']}")
        return
    
    # Collect properties to post
    all_to_post = []
    file_mapping = {}  # Track which file each property belongs to
    
    for file_group in current_files:
        try:
            with open(file_group['current'], 'r', encoding='utf-8') as f:
                props = json.load(f)
            
            for prop in props[:batch_size]:  # Limit per city
                prop['_city_name'] = file_group['city_name']
                all_to_post.append(prop)
                file_mapping[prop.get('_hash', generate_property_hash(prop))] = file_group
            
        except Exception as e:
            st.error(f"Error loading {file_group['current']}: {e}")
    
    if not all_to_post:
        st.warning("⚠️ No properties to post!")
        return
    
    # Limit total batch
    all_to_post = all_to_post[:batch_size]
    
    total = len(all_to_post)
    st.info(f"📤 Posting {total} properties from {SCRAPERS[scraper_key]['name']}...")
    
    progress_bar = st.progress(0, text=f"Processing 0/{total}")
    live_logs = []
    
    posted_by_file = {}  # Track posted properties per file
    failed_by_file = {}
    
    for i, prop in enumerate(all_to_post):
        if st.session_state.stop_requested:
            live_logs.append("🛑 Stopped by user")
            break
        
        prop_hash = prop.get('_hash', generate_property_hash(prop))
        prop_name = prop.get('propname') or prop.get('title') or prop.get('project_name') or 'Unknown'
        
        # AI mapping
        try:
            mapped_prop = ai_map_property_complete(prop, st.session_state.lookups, use_cache=True)
        except Exception as e:
            live_logs.append(f"❌ [{i+1}/{total}] Mapping failed: {prop_name} -> {str(e)[:100]}")
            
            # Add to failed
            file_group = file_mapping.get(prop_hash)
            if file_group:
                if file_group['current'] not in failed_by_file:
                    failed_by_file[file_group['current']] = []
                prop['_error'] = f"Mapping failed: {str(e)[:200]}"
                failed_by_file[file_group['current']].append(prop)
            
            continue
        
        # Build payload
        payload = {}
        for key, value in mapped_prop.items():
            if key.startswith('_'):
                continue
            if value is not None and str(value).strip():
                payload[key] = str(value).strip()
        
        # Ensure required fields
        if 'city_id' not in payload or not payload['city_id']:
            payload['city_id'] = '1'
        if 'propname' not in payload:
            payload['propname'] = prop_name
        if 'refer_links' not in payload:
            payload['refer_links'] = prop.get('url', '')
        
        # Post to API
        try:
            response = requests.post(
                API_ENDPOINT,
                data=payload,
                timeout=15
            )
            
            response_text = response.text.strip()
            
            if response.status_code in [200, 201]:
                status_msg = f"✅ [{i+1}/{total}] SUCCESS: {prop_name[:50]}"
                live_logs.append(status_msg)
                st.session_state.api_stats['posted'] += 1
                
                # Mark as posted
                prop['_posted'] = True
                prop['_posted_at'] = datetime.now().isoformat()
                prop['_api_response'] = response_text[:200]
                
                # Add to posted list
                file_group = file_mapping.get(prop_hash)
                if file_group:
                    if file_group['current'] not in posted_by_file:
                        posted_by_file[file_group['current']] = []
                    posted_by_file[file_group['current']].append(prop)
                
                # Track globally
                st.session_state.posted_property_hashes.add(prop_hash)
                
            else:
                error_detail = response_text[:300]
                status_msg = f"❌ [{i+1}/{total}] FAILED: {prop_name[:50]} (Code {response.status_code})"
                live_logs.append(status_msg)
                st.session_state.api_stats['failed'] += 1
                
                # Add to failed
                file_group = file_mapping.get(prop_hash)
                if file_group:
                    if file_group['current'] not in failed_by_file:
                        failed_by_file[file_group['current']] = []
                    prop['_error'] = f"HTTP {response.status_code}: {error_detail}"
                    failed_by_file[file_group['current']].append(prop)
        
        except Exception as e:
            status_msg = f"⚠️ [{i+1}/{total}] Error: {prop_name[:50]} -> {str(e)[:100]}"
            live_logs.append(status_msg)
            st.session_state.api_stats['failed'] += 1
            
            # Add to failed
            file_group = file_mapping.get(prop_hash)
            if file_group:
                if file_group['current'] not in failed_by_file:
                    failed_by_file[file_group['current']] = []
                prop['_error'] = str(e)[:200]
                failed_by_file[file_group['current']].append(prop)
        
        # Update UI
        log_display = "\n".join(live_logs[-20:])
        log_placeholder.text_area("Live Logs", log_display, height=400, key=f"log_post_{scraper_key}_{i}")
        progress_bar.progress((i + 1) / total, text=f"Processing {i+1}/{total}")
    
    # Update files
    for file_group in current_files:
        current_file = file_group['current']
        posted_file = file_group['posted']
        failed_file = file_group['failed']
        metadata_file = file_group['metadata']
        
        # Load current
        with open(current_file, 'r', encoding='utf-8') as f:
            current_props = json.load(f)
        
        # Get posted/failed for this file
        posted_props = posted_by_file.get(current_file, [])
        failed_props = failed_by_file.get(current_file, [])
        
        # Remove posted/failed from current
        posted_hashes = {p.get('_hash') for p in posted_props}
        failed_hashes = {p.get('_hash') for p in failed_props}
        
        remaining = [p for p in current_props 
                    if p.get('_hash') not in posted_hashes 
                    and p.get('_hash') not in failed_hashes]
        
        # Update current.json
        with open(current_file, 'w', encoding='utf-8') as f:
            json.dump(remaining, f, indent=2, ensure_ascii=False)
        
        # Append to posted.json
        if posted_props:
            existing_posted = []
            if posted_file.exists():
                with open(posted_file, 'r', encoding='utf-8') as f:
                    existing_posted = json.load(f)
            
            existing_posted.extend(posted_props)
            
            with open(posted_file, 'w', encoding='utf-8') as f:
                json.dump(existing_posted, f, indent=2, ensure_ascii=False)
        
        # Append to failed.json
        if failed_props:
            existing_failed = []
            if failed_file.exists():
                with open(failed_file, 'r', encoding='utf-8') as f:
                    existing_failed = json.load(f)
            
            existing_failed.extend(failed_props)
            
            with open(failed_file, 'w', encoding='utf-8') as f:
                json.dump(existing_failed, f, indent=2, ensure_ascii=False)
        
        # Update metadata
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                meta = json.load(f)
        else:
            meta = {}
        
        meta['posted'] = meta.get('posted', 0) + len(posted_props)
        meta['failed'] = meta.get('failed', 0) + len(failed_props)
        meta['pending'] = len(remaining)
        meta['total_current'] = len(remaining)
        meta['last_post'] = datetime.now().isoformat()
        
        with open(metadata_file, 'w') as f:
            json.dump(meta, f, indent=2)
    
    # Save global posted hashes
    save_posted_hashes(st.session_state.posted_property_hashes)
    
    # Summary
    total_posted = sum(len(props) for props in posted_by_file.values())
    total_failed = sum(len(props) for props in failed_by_file.values())
    
    summary = (f"\n{'='*60}\n"
              f"🎉 POSTING COMPLETE - {SCRAPERS[scraper_key]['name']}\n"
              f"✅ Posted: {total_posted}\n"
              f"❌ Failed: {total_failed}\n"
              f"{'='*60}\n")
    live_logs.append(summary)
    log_placeholder.text_area("Final Results", "\n".join(live_logs[-25:]), height=400)

def retry_failed_posts(scraper_key, log_placeholder):
    """Retry posting properties from failed.json"""
    
    scraper_folder = OUTPUT_DIR / scraper_key
    
    # Collect all failed.json files
    failed_files = []
    city_folders = [f for f in scraper_folder.iterdir() if f.is_dir() and f.name != 'archive']
    
    if city_folders:
        for city_folder in city_folders:
            failed_file = city_folder / "failed.json"
            if failed_file.exists():
                failed_files.append({
                    'failed': failed_file,
                    'posted': city_folder / "posted.json",
                    'metadata': city_folder / "metadata.json",
                    'city_name': city_folder.name
                })
    else:
        failed_file = scraper_folder / "failed.json"
        if failed_file.exists():
            failed_files.append({
                'failed': failed_file,
                'posted': scraper_folder / "posted.json",
                'metadata': scraper_folder / "metadata.json",
                'city_name': 'all'
            })
    
    if not failed_files:
        st.warning("⚠️ No failed properties to retry!")
        return
    
    # Collect failed properties
    all_failed = []
    file_mapping = {}
    
    for file_group in failed_files:
        try:
            with open(file_group['failed'], 'r', encoding='utf-8') as f:
                props = json.load(f)
            
            for prop in props:
                all_failed.append(prop)
                file_mapping[prop.get('_hash', generate_property_hash(prop))] = file_group
        except:
            pass
    
    if not all_failed:
        st.info("✅ No failed properties found!")
        return
    
    total = len(all_failed)
    st.info(f"🔄 Retrying {total} failed properties...")
    
    progress_bar = st.progress(0, text=f"Retrying 0/{total}")
    live_logs = []
    
    posted_by_file = {}
    still_failed_by_file = {}
    
    for i, prop in enumerate(all_failed):
        if st.session_state.stop_requested:
            live_logs.append("🛑 Stopped by user")
            break
        
        prop_hash = prop.get('_hash', generate_property_hash(prop))
        prop_name = prop.get('propname') or prop.get('title') or 'Unknown'
        
        # Remove old error
        if '_error' in prop:
            del prop['_error']
        
        # AI mapping
        try:
            mapped_prop = ai_map_property_complete(prop, st.session_state.lookups, use_cache=True)
        except Exception as e:
            live_logs.append(f"❌ [{i+1}/{total}] Mapping failed: {prop_name}")
            file_group = file_mapping.get(prop_hash)
            if file_group:
                if file_group['failed'] not in still_failed_by_file:
                    still_failed_by_file[file_group['failed']] = []
                prop['_error'] = f"Mapping: {str(e)[:200]}"
                still_failed_by_file[file_group['failed']].append(prop)
            continue
        
        # Build payload
        payload = {k: str(v).strip() for k, v in mapped_prop.items() 
                  if not k.startswith('_') and v and str(v).strip()}
        
        if 'city_id' not in payload:
            payload['city_id'] = '1'
        if 'propname' not in payload:
            payload['propname'] = prop_name
        if 'refer_links' not in payload:
            payload['refer_links'] = prop.get('url', '')
        
        # Post
        try:
            response = requests.post(API_ENDPOINT, data=payload, timeout=15)
            
            if response.status_code in [200, 201]:
                live_logs.append(f"✅ [{i+1}/{total}] RETRY SUCCESS: {prop_name[:50]}")
                
                prop['_posted'] = True
                prop['_posted_at'] = datetime.now().isoformat()
                prop['_retried'] = True
                
                file_group = file_mapping.get(prop_hash)
                if file_group:
                    if file_group['failed'] not in posted_by_file:
                        posted_by_file[file_group['failed']] = []
                    posted_by_file[file_group['failed']].append(prop)
                
                st.session_state.posted_property_hashes.add(prop_hash)
            else:
                live_logs.append(f"❌ [{i+1}/{total}] Still failed: {prop_name[:50]}")
                
                file_group = file_mapping.get(prop_hash)
                if file_group:
                    if file_group['failed'] not in still_failed_by_file:
                        still_failed_by_file[file_group['failed']] = []
                    prop['_error'] = f"HTTP {response.status_code}"
                    still_failed_by_file[file_group['failed']].append(prop)
        
        except Exception as e:
            live_logs.append(f"⚠️ [{i+1}/{total}] Error: {str(e)[:50]}")
            file_group = file_mapping.get(prop_hash)
            if file_group:
                if file_group['failed'] not in still_failed_by_file:
                    still_failed_by_file[file_group['failed']] = []
                prop['_error'] = str(e)[:200]
                still_failed_by_file[file_group['failed']].append(prop)
        
        log_display = "\n".join(live_logs[-20:])
        log_placeholder.text_area("Retry Logs", log_display, height=400, key=f"retry_{scraper_key}_{i}")
        progress_bar.progress((i + 1) / total, text=f"Retrying {i+1}/{total}")
    
    # Update files
    for file_group in failed_files:
        failed_file = file_group['failed']
        posted_file = file_group['posted']
        metadata_file = file_group['metadata']
        
        posted_props = posted_by_file.get(failed_file, [])
        still_failed = still_failed_by_file.get(failed_file, [])
        
        # Update failed.json
        with open(failed_file, 'w', encoding='utf-8') as f:
            json.dump(still_failed, f, indent=2, ensure_ascii=False)
        
        # Append to posted.json
        if posted_props:
            existing_posted = []
            if posted_file.exists():
                with open(posted_file, 'r', encoding='utf-8') as f:
                    existing_posted = json.load(f)
            
            existing_posted.extend(posted_props)
            
            with open(posted_file, 'w', encoding='utf-8') as f:
                json.dump(existing_posted, f, indent=2, ensure_ascii=False)
        
        # Update metadata
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                meta = json.load(f)
            
            meta['posted'] = meta.get('posted', 0) + len(posted_props)
            meta['failed'] = len(still_failed)
            
            with open(metadata_file, 'w') as f:
                json.dump(meta, f, indent=2)
    
    save_posted_hashes(st.session_state.posted_property_hashes)
    
    total_success = sum(len(props) for props in posted_by_file.values())
    total_still_failed = sum(len(props) for props in still_failed_by_file.values())
    
    summary = (f"\n{'='*60}\n"
              f"🔄 RETRY COMPLETE\n"
              f"✅ Now Posted: {total_success}\n"
              f"❌ Still Failed: {total_still_failed}\n"
              f"{'='*60}\n")
    live_logs.append(summary)
    log_placeholder.text_area("Retry Results", "\n".join(live_logs[-25:]), height=400)

def post_all_sources_to_api(batch_size, log_placeholder):
    """Post properties from ALL sources to API"""
    
    live_logs = []
    live_logs.append("🚀 Starting global posting across all sources...")
    log_placeholder.text_area("Global Posting Logs", "\n".join(live_logs), height=400)
    
    # Get all sources with pending properties
    all_stats = get_scraper_statistics()
    sources_with_pending = []
    
    for key, stats in all_stats.items():
        pending = stats['extracted'] - stats['posted'] - stats['failed']
        if pending > 0:
            sources_with_pending.append({
                'key': key,
                'name': SCRAPERS[key]['name'],
                'pending': pending
            })
    
    if not sources_with_pending:
        live_logs.append("✅ No pending properties found!")
        log_placeholder.text_area("Global Posting Logs", "\n".join(live_logs), height=400)
        return
    
    live_logs.append(f"\n📊 Found {len(sources_with_pending)} sources with pending properties:")
    for src in sources_with_pending:
        live_logs.append(f"   - {src['name']}: {src['pending']} pending")
    
    log_placeholder.text_area("Global Posting Logs", "\n".join(live_logs), height=400)
    
    # Calculate per-source batch size
    total_pending = sum(src['pending'] for src in sources_with_pending)
    
    # Distribute batch size proportionally
    remaining_batch = batch_size
    
    for src in sources_with_pending:
        if st.session_state.stop_requested:
            live_logs.append("\n🛑 Stopped by user")
            break
        
        # Calculate this source's share
        proportion = src['pending'] / total_pending
        source_batch = max(1, int(batch_size * proportion))
        source_batch = min(source_batch, remaining_batch, src['pending'])
        
        if source_batch <= 0:
            continue
        
        live_logs.append(f"\n{'='*60}")
        live_logs.append(f"📤 Posting {src['name']}: {source_batch} properties")
        live_logs.append(f"{'='*60}")
        log_placeholder.text_area("Global Posting Logs", "\n".join(live_logs[-30:]), height=400)
        
        # Create a temporary log placeholder for this source
        source_log_placeholder = st.empty()
        
        # Post this source
        post_source_to_api(src['key'], source_batch, source_log_placeholder)
        
        remaining_batch -= source_batch
        
        live_logs.append(f"✅ {src['name']} completed")
        log_placeholder.text_area("Global Posting Logs", "\n".join(live_logs[-30:]), height=400)
        
        if remaining_batch <= 0:
            break
    
    # Final summary
    final_stats = get_scraper_statistics()
    total_posted_now = sum(stats['posted'] for stats in final_stats.values())
    total_failed_now = sum(stats['failed'] for stats in final_stats.values())
    
    live_logs.append(f"\n{'='*60}")
    live_logs.append(f"🎉 GLOBAL POSTING COMPLETE")
    live_logs.append(f"{'='*60}")
    live_logs.append(f"✅ Total Posted (All Time): {total_posted_now:,}")
    live_logs.append(f"❌ Total Failed (All Time): {total_failed_now:,}")
    live_logs.append(f"{'='*60}")
    
    log_placeholder.text_area("Global Posting Results", "\n".join(live_logs[-40:]), height=500)

def retry_all_failed_posts(log_placeholder):
    """Retry failed posts from ALL sources"""
    
    live_logs = []
    live_logs.append("🔄 Starting global retry across all sources...")
    log_placeholder.text_area("Global Retry Logs", "\n".join(live_logs), height=400)
    
    # Get all sources with failed properties
    all_stats = get_scraper_statistics()
    sources_with_failed = []
    
    for key, stats in all_stats.items():
        if stats['failed'] > 0:
            sources_with_failed.append({
                'key': key,
                'name': SCRAPERS[key]['name'],
                'failed': stats['failed']
            })
    
    if not sources_with_failed:
        live_logs.append("✅ No failed properties found!")
        log_placeholder.text_area("Global Retry Logs", "\n".join(live_logs), height=400)
        return
    
    live_logs.append(f"\n📊 Found {len(sources_with_failed)} sources with failed properties:")
    for src in sources_with_failed:
        live_logs.append(f"   - {src['name']}: {src['failed']} failed")
    
    log_placeholder.text_area("Global Retry Logs", "\n".join(live_logs), height=400)
    
    # Retry each source
    for src in sources_with_failed:
        if st.session_state.stop_requested:
            live_logs.append("\n🛑 Stopped by user")
            break
        
        live_logs.append(f"\n{'='*60}")
        live_logs.append(f"🔄 Retrying {src['name']}: {src['failed']} properties")
        live_logs.append(f"{'='*60}")
        log_placeholder.text_area("Global Retry Logs", "\n".join(live_logs[-30:]), height=400)
        
        # Create temporary log placeholder
        source_log_placeholder = st.empty()
        
        # Retry this source
        retry_failed_posts(src['key'], source_log_placeholder)
        
        live_logs.append(f"✅ {src['name']} retry completed")
        log_placeholder.text_area("Global Retry Logs", "\n".join(live_logs[-30:]), height=400)
    
    # Final summary
    final_stats = get_scraper_statistics()
    total_failed_remaining = sum(stats['failed'] for stats in final_stats.values())
    
    live_logs.append(f"\n{'='*60}")
    live_logs.append(f"🔄 GLOBAL RETRY COMPLETE")
    live_logs.append(f"{'='*60}")
    live_logs.append(f"❌ Remaining Failed: {total_failed_remaining:,}")
    live_logs.append(f"{'='*60}")
    
    log_placeholder.text_area("Global Retry Results", "\n".join(live_logs[-40:]), height=500)

def archive_posted_properties_weekly():
    """Archive posted properties to compressed weekly files"""
    
    archive_summary = []
    archive_summary.append("📦 Starting weekly archive process...")
    
    # Get current week identifier
    from datetime import datetime, timedelta
    today = datetime.now()
    week_start = today - timedelta(days=today.weekday())
    week_id = week_start.strftime("%Y-W%U")  # e.g., "2024-W05"
    
    archive_summary.append(f"📅 Archive for week: {week_id}")
    
    total_archived = 0
    
    for key in SCRAPERS.keys():
        scraper_folder = OUTPUT_DIR / key
        
        if not scraper_folder.exists():
            continue
        
        scraper_name = SCRAPERS[key]['name']
        archive_summary.append(f"\n🔍 Processing {scraper_name}...")
        
        # Find all posted.json files
        city_folders = [f for f in scraper_folder.iterdir() if f.is_dir() and f.name != 'archive']
        
        posted_files = []
        
        if city_folders:
            # Multi-city structure
            for city_folder in city_folders:
                posted_file = city_folder / "posted.json"
                archive_dir = city_folder / "archive"
                
                if posted_file.exists():
                    posted_files.append({
                        'posted': posted_file,
                        'archive_dir': archive_dir,
                        'metadata': city_folder / "metadata.json",
                        'location': f"{scraper_name}/{city_folder.name}"
                    })
        else:
            # Single file structure
            posted_file = scraper_folder / "posted.json"
            archive_dir = scraper_folder / "archive"
            
            if posted_file.exists():
                posted_files.append({
                    'posted': posted_file,
                    'archive_dir': archive_dir,
                    'metadata': scraper_folder / "metadata.json",
                    'location': scraper_name
                })
        
        # Archive each posted.json
        for file_info in posted_files:
            try:
                with open(file_info['posted'], 'r', encoding='utf-8') as f:
                    posted_props = json.load(f)
                
                if not posted_props:
                    archive_summary.append(f"   ⏭️ {file_info['location']}: No properties to archive")
                    continue
                
                # Create archive directory
                file_info['archive_dir'].mkdir(exist_ok=True)
                
                # Archive filename
                archive_file = file_info['archive_dir'] / f"{week_id}.json.gz"
                
                # If archive already exists, append to it
                existing_archived = []
                if archive_file.exists():
                    try:
                        with gzip.open(archive_file, 'rt', encoding='utf-8') as f:
                            existing_archived = json.load(f)
                    except:
                        existing_archived = []
                
                # Merge
                all_archived = existing_archived + posted_props
                
                # Remove duplicates by hash
                seen_hashes = set()
                unique_archived = []
                for prop in all_archived:
                    prop_hash = prop.get('_hash', generate_property_hash(prop))
                    if prop_hash not in seen_hashes:
                        seen_hashes.add(prop_hash)
                        unique_archived.append(prop)
                
                # Compress and save
                with gzip.open(archive_file, 'wt', encoding='utf-8') as f:
                    json.dump(unique_archived, f, indent=2, ensure_ascii=False)
                
                # Clear posted.json
                with open(file_info['posted'], 'w', encoding='utf-8') as f:
                    json.dump([], f)
                
                # Update metadata
                if file_info['metadata'].exists():
                    with open(file_info['metadata'], 'r') as f:
                        meta = json.load(f)
                    
                    meta['posted'] = 0
                    meta['last_archive'] = datetime.now().isoformat()
                    meta['last_archive_week'] = week_id
                    
                    with open(file_info['metadata'], 'w') as f:
                        json.dump(meta, f, indent=2)
                
                total_archived += len(posted_props)
                archive_summary.append(f"   ✅ {file_info['location']}: Archived {len(posted_props)} properties → {archive_file.name}")
                
            except Exception as e:
                archive_summary.append(f"   ❌ {file_info['location']}: Error - {str(e)[:100]}")
    
    archive_summary.append(f"\n{'='*60}")
    archive_summary.append(f"📦 ARCHIVE COMPLETE")
    archive_summary.append(f"{'='*60}")
    archive_summary.append(f"✅ Total properties archived: {total_archived:,}")
    archive_summary.append(f"📅 Archive week: {week_id}")
    archive_summary.append(f"{'='*60}")
    
    return "\n".join(archive_summary)

def get_global_statistics():
    """Get overall statistics from new folder structure"""
    
    all_stats = get_scraper_statistics()
    
    total_extracted = sum(stats['extracted'] for stats in all_stats.values())
    total_posted = sum(stats['posted'] for stats in all_stats.values())
    total_failed = sum(stats['failed'] for stats in all_stats.values())
    total_pending = sum(stats['pending'] for stats in all_stats.values())
    
    # Calculate duplicates (estimated from scraper metadata)
    total_duplicates = 0
    for key in SCRAPERS.keys():
        scraper_folder = OUTPUT_DIR / key
        
        if not scraper_folder.exists():
            continue
        
        city_folders = [f for f in scraper_folder.iterdir() if f.is_dir() and f.name != 'archive']
        
        if city_folders:
            for city_folder in city_folders:
                metadata_file = city_folder / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            meta = json.load(f)
                        total_duplicates += meta.get('skipped_duplicates', 0)
                    except:
                        pass
        else:
            metadata_file = scraper_folder / "metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        meta = json.load(f)
                    total_duplicates += meta.get('skipped_duplicates', 0)
                except:
                    pass
    
    return {
        'total_extracted': total_extracted,
        'total_unique': total_extracted,  # Already deduplicated in scrapers
        'total_duplicates': total_duplicates,
        'total_posted': total_posted,
        'total_failed': total_failed,
        'total_pending': total_pending,
        'duplicate_rate': round((total_duplicates / (total_extracted + total_duplicates) * 100) if (total_extracted + total_duplicates) > 0 else 0, 1)
    }

def render_global_metrics():
    """Render top-level dashboard metrics"""
    st.markdown("### 📊 Overall Statistics")
    
    global_stats = get_global_statistics()
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric(
            "Total Properties", 
            f"{global_stats['total_extracted']:,}",
            help="Total unique properties across all sources"
        )
    
    with col2:
        st.metric(
            "Posted to API", 
            f"{global_stats['total_posted']:,}",
            help="Successfully posted to database"
        )
    
    with col3:
        st.metric(
            "Pending Upload", 
            f"{global_stats['total_pending']:,}",
            help="Not yet posted to API"
        )
    
    with col4:
        st.metric(
            "Failed Posts", 
            f"{global_stats['total_failed']:,}",
            delta=f"{round((global_stats['total_failed'] / global_stats['total_extracted'] * 100) if global_stats['total_extracted'] > 0 else 0, 1)}%",
            delta_color="inverse",
            help="Failed to post to API"
        )
    
    with col5:
        st.metric(
            "Duplicates Skipped", 
            f"{global_stats['total_duplicates']:,}",
            delta=f"{global_stats['duplicate_rate']}%",
            delta_color="inverse",
            help="Duplicates prevented by scrapers"
        )
    
    with col6:
        completion = round((global_stats['total_posted'] / global_stats['total_extracted'] * 100) if global_stats['total_extracted'] > 0 else 0, 1)
        st.metric(
            "Completion Rate",
            f"{completion}%",
            help="Percentage of properties posted"
        )
    
    st.markdown("---")

def get_scraper_city_distribution(scraper_key):
    """Get city-wise property distribution - NEW FOLDER STRUCTURE"""
    scraper_folder = OUTPUT_DIR / scraper_key
    scraper_name = SCRAPERS[scraper_key]['name']
    
    print(f"\n{'='*60}")
    print(f"🔍 Analyzing City Distribution for: {scraper_name}")
    
    city_dist = {}
    total_properties = 0
    
    if not scraper_folder.exists():
        print(f"   ⚠️ Folder not found: {scraper_folder}")
        return {}, 0
    
    # Check for city subfolders
    city_folders = [f for f in scraper_folder.iterdir() if f.is_dir() and f.name != 'archive']
    
    if city_folders:
        # Multi-city structure
        print(f"   📂 Found {len(city_folders)} city folders")
        
        for city_folder in city_folders:
            city_name = city_folder.name.replace('_', ' ').title()
            metadata_file = city_folder / "metadata.json"
            
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        meta = json.load(f)
                    
                    count = meta.get('total_current', 0)
                    city_dist[city_name] = count
                    total_properties += count
                    
                    print(f"   ✅ {city_name}: {count} properties")
                except Exception as e:
                    print(f"   ⚠️ Error reading {city_name}: {e}")
    else:
        # Single file structure (no city breakdown)
        metadata_file = scraper_folder / "metadata.json"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    meta = json.load(f)
                
                count = meta.get('total_current', 0)
                city_dist['All Cities'] = count
                total_properties = count
                
                print(f"   ✅ All Cities: {count} properties")
            except Exception as e:
                print(f"   ⚠️ Error reading metadata: {e}")
    
    print(f"\n📊 Total Properties: {total_properties}")
    print(f"🌆 Cities: {len(city_dist)}")
    print(f"{'='*60}\n")
    
    return dict(sorted(city_dist.items(), key=lambda x: x[1], reverse=True)), total_properties


def render_enhanced_scraper_card(key, info):
    """Enhanced scraper card with improved city distribution display"""
    status = st.session_state.scraper_status[key]
    scraper_stats = get_scraper_statistics().get(key, {
        'extracted': 0, 'posted': 0, 'duplicates': 0
    })
    
    status_config = {
        'idle': {'icon': '⚪', 'color': '#808080', 'border': '#B0BEC5', 'bg': '#FAFAFA', 'text': 'Ready'},
        'running': {'icon': '🟡', 'color': '#FF9800', 'border': '#FFB74D', 'bg': '#FFF8E1', 'text': 'Running'},
        'completed': {'icon': '🟢', 'color': '#4CAF50', 'border': '#81C784', 'bg': '#E8F5E9', 'text': 'Completed'},
        'error': {'icon': '🔴', 'color': '#F44336', 'border': '#E57373', 'bg': '#FFEBEE', 'text': 'Error'}
    }
    
    current = status_config.get(status['status'], status_config['idle'])
    
    # Inject CSS
    st.markdown(f"""
        <style>
        .city-dist-{key} {{
            background: linear-gradient(145deg, #F8F9FA, white);
            border: 2px solid {current['border']};
            border-radius: 12px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 4px 12px {current['border']}20;
            max-height: 400px;
            overflow-y: auto;
        }}
        
        .city-dist-{key}::-webkit-scrollbar {{
            width: 8px;
        }}
        
        .city-dist-{key}::-webkit-scrollbar-track {{
            background: #f1f1f1;
            border-radius: 10px;
        }}
        
        .city-dist-{key}::-webkit-scrollbar-thumb {{
            background: {current['border']};
            border-radius: 10px;
        }}
        
        .city-dist-{key}::-webkit-scrollbar-thumb:hover {{
            background: {current['color']};
        }}
        </style>
    """, unsafe_allow_html=True)
    
    # Use native Streamlit container with border=True
    with st.container(border=True):
        # HEADER SECTION
        col_icon, col_name, col_status = st.columns([1, 6, 2])
        
        with col_icon:
            st.markdown(
                f"<div style='font-size: 3.5em; text-align: center; "
                f"filter: drop-shadow(0 4px 8px {current['color']}40);'>{info['icon']}</div>", 
                unsafe_allow_html=True
            )
        
        with col_name:
            st.markdown(f"### {info['name']}")
        
        with col_status:
            st.markdown(
                f"<div style='background: linear-gradient(135deg, {current['color']}, {current['border']}); "
                f"color: white; padding: 10px 20px; border-radius: 20px; font-weight: bold; "
                f"font-size: 0.85em; text-align: center; box-shadow: 0 4px 12px {current['color']}40;'>"
                f"{current['icon']} {current['text'].upper()}</div>",
                unsafe_allow_html=True
            )
        
        st.markdown(f'<hr style="height: 2px; background: linear-gradient(90deg, transparent, {current["border"]}, transparent); margin: 20px 0; border: none;">', unsafe_allow_html=True)
        
       # METRICS SECTION - 5 COLUMNS NOW
        m1, m2, m3, m4, m5 = st.columns(5)
        
        pending = scraper_stats['extracted'] - scraper_stats['posted'] - scraper_stats['failed']
        
        metrics_data = [
            ("📊 Extracted", scraper_stats['extracted'], "#2196F3"),
            ("✅ Posted", scraper_stats['posted'], "#4CAF50"),
            ("⏳ Pending", pending, "#FF9800"),
            ("❌ Fail To Post", scraper_stats['failed'], "#F44336"),
            ("🔄 Duplicates", scraper_stats.get('duplicates', 0), "#9E9E9E")
        ]
        
        for col, (label, value, color) in zip([m1, m2, m3, m4, m5], metrics_data):
            
            with col:
                st.markdown(
                    f"<div style='text-align: center; padding: 16px; background: white; "
                    f"border-radius: 12px; border: 2px solid {current['border']}40; "
                    f"box-shadow: 0 2px 8px {current['border']}20;'>"
                    f"<div style='color: #666; font-size: 0.75em; font-weight: 600; "
                    f"text-transform: uppercase; margin-bottom: 8px;'>{label}</div>"
                    f"<div style='font-size: 1.8em; font-weight: bold; color: {color};'>{value:,}</div>"
                    "</div>",
                    unsafe_allow_html=True
                )
        
        st.markdown(f'<hr style="height: 2px; background: linear-gradient(90deg, transparent, {current["border"]}, transparent); margin: 20px 0; border: none;">', unsafe_allow_html=True)
        
       # ACTION BUTTONS SECTION - 3 COLUMNS
        btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
        
        with btn_col1:
            if status['status'] == 'running':
                if st.button(
                    "🛑 Stop Scraper", 
                    key=f"stop_btn_{key}", 
                    use_container_width=True, 
                    type="secondary"
                ):
                    st.session_state.stop_requested = True
                    st.warning("⚠️ Stop signal sent...")
            else:
                if st.button(
                    "▶️ Run Scraper", 
                    key=f"run_btn_{key}", 
                    use_container_width=True, 
                    type="primary"
                ):
                    if trigger_scraper(key):
                        st.success("✅ Started!")
                        time.sleep(0.3)
                        st.rerun()
        
        # NEW: Post to API button
        with btn_col2:
            pending = scraper_stats['extracted'] - scraper_stats['posted'] - scraper_stats['failed']
            
            if pending > 0:
                if st.button(
                    f"📤 Post to API ({pending})",
                    key=f"post_btn_{key}",
                    use_container_width=True,
                    type="primary"
                ):
                    st.session_state[f"show_post_{key}"] = True
                    st.rerun()
        
        with btn_col3:
            if scraper_stats['extracted'] > 0:
                if f"show_city_{key}" not in st.session_state:
                    st.session_state[f"show_city_{key}"] = False
                
                is_showing = st.session_state[f"show_city_{key}"]
                btn_label = "✖ Close Distribution" if is_showing else "🌆 City Distribution"
                btn_type = "secondary" if is_showing else "primary"
                
                if st.button(
                    btn_label, 
                    key=f"city_btn_{key}", 
                    use_container_width=True,
                    type=btn_type
                ):
                    st.session_state[f"show_city_{key}"] = not is_showing
                    st.rerun()
        
        # NEW: Posting Interface (appears when Post to API is clicked)
        if st.session_state.get(f"show_post_{key}", False):
            st.markdown(f'<hr style="height: 2px; background: linear-gradient(90deg, transparent, {current["border"]}, transparent); margin: 20px 0; border: none;">', unsafe_allow_html=True)
            
            st.markdown("#### 📤 Post Properties to API")
            
            post_col1, post_col2, post_col3 = st.columns([2, 1, 1])
            
            with post_col1:
                batch_size = st.number_input(
                    "Number of properties to post",
                    min_value=1,
                    max_value=pending,
                    value=min(10, pending),
                    key=f"batch_size_{key}"
                )
            
            with post_col2:
                if st.button("🚀 Start Posting", key=f"start_post_{key}", use_container_width=True, type="primary"):
                    st.session_state.stop_requested = False
                    log_placeholder = st.empty()
                    post_source_to_api(key, batch_size, log_placeholder)
                    st.success("✅ Posting complete!")
                    time.sleep(1)
                    st.session_state[f"show_post_{key}"] = False
                    st.rerun()
            
            with post_col3:
                if st.button("✖ Cancel", key=f"cancel_post_{key}", use_container_width=True):
                    st.session_state[f"show_post_{key}"] = False
                    st.rerun()
        
        # NEW: Retry Failed Button (if failed > 0)
        if scraper_stats['failed'] > 0:
            st.markdown(f'<hr style="height: 2px; background: linear-gradient(90deg, transparent, {current["border"]}, transparent); margin: 20px 0; border: none;">', unsafe_allow_html=True)
            
            retry_col1, retry_col2 = st.columns([3, 1])
            
            with retry_col1:
                st.warning(f"⚠️ {scraper_stats['failed']} properties failed to post")
            
            with retry_col2:
                if st.button(
                    f"🔄 Retry Failed ({scraper_stats['failed']})",
                    key=f"retry_btn_{key}",
                    use_container_width=True,
                    type="secondary"
                ):
                    st.session_state[f"show_retry_{key}"] = True
                    st.rerun()
            
            # Retry Interface
            if st.session_state.get(f"show_retry_{key}", False):
                st.markdown("#### 🔄 Retry Failed Posts")
                
                retry_btn_col1, retry_btn_col2 = st.columns([1, 1])
                
                with retry_btn_col1:
                    if st.button("🚀 Start Retry", key=f"start_retry_{key}", use_container_width=True, type="primary"):
                        st.session_state.stop_requested = False
                        log_placeholder = st.empty()
                        retry_failed_posts(key, log_placeholder)
                        st.success("✅ Retry complete!")
                        time.sleep(1)
                        st.session_state[f"show_retry_{key}"] = False
                        st.rerun()
                
                with retry_btn_col2:
                    if st.button("✖ Cancel", key=f"cancel_retry_{key}", use_container_width=True):
                        st.session_state[f"show_retry_{key}"] = False
                        st.rerun()

        # CITY DISTRIBUTION SECTION (CONDITIONAL)
        if st.session_state.get(f"show_city_{key}", False):
            st.markdown(f'<hr style="height: 2px; background: linear-gradient(90deg, transparent, {current["border"]}, transparent); margin: 20px 0; border: none;">', unsafe_allow_html=True)
            
            with st.spinner(f"📊 Loading city distribution for {info['name']}..."):
                try:
                    result = get_scraper_city_distribution(key)
                    if isinstance(result, tuple) and len(result) == 2:
                        city_dist, file_total = result
                    else:
                        city_dist = result if isinstance(result, dict) else {}
                        file_total = sum(city_dist.values()) if city_dist else 0
                except Exception as e:
                    st.error(f"Error loading distribution: {e}")
                    city_dist = {}
                    file_total = 0
            
            if city_dist and file_total > 0:
                display_dist = {k: v for k, v in city_dist.items() if k != 'Unknown'} if len(city_dist) > 1 else city_dist
                
                if display_dist:
                    st.markdown(f'<div class="city-dist-{key}">', unsafe_allow_html=True)
                    
                    st.markdown(f"#### 🌆 City-wise Distribution - {info['name']}")
                    
                    total_shown = sum(display_dist.values())
                    col_info1, col_info2 = st.columns(2)
                    
                    with col_info1:
                        st.metric("📂 Total in Files", f"{file_total:,}")
                    
                    with col_info2:
                        percentage = round((total_shown/file_total*100) if file_total > 0 else 0, 1)
                        st.metric("🏙️ With City Info", f"{total_shown:,}", delta=f"{percentage}%")
                    
                    city_data = []
                    for city, count in display_dist.items():
                        pct = round((count / total_shown * 100) if total_shown > 0 else 0, 2)
                        city_data.append({
                            '🏙️ City': city,
                            '📊 Properties': f"{count:,}",
                            '📈 Percentage': f"{pct}%"
                        })
                    
                    df_cities = pd.DataFrame(city_data)
                    st.dataframe(
                        df_cities,
                        use_container_width=True,
                        height=min(280, len(df_cities) * 35 + 38),
                        hide_index=True
                    )
                    
                    if 'Unknown' in city_dist and 'Unknown' not in display_dist:
                        st.caption(f"ℹ️ Note: {city_dist['Unknown']:,} properties with unknown cities were excluded")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # LOGS SECTION (COLLAPSIBLE)
        if status['status'] in ['running', 'completed', 'error'] and status['log']:
            st.markdown(f'<hr style="height: 2px; background: linear-gradient(90deg, transparent, {current["border"]}, transparent); margin: 20px 0; border: none;">', unsafe_allow_html=True)
            
            with st.expander("📋 View Logs", expanded=(status['status'] == 'running')):
                log_lines = status['log'][-50:]
                log_text = '\n'.join(log_lines)
                st.code(log_text, language='log', line_numbers=False)
                st.caption(f"📊 Showing last 50 of {len(status['log'])} total lines")
                
                if len(status['log']) > 50:
                    full_log = '\n'.join(status['log'])
                    st.download_button(
                        label="💾 Download Full Log",
                        data=full_log,
                        file_name=f"{key}_scraper_log.txt",
                        mime="text/plain",
                        key=f"download_log_{key}"
                    )

def render_enhanced_scrapers_section():
    """Enhanced scraper dashboard with card layout"""
    
    # Global metrics at the top
    render_global_metrics()
    
    # Process background messages
    msg_count = process_messages()
    
    # Check if any scraper is running
    any_running = any(
        st.session_state.scraper_status[k]['status'] == 'running' 
        for k in SCRAPERS
    )
    
    if any_running:
        st.info(f"🔄 Live Updates | Messages processed: {msg_count}")
    
    # Section header
    st.markdown("### 🔧 Individual Scrapers")
    st.markdown("")  # Add spacing
    
    # Get all scrapers
    scrapers = list(SCRAPERS.items())
    
    # Render scrapers in 2-column grid
    for i in range(0, len(scrapers), 2):
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            if i < len(scrapers):
                render_enhanced_scraper_card(scrapers[i][0], scrapers[i][1])
        
        with col2:
            if i + 1 < len(scrapers):
                render_enhanced_scraper_card(scrapers[i+1][0], scrapers[i+1][1])
    
    # Auto-refresh while any scraper is running
    if any_running:
        time.sleep(0.5)
        st.rerun()

def render_data_statistics_section():
    """Render city distribution - NO LOAD BUTTON, auto-loads from files"""
    
    st.markdown("---")
    st.subheader("📊 Overall City Distribution")
    
    # Auto-load if not loaded
    if not st.session_state.unique_props:
        with st.spinner("📂 Loading property data from files..."):
            get_global_statistics()  # This will populate unique_props and by_city
    
    by_c = st.session_state.by_city
    uni_p = st.session_state.unique_props
    
    if by_c:
        valid_cities = {k: v for k, v in by_c.items() 
                      if k and str(k).strip() and k != 'Unknown' and k.lower() != 'none'}
        
        if valid_cities:
            sorted_cities = sorted(valid_cities.items(), key=lambda x: x[1], reverse=True)
            
            city_data = []
            for city, count in sorted_cities:
                city_str = str(city).strip()
                percentage = round((count / len(uni_p) * 100) if len(uni_p) > 0 else 0, 2)
                city_data.append({
                    '🏙️ City': city_str,
                    '📊 Count': f"{count:,}",
                    '📈 Percentage': f"{percentage}%"
                })
            
            df_cities = pd.DataFrame(city_data)
            
            st.dataframe(
                df_cities,
                use_container_width=True,
                height=min(400, len(df_cities) * 35 + 38),
                hide_index=True
            )
            
            st.caption(f"📊 Total: {len(valid_cities)} cities | {len(uni_p):,} unique properties")
            
            if st.checkbox("Show Top 10 Cities Chart", key="show_city_chart"):
                top_10 = sorted_cities[:10]
                chart_data = pd.DataFrame({
                    'City': [city for city, _ in top_10],
                    'Properties': [count for _, count in top_10]
                })
                st.bar_chart(chart_data.set_index('City'))
        else:
            st.info("ℹ️ No city information available in loaded properties")
    else:
        st.info("ℹ️ No city distribution data available")
def trigger_scraper(key):
    """Trigger scraper with enhanced validation"""
    if st.session_state.scraper_status[key]['status'] == 'running':
        st.warning(f"{SCRAPERS[key]['name']} is already running!")
        return False
    
    scraper_file = SCRAPERS[key]['file']
    if not os.path.exists(scraper_file):
        st.error(f"❌ Scraper file not found: {scraper_file}")
        return False
    
    # Check if Chrome is available (for Selenium scrapers)
    if 'nobroker' in key.lower():
        try:
            chrome_check = subprocess.run(
                ['google-chrome', '--version'] if platform.system() == 'Linux' else ['chrome', '--version'],
                capture_output=True,
                timeout=5
            )
            if chrome_check.returncode != 0:
                st.warning("⚠️ Chrome browser may not be installed. Scraper might fail.")
        except:
            st.warning("⚠️ Could not verify Chrome installation. Scraper might fail.")
    
    st.session_state.scraper_status[key]['log'] = []
    st.session_state.scraper_status[key]['status'] = 'running'
    
    thread = threading.Thread(
        target=run_scraper, 
        args=(key, st.session_state.msg_queue), 
        daemon=True
    )
    thread.start()
    
    return True


def count_properties_in_file(scraper_key):
    """Count properties extracted by a specific scraper - FIXED VERSION"""
    scraper = SCRAPERS[scraper_key]
    
    # Get scraper name to match against loaded data
    scraper_name = scraper['name']
    
    # Count from loaded data instead of files
    count = 0
    if st.session_state.all_props:
        for prop in st.session_state.all_props:
            if prop.get('_source') == scraper_name:
                count += 1
    
    return count


# ===========================================================================
def test_scraper_execution():
    """Test if scraper can be executed - FIXED VERSION"""
    import subprocess
    import sys
    
    test_file = 'magicbricks_scraper.py'
    
    st.write("### 🧪 Testing Scraper Execution")
    
    # Check if file exists
    if not os.path.exists(test_file):
        st.error(f"❌ File not found: {test_file}")
        return
    
    st.success(f"✅ File found: {test_file}")
    
    # FIXED: Use sys.executable instead of 'python'
    python_cmd = sys.executable
    
    try:
        # Test 1: Quick syntax check
        cmd = [python_cmd, '-m', 'py_compile', test_file]
        st.code(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            st.success("✅ Syntax check passed")
        else:
            st.error(f"❌ Syntax error: {result.stderr}")
            return
        
        # Test 2: Check imports
        st.write("**Testing imports...**")
        cmd = [python_cmd, '-c', '''
import sys
sys.path.insert(0, ".")
try:
    import requests
    import selenium
    print("SUCCESS: All imports available")
except Exception as e:
    print(f"ERROR: {e}")
''']
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=5
        )
        
        st.write("**Output:**")
        st.code(result.stdout)
        
        if "SUCCESS" in result.stdout:
            st.success("✅ All required packages available!")
        else:
            st.error("❌ Package import failed")
            st.code(result.stderr)
            
    except Exception as e:
        st.error(f"❌ Error: {e}")
def diagnose_python_environment():
    """Diagnose Python environment and package availability"""
    import sys
    import subprocess
    
    st.write("### 🔍 Python Environment Diagnostics")
    
    # 1. Current Python (Streamlit is using)
    st.write("**1. Streamlit's Python:**")
    st.code(sys.executable)
    
    # 2. Check if packages are installed
    st.write("**2. Installed Packages Check:**")
    try:
        import requests
        st.success("✅ requests installed")
    except:
        st.error("❌ requests NOT installed")
    
    try:
        import selenium
        st.success("✅ selenium installed")
    except:
        st.error("❌ selenium NOT installed")
    
    try:
        import pandas
        st.success("✅ pandas installed")
    except:
        st.error("❌ pandas NOT installed")
    
    # 3. Python path
    st.write("**3. Python Search Paths:**")
    st.code('\n'.join(sys.path[:5]))
    
    # 4. Test subprocess with same Python
    st.write("**4. Test Subprocess Command:**")
    test_cmd = f'"{sys.executable}" -c "import requests; print(\'SUCCESS\')"'
    st.code(test_cmd)
    
    try:
        result = subprocess.run(
            [sys.executable, '-c', 'import requests; print("SUCCESS")'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            st.success(f"✅ Subprocess test passed: {result.stdout.strip()}")
        else:
            st.error(f"❌ Subprocess failed: {result.stderr}")
    except Exception as e:
        st.error(f"❌ Test failed: {e}")
    
    # 5. Virtual environment check
    st.write("**5. Virtual Environment:**")
    venv = os.environ.get('VIRTUAL_ENV', 'Not detected')
    st.code(f"VIRTUAL_ENV = {venv}")
    
    if 'env' in sys.executable.lower() or 'venv' in sys.executable.lower():
        st.success("✅ Running in virtual environment")
    else:
        st.warning("⚠️ May not be in virtual environment")
def apply_custom_css():
    """Fix font zoom and styling issues on Linux"""
    st.markdown("""
        <style>
        /* Fix font zoom issues */
        html, body, [class*="css"] {
            font-size: 14px !important;
        }
        
        h1 {
            font-size: 2.5rem !important;
        }
        
        h2 {
            font-size: 2rem !important;
        }
        
        h3 {
            font-size: 1.5rem !important;
        }
        
        /* Fix metric text size */
        [data-testid="stMetricValue"] {
            font-size: 1.8rem !important;
        }
        
        [data-testid="stMetricLabel"] {
            font-size: 0.9rem !important;
        }
        
        /* Fix button text */
        .stButton button {
            font-size: 0.9rem !important;
        }
        
        /* Fix dataframe */
        .dataframe {
            font-size: 0.85rem !important;
        }
        </style>
    """, unsafe_allow_html=True)
def list_all_output_files():
    """Show all files in output directory"""
    st.write("### 📁 Output Directory Contents")
    st.write(f"**Path:** `{OUTPUT_DIR}`")
    
    try:
        all_files = sorted(OUTPUT_DIR.glob('*.json'))
        
        if not all_files:
            st.error("❌ No JSON files found!")
            return
        
        st.success(f"✅ Found {len(all_files)} JSON files")
        
        excluded = {
            'info.json', 'comparison_log.json', 'main.json', 
            'unique.json', 'posted_properties.json', 'new_amenities.json'
        }
        
        for file_path in all_files:
            is_excluded = file_path.name in excluded
            size_kb = file_path.stat().st_size / 1024
            
            # Try to count properties
            count = "?"
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        count = len(data)
                    elif isinstance(data, dict):
                        for key in ['properties', 'data', 'results']:
                            if key in data:
                                count = len(data[key])
                                break
            except:
                count = "ERROR"
            
            status = "🚫 EXCLUDED" if is_excluded else f"📊 {count} props"
            
            st.write(f"**{file_path.name}**")
            st.write(f"   {status} | {size_kb:.1f} KB")
    
    except Exception as e:
        st.error(f"❌ Error listing files: {e}")

def main():
    """Main application function - REMOVED Load Properties button"""
    apply_custom_css()
    # 1. TITLE
    st.title("🎯 Scraper Dashboard")
    
    # 2. SCRAPER SECTION
    render_enhanced_scrapers_section()
    
    # 3. PROCESS BACKGROUND MESSAGES
    process_messages()

    # 4. AI STATUS
    col_ai1, col_ai2 = st.columns([3, 1])
    with col_ai1:
        if st.session_state.ai_enabled:
            st.success("🤖 AI Mapping: ENABLED")
        else:
            st.error("⚠️ AI Mapping: DISABLED (Check .env file)")
    
    with col_ai2:
        st.metric("AI Cache", len(st.session_state.ai_cache))

    # 5. SIDEBAR - REMOVED Load Properties button
    with st.sidebar:
        st.header("⚙️ Quick Stats")
        json_files = [f for f in OUTPUT_DIR.glob('*.json') 
                      if f.name not in ['info.json', 'comparison_log.json', 'main.json', 
                                       'unique.json', 'posted_properties.json', 'new_amenities.json']]
        st.metric("JSON Files", len(json_files))
        
        lookups = st.session_state.lookups
        st.markdown("---")
        st.subheader("📚 Database Lookups")
        st.metric("Cities", len(lookups.get('cities', {})))
        st.metric("Property Types", len(lookups.get('property_types', {})))
        st.metric("Statuses", len(lookups.get('property_status', {})))
        st.metric("BHK Types", len(lookups.get('bhk', {})))
        st.metric("Amenities", len(lookups.get('amenities', {})))
        st.metric("Localities", len(lookups.get('localities', [])))
        
        st.markdown("---")
        st.subheader("🔧 Diagnostic Tools")
        
        if st.button("🧪 Test Scraper Execution", use_container_width=True):
            test_scraper_execution()

        st.markdown("---")
        st.subheader("📦 Archive Management")
        
        if st.button("📦 Archive Posted (Weekly)", use_container_width=True, type="primary"):
            with st.spinner("Archiving posted properties..."):
                result = archive_posted_properties_weekly()
                st.code(result, language='text')
                st.success("✅ Archive complete!")
        
        st.caption("💡 Archives posted.json files to compressed weekly archives")
        
        if st.button("🔍 Diagnose Python Environment", use_container_width=True):
            diagnose_python_environment()
        
        if st.button("🔍 Diagnose Locality Data", use_container_width=True):
            diagnose_locality_data(st.session_state.lookups)
            st.toast("Check terminal for diagnostics")
        if st.button("📁 List All Files", use_container_width=True):
            list_all_output_files()
        
        if st.button("🔄 Refresh Statistics", use_container_width=True, type="primary"):
            # Clear cached data to force reload from files
            st.session_state.unique_props = []
            st.session_state.by_city = {}
            st.success("✅ Statistics will refresh on next load")
            time.sleep(0.5)
            st.rerun()

    # 6. DATA STATISTICS (City Distribution - Auto-loads from files)
    render_data_statistics_section()

    st.markdown("---")
    
    # 7. GROUND TRUTH COMPARISON
    st.header("🎯 Ground Truth Comparison")
    gt_col1, gt_col2 = st.columns([1, 2])
    
    with gt_col1:
        threshold = st.slider("Match Threshold (%)", 0, 100, 60)
        if st.button("🔍 Run Comparison", use_container_width=True, type="primary"):
            with st.spinner("Analyzing..."):
                result = compare_ground_truth(threshold)
                if result:
                    st.success("✅ Comparison complete!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.warning("⚠️ No ground truth data")
    
    with gt_col2:
        if COMPARISON_LOG.exists():
            try:
                with open(COMPARISON_LOG, 'r') as f:
                    comp = json.load(f)
                
                rate = comp.get('overall_match_rate', 0)
                matched = comp.get('overall_matched', 0)
                total = comp.get('ground_truth_count', 1)
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Ground Truth", total)
                m2.metric("Matched", matched)
                m3.metric("Match Rate", f"{rate}%")
                
                st.progress(min(1.0, matched / total) if total > 0 else 0)
                
                if st.checkbox("Show detailed results"):
                    file_results = comp.get('file_results', {})
                    df_data = []
                    for filename, stats in file_results.items():
                        df_data.append({
                            'File': filename,
                            'Scraped': stats['total_scraped'],
                            'Matches': stats['matches'],
                            'Match Rate': f"{stats['match_rate']}%"
                        })
                    if df_data:
                        st.dataframe(pd.DataFrame(df_data), use_container_width=True)
            except Exception as e:
                st.info(f"Unable to load comparison: {e}")
        else:
            st.info("Run comparison to see results")

    st.markdown("---")
    
    # 9. API POSTING - GLOBAL (ALL SOURCES)
    st.header("📡 Global API Posting (All Sources)")
    
    st.info("💡 **Tip:** You can post individual sources using the 'Post to API' button on each scraper card above, or post all sources together here.")
    
    # Count total pending across all sources
    all_stats = get_scraper_statistics()
    total_pending = sum(
        (stats['extracted'] - stats['posted'] - stats['failed']) 
        for stats in all_stats.values()
    )
    total_posted = sum(stats['posted'] for stats in all_stats.values())
    total_failed = sum(stats['failed'] for stats in all_stats.values())
    
    global_col1, global_col2, global_col3 = st.columns(3)
    
    with global_col1:
        st.metric("⏳ Total Pending", f"{total_pending:,}")
    
    with global_col2:
        st.metric("✅ Total Posted", f"{total_posted:,}")
    
    with global_col3:
        st.metric("❌ Total Failed", f"{total_failed:,}")
    
    if total_pending > 0:
        st.markdown("---")
        
        global_post_col1, global_post_col2, global_post_col3 = st.columns([2, 1, 1])
        
        with global_post_col1:
            global_batch = st.number_input(
                "Properties to post (across all sources)",
                min_value=1,
                max_value=total_pending,
                value=min(50, total_pending),
                key="global_batch_size"
            )
        
        with global_post_col2:
            if st.button("🚀 Post All Sources", use_container_width=True, type="primary"):
                st.session_state.stop_requested = False
                log_placeholder = st.empty()
                post_all_sources_to_api(global_batch, log_placeholder)
                st.success("✅ Global posting complete!")
                time.sleep(1)
                st.rerun()
        
        with global_post_col3:
            if st.button("🛑 Stop Posting", use_container_width=True):
                st.session_state.stop_requested = True
                st.warning("Stopping...")
    else:
        st.success("✅ No pending properties! All sources are up to date.")
    
    # Global Retry Failed
    if total_failed > 0:
        st.markdown("---")
        st.warning(f"⚠️ {total_failed:,} total failed properties across all sources")
        
        if st.button(f"🔄 Retry All Failed ({total_failed})", use_container_width=True, type="secondary"):
            st.session_state.stop_requested = False
            log_placeholder = st.empty()
            retry_all_failed_posts(log_placeholder)
            st.success("✅ Global retry complete!")
            time.sleep(1)
            st.rerun()
    if st.button("Send Images To Image Dashboard"):
        send_to_image_dashboard(st.session_state.all_props)
        st.success("Data sent successfully")


                                                
    # 10. FOOTER
    st.caption("🏢 Real Estate Scraper Dashboard v2.0 | AI-Powered Property Aggregation")
    
    with st.expander("ℹ️ Session Information"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Data Status:**")
            st.write(f"- Unique Props: {len(st.session_state.unique_props)}")
            st.write(f"- Cities: {len(st.session_state.by_city)}")
        with col2:
            st.write("**AI Status:**")
            st.write(f"- Enabled: {st.session_state.ai_enabled}")
            st.write(f"- Cache Size: {len(st.session_state.ai_cache)}")
            st.write(f"- Posted Hashes: {len(st.session_state.posted_property_hashes)}")
        with col3:
            st.write("**System:**")
            st.write(f"- Platform: {platform.system()}")
            st.write(f"- Python: {sys.version.split()[0]}")
            st.write(f"- Output Dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()