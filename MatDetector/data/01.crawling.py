import json
import re
import time
from semanticscholar import SemanticScholar
from tenacity import retry, stop_after_attempt, wait_fixed, RetryError
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEYS_FILE = 'api.txt'
COMBINED_OUTPUT_FILE = '/home/user25/Work_Space/personal_work/01.PubMed/0624/final_entity_list.json'
OUTPUT_FILE = 'one_syn.json'
API_KEY_INDEX = 0
API_KEYS = []

def load_api_keys(filename=API_KEYS_FILE):
    with open(filename, 'r') as f:
        keys = [line.strip() for line in f if line.strip()]
    return keys

def get_next_api_key():
    global API_KEY_INDEX
    key = API_KEYS[API_KEY_INDEX]
    API_KEY_INDEX = (API_KEY_INDEX + 1) % len(API_KEYS)
    return key

@retry(stop=stop_after_attempt(1), wait=wait_fixed(0.2))
def fetch_papers(keyword, api_key):
    sch = SemanticScholar(api_key=api_key)
    results = sch.search_paper(keyword, limit=100)
    time.sleep(0.2)  # Add a delay between each API call
    return results

def AppearYearSementic(kw, kw_type, cmpdname):
    paper_data = []
    keyword = f'{kw}'.replace(' ', '_')  # Replace spaces with underscores
    api_key = get_next_api_key()

    try:
        results = fetch_papers(keyword, api_key)
    except RetryError:
        print(f"API error occurred while fetching papers for keyword: {keyword} with API key: {api_key}")
        return paper_data
    except Exception as e:
        print(f"An unexpected error occurred: {e} with API key: {api_key}")
        return paper_data

    if results:
        count = 0
        for item in results:
            title = item.title
            abstract = item.abstract if item.abstract else ""
            doi = item.externalIds.get('DOI', "None")
            da = item.year

            kw_lower = kw.lower()
            cmpdname_lower = cmpdname.lower()

            if not title and not abstract:
                continue

            if title:
                title_lower = title.lower()
            else:
                title_lower = ""

            if abstract:
                abstract_lower = abstract.lower()

                if (kw_lower in abstract_lower and cmpdname_lower in title_lower) or \
                   (kw_lower in title_lower and cmpdname_lower in abstract_lower) or \
                   (kw_lower in title_lower and cmpdname_lower in title_lower) or \
                   (kw_lower in abstract_lower and cmpdname_lower in abstract_lower):
                    paper_data.append({
                        "title": title,
                        "date": da,
                        "cmpdname": cmpdname,
                        kw_type: kw,
                        "abstract": abstract,
                        "doi": doi
                    })
                    count += 1

            else:
                if kw_lower in title_lower and cmpdname_lower in title_lower:
                    paper_data.append({
                        "title": title,
                        "date": da,
                        "cmpdname": cmpdname,
                        kw_type: kw,
                        "abstract": "",
                        "doi": doi
                    })
                    count += 1

            if count == 200:
                break

    return paper_data

def save_to_json(data, existing_data, filename=OUTPUT_FILE):
    new_data = [item for item in data if item not in existing_data]
    existing_data.extend(new_data)

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)

def load_existing_data(filename=OUTPUT_FILE):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def process_entry(entry, existing_data):
    cmpdname = entry.get("cmpdname", "")
    iupacname = entry.get("iupacname", "")
    mf = entry.get("mf", "")
    cmpdsynonyms = entry.get("cmpdsynonym", [])
    results = []

    if cmpdname:
        results.extend(AppearYearSementic(cmpdname, "cmpdname", cmpdname))

    if iupacname:
        results.extend(AppearYearSementic(iupacname, "iupacname", cmpdname))

    if mf:
        results.extend(AppearYearSementic(mf, "mf", cmpdname))

    if cmpdsynonyms and isinstance(cmpdsynonyms, list):
        for synonym in cmpdsynonyms:
            results.extend(AppearYearSementic(synonym, "cmpdsynonym", cmpdname))

    save_to_json(results, existing_data)
    save_to_json([], existing_data)  # Save progress after each entry is processed

def main():
    global API_KEYS
    API_KEYS = load_api_keys()
    with open(COMBINED_OUTPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    existing_data = load_existing_data()
    total_tasks = sum(len(entry.get("cmpdname", [])) + 1 + (1 if "iupacname" in entry else 0) + (1 if "mf" in entry else 0) + (len(entry.get("cmpdsynonym", [])) if "cmpdsynonym" in entry else 0) for entry in data)
    progress_bar = tqdm(total=total_tasks, desc="Processing")

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(process_entry, entry, existing_data) for entry in data]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred during processing: {e}")
            progress_bar.update(1)

    progress_bar.close()

    print(f"Data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
