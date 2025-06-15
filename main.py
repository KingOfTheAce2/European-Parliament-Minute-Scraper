from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from lxml import etree
from tqdm import tqdm
import time
import re
import json
import os
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import login, create_repo, HfApi # Import login and HfApi

# --- Configuration ---
# Base URL for the Dutch minutes archive
BASE_MINUTES_URL = "https://www.europarl.europa.eu/plenary/nl/minutes.html"
# Base URL for the documents themselves
BASE_DOC_URL = "https://www.europarl.europa.eu/doceo/document/"

# Hugging Face Dataset configuration
# HF_USERNAME will be read from GitHub Secrets in Actions, or defaults locally
HF_USERNAME = os.environ.get("HF_USERNAME", "YOUR_HUGGINGFACE_USERNAME") 
HF_DATASET_NAME = "europarl-dutch-minutes" # You can change this name if you like
HF_REPO_ID = f"{HF_USERNAME}/{HF_DATASET_NAME}"

# Define the XML namespaces for lxml parsing
NAMESPACES = {
    'text': "http://openoffice.org/2000/text",
    'table': "http://openoffice.org/2000/table",
}

# --- Utility Functions ---

def download_content(url, session):
    """Downloads content from the given URL using a requests session."""
    try:
        response = session.get(url, timeout=20)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return None

def extract_xml_links_from_html(html_content):
    """
    Parses HTML content to extract all unique XML document links.
    """
    soup = BeautifulSoup(html_content, 'lxml')
    xml_links = set()

    for a_tag in soup.find_all('a', href=re.compile(r'\.xml$')):
        # Filter for specific 'XML' links within the structure observed
        if 'nopadding' in a_tag.get('class', []) or 'link_simple_iconsmall' in a_tag.get('class', []):
            href = a_tag['href']
            full_url = urljoin(BASE_DOC_URL, href)
            xml_links.add(full_url)
    return list(xml_links)

def get_all_archive_xml_links():
    """
    Navigates the Europarl archive by parliamentary term and collects all XML links.
    """
    all_xml_urls = set()
    
    with requests.Session() as session:
        # Get the main page to find the parliamentary terms dropdown
        print("Fetching main minutes page to identify parliamentary terms...")
        html_content_main = download_content(BASE_MINUTES_URL, session)
        if not html_content_main:
            print("Failed to fetch main minutes page.")
            return []

        soup_main = BeautifulSoup(html_content_main, 'lxml')

        term_select = soup_main.find('select', id='criteriaSidesLeg')
        if not term_select:
            print("Could not find the parliamentary term dropdown. Check HTML structure.")
            return []

        parliamentary_terms = []
        for option in term_select.find_all('option'):
            term_value = option.get('value')
            term_title = option.get('title')
            if term_value and term_title:
                parliamentary_terms.append((term_value, term_title))
        
        print(f"Found {len(parliamentary_terms)} parliamentary terms: {parliamentary_terms}")

        for term_value, term_title in tqdm(parliamentary_terms, desc="Collecting XML Links by Term"):
            # print(f"\nProcessing term: {term_title} (Value: {term_value})") # Commented for cleaner tqdm output
            
            # Simulate form submission to get minutes for this term
            form_data = {
                'clean': 'false', 'legChange': 'false', 'source': '', 'dateSys': '',
                'tabActif': 'tabResult', 'leg': term_value,
                'refSittingDateStart': '', 'refSittingDateEnd': '',
                'miType': 'text', 'miText': '', 'sortResults': ''
            }

            try:
                term_response = session.post(BASE_MINUTES_URL, data=form_data, timeout=30)
                term_response.raise_for_status()
            except requests.exceptions.RequestException as e:
                print(f"Error submitting form for term {term_title}: {e}")
                time.sleep(5)
                continue

            links_on_term_page = extract_xml_links_from_html(term_response.text)
            initial_count = len(all_xml_urls)
            all_xml_urls.update(links_on_term_page)
            # print(f"  Found {len(all_xml_urls) - initial_count} new XML links for {term_title}. Total: {len(all_xml_urls)}") # Commented for cleaner tqdm output
            time.sleep(1) # Be polite, add a small delay between term requests

    return list(all_xml_urls)

def clean_text(text):
    """Applies common text cleaning rules."""
    # Remove XML-like tags (e.g., <text:line-break /> that might survive string() if not handled by lxml)
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove excessive whitespace, including newlines, tabs, and multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove procedural notes (common patterns in English/Dutch minutes)
    text = re.sub(r'\(The sitting (?:was suspended|opened|closed|ended) at.*?\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(Voting time ended at.*?\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\((?:debat|stemming|vraag|interventie)\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(Het woord wordt gevoerd door:.*?\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(\(|\[)\s*(?:(?:[a-zA-Z]{2,3})\s*(?:|\s|))?\s*(?:artikel|rule|punt|item)\s*\d+(?:,\s*lid\s*\d+)?\s*(?:\s+\w+)?\s*(\)|\])', '', text, flags=re.IGNORECASE)
    
    # Remove reference texts (like links, document references)
    text = re.sub(r'\[(COM|A)\d+-\d+(/\d+)?\]', '', text)
    text = re.sub(r'\(?(?:http|https):\/\/[^\s]+?\)', '', text)
    text = re.sub(r'\[\s*\d{4}/\d{4}\(COD\)\]', '', text)
    text = re.sub(r'\[\s*\d{4}/\d{4}\(INI\)\]', '', text)
    text = re.sub(r'\[\s*\d{4}/\d{4}\(RSP\)\]', '', text)
    text = re.sub(r'\[\s*\d{4}/\d{4}\(IMM\)\]', '', text)
    text = re.sub(r'\[\s*\d{4}/\d{4}\(NLE\)\]', '', text)
    text = re.sub(r'\[\s*\d{5}/\d{4}\s*-\s*C\d+-\d+/\d+\s*-\s*\d{4}/\d{4}\(NLE\)\]', '', text)
    
    text = re.sub(r'\(“Stemmingsuitslagen”, punt \d+\)', '', text)
    text = re.sub(r'\(de Voorzitter(?: maakt na de toespraak van.*?| weigert in te gaan op.*?| stemt toe| herinnert eraan dat de gedragsregels moeten worden nageleefd| neemt er akte van|)\)', '', text)
    text = re.sub(r'\(zie bijlage.*?\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(\s*De vergadering wordt om.*?geschorst\.\)', '', text)
    text = re.sub(r'\(\s*De vergadering wordt om.*?hervat\.\)', '', text)
    text = re.sub(r'Volgens de “catch the eye”-procedure wordt het woord gevoerd door.*?\.', '', text)
    text = re.sub(r'Het woord wordt gevoerd door .*?\.', '', text)
    text = re.sub(r'De vergadering wordt om \d{1,2}\.\d{2} uur gesloten.', '', text)
    text = re.sub(r'De vergadering wordt om \d{1,2}\.\d{2} uur geopend.', '', text)
    text = re.sub(r'Het debat wordt gesloten.', '', text)
    text = re.sub(r'Stemming:.*?\.', '', text)

    # Remove remaining multiple spaces and strip leading/trailing whitespace
    text = re.sub(r'\s{2,}', ' ', text).strip()
    
    return text

def extract_dutch_text_from_xml(xml_content):
    """
    Parses XML content using lxml and extracts relevant Dutch text.
    Assumes the XML file is already identified as Dutch via its URL (_NL.xml).
    """
    try:
        parser = etree.XMLParser(recover=True, ns_clean=True)
        root = etree.fromstring(xml_content, parser=parser)
    except etree.XMLSyntaxError as e:
        # print(f"LXML XMLSyntaxError: {e}")
        return None
    except Exception as e:
        # print(f"Unexpected LXML parsing error: {e}")
        return None

    dutch_texts = []

    # Iterate over specific top-level sections that contain narrative content
    relevant_sections = [
        "PV.Other.Text",
        "PV.Debate.Text",
        "PV.Vote.Text",
        "PV.Sitting.Resumption.Text",
        "PV.Approval.Text",
        "PV.Agenda.Text",
        "PV.Sitting.Closure.Text",
    ]

    for section_name in relevant_sections:
        xpath_query = f"//{section_name}//text:p"
        
        for p_tag in root.xpath(xpath_query, namespaces=NAMESPACES):
            text_content = p_tag.xpath("string()").strip()
            
            if text_content:
                # Exclude content from within <table:table> which often contains lists/legends
                if p_tag.xpath("ancestor::table:table", namespaces=NAMESPACES):
                    continue
                
                # Check if it's primarily a list of names (common in <Orator.List.Text>)
                if p_tag.xpath("./Orator.List.Text", namespaces=NAMESPACES) or \
                   p_tag.xpath("./Attendance.Participant.Name", namespaces=NAMESPACES):
                    name_list_text = p_tag.xpath("string(./Orator.List.Text)", namespaces=NAMESPACES).strip()
                    if len(text_content) < 100 and name_list_text and name_list_text == text_content:
                        continue

                # Additional filtering for very short or non-substantive text
                if len(text_content) < 20 and not re.search(r'[a-zA-Z]{5,}', text_content):
                    continue

                dutch_texts.append(text_content)

    final_text = clean_text("\n".join(dutch_texts))
    
    return final_text if final_text and len(final_text) > 50 else None # Minimum length for final text

def process_and_extract_data(xml_urls):
    """
    Processes a list of XML URLs, extracts Dutch text, and returns structured data.
    """
    processed_data = []
    
    with requests.Session() as session:
        for url in tqdm(xml_urls, desc="Extracting Dutch Text from URLs"):
            # Skip non-Dutch XMLs if they somehow got into the list (shouldn't happen with updated scraper)
            if "_NL.xml" not in url:
                continue

            xml_content = download_content(url, session)
            if xml_content:
                text = extract_dutch_text_from_xml(xml_content)
                if text:
                    processed_data.append({
                        "URL": url,
                        "text": text,
                        "source": "European Parliament Minutes"
                    })
            time.sleep(0.1) # Small delay to be polite to the server
    return processed_data

def load_existing_dataset(repo_id):
    """Loads an existing dataset from Hugging Face Hub, handling potential errors."""
    try:
        # Load only the 'train' split if it exists
        dataset = load_dataset(repo_id, split='train')
        print(f"Loaded existing dataset from {repo_id} with {len(dataset)} records.")
        return dataset
    except Exception as e:
        print(f"Could not load existing dataset from {repo_id}: {e}")
        print("Assuming this is the first run or dataset is private/non-existent.")
        return None

def main():
    print("--- Starting Europarl Dutch Minutes Pipeline ---")

    # --- Authenticate with Hugging Face ---
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token) # Use huggingface_hub.login()
        print("Successfully logged into Hugging Face Hub.")
    else:
        print("Hugging Face token not found in environment. Please set HF_TOKEN.")
        print("Set it as an environment variable (e.g., in GitHub Secrets) or run 'huggingface-cli login' locally.")
        return # Exit if no token is available

    # Step 1: Collect all XML URLs
    print("\n--- Phase 1: Collecting XML URLs ---")
    all_xml_urls = get_all_archive_xml_links()
    if not all_xml_urls:
        print("No XML URLs collected. Exiting.")
        return
    print(f"Collected {len(all_xml_urls)} potential XML URLs.")

    # Save scraped URLs to a file for artifact upload (for debugging/audit)
    with open("europarl_xml_urls.txt", "w", encoding="utf-8") as f:
        for url in all_xml_urls:
            f.write(url + "\n")
    print("XML URLs saved to 'europarl_xml_urls.txt' artifact.")

    # Step 2: Process and Extract Dutch Text
    print("\n--- Phase 2: Extracting Dutch Content ---")
    newly_scraped_data = process_and_extract_data(all_xml_urls)
    if not newly_scraped_data:
        print("No Dutch text extracted from collected URLs. Exiting.")
        return
    print(f"Successfully extracted {len(newly_scraped_data)} new records with Dutch text.")

    # Save processed data to a file for artifact upload (for debugging/audit)
    with open("europarl_dutch_data_sample.json", "w", encoding="utf-8") as f:
        json.dump(newly_scraped_data, f, ensure_ascii=False, indent=2)
    print("Processed Dutch data saved to 'europarl_dutch_data_sample.json' artifact.")

    # Step 3: Prepare for Hugging Face Upload (Incremental Update)
    print("\n--- Phase 3: Preparing for Hugging Face Upload (Incremental Update) ---")
    
    existing_dataset = load_existing_dataset(HF_REPO_ID)
    
    # Convert new data to a Dataset object
    new_data_dataset = Dataset.from_list(newly_scraped_data)

    combined_dataset_to_push = None

    if existing_dataset is not None:
        # Create a set of existing URLs for efficient lookup
        existing_urls = set(existing_dataset["URL"])
        
        # Filter out records that are already in the existing dataset
        def is_new_record(record):
            return record["URL"] not in existing_urls

        filtered_new_data = new_data_dataset.filter(is_new_record)

        if len(filtered_new_data) > 0:
            print(f"Found {len(filtered_new_data)} truly new records to add.")
            # Concatenate existing data with new data
            combined_dataset_to_push = DatasetDict({
                'train': existing_dataset.concatenate(filtered_new_data)
            })
        else:
            print("No new records found since the last run. Dataset is up to date.")
            return # Exit if nothing new to push
    else:
        print(f"No existing dataset '{HF_REPO_ID}' found. Creating a new one.")
        combined_dataset_to_push = DatasetDict({'train': new_data_dataset})
    
    # Step 4: Push to Hugging Face Hub
    print(f"\n--- Phase 4: Pushing to Hugging Face Hub: {HF_REPO_ID} ---")
    try:
        # Ensure the repo exists before pushing the dataset for the first time
        # This is important especially for private repos or if there's no pre-existing dataset.
        api = HfApi()
        try:
            api.create_repo(repo_id=HF_REPO_ID, repo_type="dataset", exist_ok=True)
            print(f"Ensured Hugging Face dataset repository '{HF_REPO_ID}' exists.")
        except Exception as e:
            print(f"Could not ensure Hugging Face dataset repository exists: {e}")
            print("Please check your Hugging Face token and permissions.")
            return

        # Push the dataset. The library handles updating if it exists.
        combined_dataset_to_push.push_to_hub(HF_REPO_ID, private=False) # Set private=True if desired
        print(f"Dataset successfully pushed/updated to https://huggingface.co/datasets/{HF_REPO_ID}")
    except Exception as e:
        print(f"Error pushing dataset to Hugging Face Hub: {e}")
        print("Please ensure your HF_TOKEN has 'write' access and HF_USERNAME is correct.")

    print("\n--- Pipeline Finished ---")

if __name__ == "__main__":
    main()
