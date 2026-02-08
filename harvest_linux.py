import requests
from bs4 import BeautifulSoup
import time
import os

# The targets: We focus on core documentation entry points
TARGETS = {
    "arch": "https://wiki.archlinux.org/title/Installation_guide",
    "alpine": "https://wiki.alpinelinux.org/wiki/Installation",
    "lfs": "https://www.linuxfromscratch.org/lfs/view/stable/chapter01/introduction.html",
    "crux": "https://crux.nu/Main/Handbook3-7",
    "slackware": "https://docs.slackware.com/slackware:install"
}

def clean_text(soup):
    # Remove scripts, styles, and nav bars
    for element in soup(["script", "style", "nav", "footer", "header"]):
        element.extract()
    return soup.get_text(separator=' ', strip=True)

def harvest():
    os.makedirs("data/linux_docs", exist_ok=True)
    
    for distro, url in TARGETS.items():
        print(f"--- Harvesting {distro.upper()} ---")
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            text_data = clean_text(soup)
            
            with open(f"data/linux_docs/{distro}.txt", "w", encoding="utf-8") as f:
                f.write(text_data)
            print(f"Success! Saved {distro}.txt")
            
            # Be nice to their servers (especially LFS/Crux)
            time.sleep(2) 
        except Exception as e:
            print(f"Failed to harvest {distro}: {e}")

if __name__ == "__main__":
    harvest()
