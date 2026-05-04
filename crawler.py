import gzip
import hashlib
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait

BASE_URL  = "https://www.bookdelivery.com"
HOME_URL  = f"{BASE_URL}/il-en/"
DELAY     = 1
MAX_PAGES = 5
USD_RATE  = 3.01
CACHE_DIR = Path(".cache")
WAF_TITLES = {"", "Human Verification"}
STAT_COLS  = ["Year", "Price_USD", "StarRating", "NumberOfReviews", "NumberOfAuthors"]


# ── Utilities ─────────────────────────────────────────────────────────────────

def ceil2(x):
    return math.ceil(x * 100) / 100


def extract_left_field(left_text, label):
    m = re.search(
        rf"^{re.escape(label)}\n(.+?)(?=^[A-Za-z]|\Z)",
        left_text,
        re.DOTALL | re.MULTILINE,
    )
    return m.group(1).strip() if m else ""


# ── Driver / fetch ────────────────────────────────────────────────────────────

def init_driver():
    options = Options()
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    )
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    return webdriver.Chrome(service=Service(), options=options)


def fetch(url, driver):
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / hashlib.md5(url.encode()).hexdigest()
    if cache_file.is_file():
        with gzip.open(cache_file, "rt", encoding="utf-8") as f:
            return BeautifulSoup(f, "lxml")
    time.sleep(DELAY)
    driver.get(url)
    WebDriverWait(driver, 15).until(lambda d: d.title not in WAF_TITLES)
    soup = BeautifulSoup(driver.page_source, "lxml")
    with gzip.open(cache_file, "wt", encoding="utf-8") as f:
        f.write(str(soup))
    return soup


# ── Crawl helpers ─────────────────────────────────────────────────────────────

def get_categories(soup):
    cats = []
    for a in soup.select(".categories a"):
        href = a.get("href", "")
        name = a.get_text(strip=True)
        if href and name:
            cats.append((name, href))
    return cats


def get_book_links(soup):
    links = set()
    for a in soup.select('a[href*="/il-en/book-"]'):
        href = a["href"]
        if not href.startswith("http"):
            href = BASE_URL + href
        links.add(href)
    return links


def get_next_page_url(soup):
    next_a = soup.find("a", string=re.compile(r"Next\s*Page", re.I))
    if next_a and next_a.get("href"):
        href = next_a["href"]
        if not href.startswith("http"):
            href = BASE_URL + href
        return href
    return None


def parse_book(soup, category_name):
    result = {}

    title = ""
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            d = json.loads(script.string)
            if d.get("@type") == "Product":
                title = d.get("name", "").strip()
                break
        except Exception:
            pass
    if not title:
        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else ""
    result["Title"] = title
    result["Category"] = category_name

    info = soup.find("div", class_="product-info")
    left_col = info.find("div", class_="col-md-3") if info else None
    left_text = left_col.get_text(separator="\n", strip=True) if left_col else ""

    skip_re = re.compile(r"view author|ver autor", re.I)
    authors = []
    if info:
        for a in info.find_all("a", href=re.compile(r"/author/")):
            name = a.get_text(strip=True)
            if name and not skip_re.search(name):
                authors.append(name)
    result["Authors"] = ", ".join(dict.fromkeys(authors))

    cats_start = left_text.find("Categories\n")
    if cats_start >= 0:
        cats_text = left_text[cats_start + len("Categories\n"):]
        cat_lines = [
            l.strip()
            for l in cats_text.split("\n")
            if l.strip() and not l.strip().startswith("(")
        ]
        result["Categories"] = ", ".join(cat_lines)
    else:
        result["Categories"] = ""

    result["Year"]     = extract_left_field(left_text, "Year")
    result["Language"] = extract_left_field(left_text, "Language")
    result["Format"]   = extract_left_field(left_text, "Format")

    dim_raw = extract_left_field(left_text, "Dimensions")
    dim_m = re.match(r"([\d.,\s x]+)\s*(\w+)", dim_raw)
    if dim_m:
        dims = re.findall(r"[\d.]+", dim_m.group(1).replace(",", "."))
        result["Dimensions"]      = ", ".join(dims)
        result["Dimensions_unit"] = dim_m.group(2)
    else:
        result["Dimensions"]      = ""
        result["Dimensions_unit"] = ""

    weight_raw = extract_left_field(left_text, "Weight")
    w_m = re.match(r"([\d.,]+)\s*([a-zA-Z]+)", weight_raw)
    if w_m:
        result["Weight"]      = w_m.group(1).replace(",", ".")
        result["Weight_unit"] = w_m.group(2)
    else:
        result["Weight"]      = ""
        result["Weight_unit"] = ""

    result["ISBN"] = extract_left_field(left_text, "ISBN13") or extract_left_field(left_text, "ISBN")

    precio = soup.find("strong", class_="precio")
    if precio:
        price_str = re.sub(r"[^\d.,]", "", precio.get_text(strip=True)).replace(",", ".")
        try:
            nis = float(price_str)
            result["Price_NIS"] = f"{ceil2(nis):.2f}"
            result["Price_USD"] = f"{ceil2(nis / USD_RATE):.2f}"
        except ValueError:
            result["Price_NIS"] = ""
            result["Price_USD"] = ""
    else:
        result["Price_NIS"] = ""
        result["Price_USD"] = ""

    syn_h = soup.find("h2", string=re.compile(r"Synopsis", re.I))
    synopsis = ""
    if syn_h:
        nxt = syn_h.find_next_sibling()
        if nxt:
            synopsis = nxt.get_text(strip=True)
    result["Synopsis"]        = synopsis
    result["Synopsis_length"] = len(synopsis)

    star_rating = "None"
    n_reviews   = 0
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            d = json.loads(script.string)
            if d.get("@type") == "Product" and "review" in d:
                reviews = d["review"]
                n_reviews = len(reviews)
                if n_reviews > 0:
                    dist = Counter(
                        int(r["reviewRating"]["ratingValue"])
                        for r in reviews
                        if "reviewRating" in r
                    )
                    total    = sum(dist.values())
                    weighted = sum(stars * count for stars, count in dist.items())
                    star_rating = f"{ceil2(weighted / total):.2f}"
        except Exception:
            pass
    result["StarRating"]      = star_rating
    result["NumberOfReviews"] = n_reviews

    return result


# ── Pipeline stages ───────────────────────────────────────────────────────────

def crawl_all(driver, categories):
    all_books    = []
    visited_books = set()

    for cat_name, cat_url in categories:
        print(f"\n=== Category: {cat_name} ===")
        page_url = cat_url
        page_num = 1

        while page_url and (MAX_PAGES is None or page_num <= MAX_PAGES):
            print(f"  Page {page_num}: {page_url}")
            cat_soup   = fetch(page_url, driver)
            book_links = get_book_links(cat_soup)
            print(f"    {len(book_links)} book links found")

            for book_url in book_links:
                if book_url in visited_books:
                    continue
                visited_books.add(book_url)
                try:
                    bsoup = fetch(book_url, driver)
                    book  = parse_book(bsoup, cat_name)
                    book["url"] = book_url
                    all_books.append(book)
                    print(f"      OK: {book['Title'][:60]}")
                except Exception as e:
                    print(f"      ERR {book_url}: {e}")

            page_url = get_next_page_url(cat_soup)
            page_num += 1

    return all_books


def build_dataframe(records):
    df = pd.DataFrame(records)
    df = df[df["Title"] != ""]
    df["Year"]       = df["Year"].replace("", pd.NA).astype(float)
    df["Weight"]     = df["Weight"].replace("", pd.NA).astype(float)
    df["Price_NIS"]  = df["Price_NIS"].replace("", pd.NA).astype(float)
    df["Price_USD"]  = df["Price_USD"].replace("", pd.NA).astype(float)
    df["StarRating"] = df["StarRating"].replace("None", pd.NA).astype(float)
    return df


def save_outputs(df):
    Path("output").mkdir(exist_ok=True)
    df.to_csv("output/books_raw.csv", index=False)
    records = df.to_dict(orient="records")
    clean = [
        {k: v for k, v in row.items()
         if v is not None and v != ""
         and not (isinstance(v, float) and math.isnan(v))}
        for row in records
    ]
    with open("output/books_raw.json", "w", encoding="utf-8") as f:
        json.dump({"records": {"record": clean}}, f, indent=2, ensure_ascii=False)


def sort_dataframe(df):
    return df.sort_values("Title")


def enrich_dataframe(df):
    df = df.copy()
    df["isExpensive"]    = df["Price_NIS"] > df["Price_NIS"].median()
    df["NumberOfAuthors"] = np.where(
        df["Authors"].str.contains(";"),
        df["Authors"].str.count(";") + 1,
        df["Authors"].str.count(",") + 1,
    )
    return df


def calculate_stats(df):
    for col in STAT_COLS:
        print(f"\n── {col} ──")
        print(df[col].dropna().describe().to_string())


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    driver     = init_driver()
    home_soup  = fetch(HOME_URL, driver)
    categories = get_categories(home_soup)
    print(f"Found {len(categories)} categories")
    records    = crawl_all(driver, categories)
    print(f"\nDone. {len(records)} books crawled.")
    df         = build_dataframe(records)
    save_outputs(df)
    df         = sort_dataframe(df)
    df         = enrich_dataframe(df)
    calculate_stats(df)


if __name__ == "__main__":
    main()
