"""
Searcher module — fetches job listings from multiple platforms.
Each platform has its own scraper method registered in a dispatcher.
"""

import hashlib
import logging
import random
import time
from typing import Callable, Optional
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup

from .config import MAX_JOBS_PER_QUERY, REQUEST_DELAY_RANGE
from .models import Job

log = logging.getLogger(__name__)

# Common headers to avoid instant blocking
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


class Searcher:
    """
    Multi-platform job searcher.
    Each platform is registered as a scraper function.
    Gracefully skips platforms that fail.
    """

    def __init__(self, platforms: Optional[list[str]] = None):
        from .config import PLATFORMS

        self.platforms = platforms or PLATFORMS

        # Registry of platform → scraper function
        self._scrapers: dict[str, Callable[[str], list[Job]]] = {
            "indeed": self._search_indeed,
            "linkedin": self._search_linkedin,
            "remoteok": self._search_remoteok,
            "appen": self._search_appen,
        }

    # ── Public API ──────────────────────────────────────

    def search_all(self, query: str) -> list[Job]:
        """Run a query across all configured platforms. Returns combined Job list."""
        all_jobs: list[Job] = []

        for platform in self.platforms:
            scraper = self._scrapers.get(platform)
            if not scraper:
                log.warning(f"No scraper registered for platform: {platform}")
                continue

            try:
                log.info(f"Searching {platform} for: '{query}'")
                jobs = scraper(query)
                log.info(f"  → Found {len(jobs)} jobs on {platform}")
                all_jobs.extend(jobs)
            except Exception as e:
                log.warning(f"Platform {platform} failed for query '{query}': {e}")

            # Rate-limit between platforms
            self._delay()

        return all_jobs

    def search_platform(self, query: str, platform: str) -> list[Job]:
        """Search a single platform."""
        scraper = self._scrapers.get(platform)
        if not scraper:
            raise ValueError(f"Unknown platform: {platform}")
        return scraper(query)

    # ── Indeed Scraper ──────────────────────────────────

    def _search_indeed(self, query: str) -> list[Job]:
        """
        Scrape Indeed's search results page.
        Extracts job cards from the HTML.
        """
        url = f"https://www.indeed.com/jobs?q={quote_plus(query)}&l=Remote&fromage=14"
        jobs: list[Job] = []

        try:
            resp = requests.get(url, headers=_HEADERS, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # Indeed uses various class names — try common selectors
            cards = soup.select("div.job_seen_beacon, div.jobsearch-ResultsList > div")

            for card in cards[:MAX_JOBS_PER_QUERY]:
                job = self._parse_indeed_card(card, query)
                if job:
                    jobs.append(job)

        except requests.RequestException as e:
            log.warning(f"Indeed request failed: {e}")

        return jobs

    def _parse_indeed_card(self, card, query: str) -> Optional[Job]:
        """Extract job fields from an Indeed job card BeautifulSoup tag."""
        try:
            title_el = card.select_one("h2.jobTitle a, h2.jobTitle span")
            company_el = card.select_one("span[data-testid='company-name'], span.companyName")
            location_el = card.select_one("div[data-testid='text-location'], div.companyLocation")
            snippet_el = card.select_one("div.job-snippet, td.resultContent div.heading6")
            link_el = card.select_one("a[data-jk], h2.jobTitle a")

            title = title_el.get_text(strip=True) if title_el else None
            if not title:
                return None

            company = company_el.get_text(strip=True) if company_el else "Unknown"
            location = location_el.get_text(strip=True) if location_el else ""
            description = snippet_el.get_text(strip=True) if snippet_el else ""

            job_url = ""
            if link_el and link_el.get("href"):
                href = link_el["href"]
                job_url = href if href.startswith("http") else f"https://www.indeed.com{href}"

            is_remote = "remote" in location.lower() or "remote" in title.lower()

            return Job(
                title=title,
                company=company,
                platform="indeed",
                url=job_url,
                description=description,
                location=location,
                is_remote=is_remote,
                raw_query=query,
            )
        except Exception as e:
            log.debug(f"Failed to parse Indeed card: {e}")
            return None

    # ── LinkedIn Scraper ────────────────────────────────

    def _search_linkedin(self, query: str) -> list[Job]:
        """
        Scrape LinkedIn's public job search page (no login required).
        Uses the guest-accessible search endpoint.
        """
        url = (
            f"https://www.linkedin.com/jobs/search/"
            f"?keywords={quote_plus(query)}&f_WT=2&sortBy=DD"  # f_WT=2 = remote
        )
        jobs: list[Job] = []

        try:
            resp = requests.get(url, headers=_HEADERS, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # LinkedIn guest page uses ul.jobs-search__results-list
            cards = soup.select(
                "div.base-card, li.result-card, div.base-search-card"
            )

            for card in cards[:MAX_JOBS_PER_QUERY]:
                job = self._parse_linkedin_card(card, query)
                if job:
                    jobs.append(job)

        except requests.RequestException as e:
            log.warning(f"LinkedIn request failed: {e}")

        return jobs

    def _parse_linkedin_card(self, card, query: str) -> Optional[Job]:
        """Extract job fields from a LinkedIn guest-page job card."""
        try:
            title_el = card.select_one("h3.base-search-card__title, h3.result-card__title")
            company_el = card.select_one("h4.base-search-card__subtitle a, a.result-card__subtitle-link")
            location_el = card.select_one("span.job-search-card__location")
            link_el = card.select_one("a.base-card__full-link, a.result-card__full-card-link")
            date_el = card.select_one("time")

            title = title_el.get_text(strip=True) if title_el else None
            if not title:
                return None

            company = company_el.get_text(strip=True) if company_el else "Unknown"
            location = location_el.get_text(strip=True) if location_el else ""
            job_url = link_el["href"].split("?")[0] if link_el and link_el.get("href") else ""
            posted = date_el.get("datetime", "") if date_el else ""

            is_remote = "remote" in location.lower() or "remote" in title.lower()

            return Job(
                title=title,
                company=company,
                platform="linkedin",
                url=job_url,
                description="",  # LinkedIn guest page doesn't show full desc
                location=location,
                is_remote=is_remote,
                posted_date=posted,
                raw_query=query,
            )
        except Exception as e:
            log.debug(f"Failed to parse LinkedIn card: {e}")
            return None

    # ── RemoteOK Scraper ────────────────────────────────

    def _search_remoteok(self, query: str) -> list[Job]:
        """
        RemoteOK provides a JSON API — very clean to parse.
        """
        url = f"https://remoteok.com/api?tag={quote_plus(query)}"
        jobs: list[Job] = []

        try:
            resp = requests.get(url, headers={**_HEADERS, "Accept": "application/json"}, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            # First element is metadata, skip it
            for item in data[1: MAX_JOBS_PER_QUERY + 1]:
                title = item.get("position", "")
                company = item.get("company", "Unknown")
                job_url = item.get("url", "")

                if not title:
                    continue

                jobs.append(
                    Job(
                        title=title,
                        company=company,
                        platform="remoteok",
                        url=job_url if job_url.startswith("http") else f"https://remoteok.com{job_url}",
                        description=item.get("description", ""),
                        location=item.get("location", "Remote"),
                        is_remote=True,
                        posted_date=item.get("date", ""),
                        salary_range=self._extract_salary(item),
                        raw_query=query,
                    )
                )

        except (requests.RequestException, ValueError) as e:
            log.warning(f"RemoteOK request failed: {e}")

        return jobs

    def _extract_salary(self, item: dict) -> Optional[str]:
        """Extract salary range from a RemoteOK API item."""
        sal_min = item.get("salary_min")
        sal_max = item.get("salary_max")
        if sal_min and sal_max:
            return f"${sal_min:,}–${sal_max:,}"
        elif sal_min:
            return f"${sal_min:,}+"
        return None

    # ── Appen Scraper ───────────────────────────────────

    def _search_appen(self, query: str) -> list[Job]:
        """
        Scrape Appen's public jobs/projects page.
        Appen is specifically known for data annotation tasks.
        """
        url = "https://appen.com/jobs/"
        jobs: list[Job] = []

        try:
            resp = requests.get(url, headers=_HEADERS, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # Look for job listing containers
            cards = soup.select("div.job-listing, article.post, div.career-listing, li.job-item")

            for card in cards[:MAX_JOBS_PER_QUERY]:
                title_el = card.select_one("h2, h3, a.job-title, span.title")
                link_el = card.select_one("a[href]")

                title = title_el.get_text(strip=True) if title_el else None
                if not title:
                    continue

                # Filter: only keep if query terms overlap
                if not self._query_overlaps(query, title):
                    continue

                job_url = ""
                if link_el and link_el.get("href"):
                    href = link_el["href"]
                    job_url = href if href.startswith("http") else f"https://appen.com{href}"

                jobs.append(
                    Job(
                        title=title,
                        company="Appen",
                        platform="appen",
                        url=job_url,
                        description="Appen crowdsourced AI data task",
                        is_remote=True,  # Appen tasks are always remote
                        raw_query=query,
                    )
                )

        except requests.RequestException as e:
            log.warning(f"Appen request failed: {e}")

        return jobs

    # ── Helpers ─────────────────────────────────────────

    def _query_overlaps(self, query: str, text: str) -> bool:
        """Check if any word from the query appears in the text."""
        query_words = set(query.lower().split())
        text_lower = text.lower()
        return any(w in text_lower for w in query_words if len(w) > 2)

    def _delay(self):
        """Sleep for a random duration to respect rate limits."""
        lo, hi = REQUEST_DELAY_RANGE
        time.sleep(random.uniform(lo, hi))
