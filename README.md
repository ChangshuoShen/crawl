# ICML 2025 Paper Crawler

This project crawls papers from ICML 2025, filters papers of interest, and searches for them on arXiv.

## Features

- **Paper Crawling**: Automatically crawls all accepted papers from ICML 2025
- **Abstract Extraction**: Extracts abstracts from paper detail pages
- **AI-Powered Filtering**: Uses OpenAI GPT models to classify papers by research direction (e.g., LLM Safety)
- **arXiv Integration**: Searches for papers on arXiv for additional information

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install requests beautifulsoup4 openai python-dotenv
   ```
3. Create a `.env` file and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

1. **Crawl ICML 2025 papers**:
   ```python
   python crawl_icml2025.py
   ```
   This will:
   - Crawl all papers from ICML 2025
   - Extract abstracts for each paper
   - Save results to `./results/icml2025.json`

2. **Filter papers by research direction**:
   The script uses OpenAI's GPT model to classify papers based on specified research directions (default: "LLM Safety").

## Output

The crawler generates a JSON file containing:
- Paper titles
- Paper URLs
- Abstracts
- Classification results

## File Structure

- `crawl_icml2025.py`: Main crawler script
- `results/icml2025.json`: Output file with crawled papers
- `.env`: Environment variables (API keys)
