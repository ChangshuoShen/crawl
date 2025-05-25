#%%

import requests
from bs4 import BeautifulSoup
import json
import os
import re

# URL to crawl all accepted papers
url = "https://icml.cc/virtual/2025/papers.html?layout=mini"

def crawl_papers(url):
    """Crawl all accepted papers from ICML 2025 and extract title and URL"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all paper links
        papers = []
        
        # Look for ul elements that contain paper links
        ul_elements = soup.find_all('ul')
        
        for ul in ul_elements:
            # Find all li elements with a tags that have href starting with /virtual/2025/poster/
            li_elements = ul.find_all('li')
            for li in li_elements:
                link = li.find('a')
                if link and link.get('href'):
                    href = link.get('href')
                    # Check if this is a poster link
                    if href.startswith('/virtual/2025/poster/'):
                        title = link.get_text().strip()
                        # Convert relative URL to absolute URL
                        full_url = f"https://icml.cc{href}"
                        
                        papers.append({
                            'title': title,
                            'url': full_url
                        })
        
        return papers
    
    except Exception as e:
        print(f"Error crawling {url}: {e}")
        return []

def save_results():
    """Crawl all accepted papers and save results to JSON file"""
    # Create results directory if it doesn't exist
    os.makedirs('./results', exist_ok=True)
    
    print("Crawling all accepted papers from ICML 2025")
    papers = crawl_papers(url)
    
    # Save to JSON file
    output_file = "./results/icml2025.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(papers)} papers to {output_file}")

if __name__ == "__main__":
    save_results()
#%%
def extract_abstract_from_url(url):
    """Extract abstract from a paper URL"""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the abstract div with id="abstractExample"
        abstract_div = soup.find('div', id='abstractExample')
        if abstract_div:
            # Extract all text content from the div, excluding script tags
            # Remove script tags first
            for script in abstract_div.find_all('script'):
                script.decompose()
            
            # Get all text content and clean it up
            abstract_text = abstract_div.get_text(separator=' ', strip=True)
            
            # Remove the "Abstract:" prefix if it exists
            if abstract_text.startswith('Abstract:'):
                abstract_text = abstract_text[9:].strip()
            
            return abstract_text
        
        return None
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def add_abstracts_to_papers():
    """Add abstracts to all papers in the JSON file"""
    # Read the JSON file
    with open('./results/icml2025.json', 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    # Extract abstracts for each paper
    for i, paper in enumerate(papers):
        if 'abstract' in paper:
            # print(f"Skipping paper {i+1}/{len(papers)}: {paper['title']} because it already has an abstract")
            continue
        print(f"Processing paper {i+1}/{len(papers)}: {paper['title']}")
        abstract = extract_abstract_from_url(paper['url'])
        if abstract:
            paper['abstract'] = abstract
            print(f"Abstract extracted successfully")
        else:
            print(f"Failed to extract abstract")
        print("-" * 50)
    
    # Save the updated papers back to the JSON file
    with open('./results/icml2025.json', 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=4)
    
    print(f"Updated {len(papers)} papers with abstracts")

def test_extract_abstracts():
    """Test extracting abstracts from the first paper in the JSON file"""
    # Read the JSON file
    with open('./results/icml2025.json', 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    # Test with the first paper only
    if papers:
        first_paper = papers[7]
        print(f"Testing with paper: {first_paper['title']}")
        print(f"URL: {first_paper['url']}")
        abstract = extract_abstract_from_url(first_paper['url'])
        if abstract:
            print(f"Abstract: {abstract}")
        else:
            print("No abstract found")

# test_extract_abstracts()
add_abstracts_to_papers()


#%%
from typing import Literal, Dict, Any
import openai
from openai import OpenAIError
import dotenv
import os
import json

# 加载环境变量并验证
dotenv.load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables")

# 初始化 OpenAI 客户端
try:
    evaluator = openai.OpenAI(api_key=api_key)
except OpenAIError as e:
    raise ValueError(f"Failed to initialize OpenAI client: {e}")

# 论文方向分类模板
DIRECTION_TEMPLATE = """You are an expert in machine learning research. I will provide you with a paper title and abstract, and you need to determine if this paper is related to a specific research direction.

Research Direction: {direction}

Please analyze the following paper and determine if it is related to the specified research direction. Consider the main contributions, methods, and applications described in the paper.

Paper Title: {title}
Paper Abstract: {abstract}
Research Direction: {direction}

Reply with 'yes' if the paper is related to the specified research direction, or 'no' if it is not related. Only reply with 'yes' or 'no', no explanations needed."""

def classify_paper_direction(title: str, abstract: str, direction: str = "LLM Safety", model: str = "gpt-4o-mini") -> bool:
    """
    使用OpenAI模型判断论文是否与指定方向相关
    
    Args:
        title: 论文标题
        abstract: 论文摘要
        direction: 研究方向，默认为"LLM Safety"
        model: 使用的OpenAI模型，默认为"gpt-4o-mini"
    
    Returns:
        bool: True表示相关，False表示不相关
    
    Raises:
        ValueError: 如果输入为空或API返回无效结果
        OpenAIError: 如果API调用失败
    """
    if not title or not abstract:
        raise ValueError("Title and abstract cannot be empty")
    
    prompt = DIRECTION_TEMPLATE.format(
        direction=direction,
        title=title,
        abstract=abstract
    )
    
    try:
        result = evaluator.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            timeout=15.0
        )
        
        # 验证返回结果
        classification = result.choices[0].message.content.strip().lower()
        if classification not in {"yes", "no"}:
            if "yes" in classification:
                return True
            elif "no" in classification:
                return False
            else:
                print(f"Title: {title}")
                print(f"Classification result: {classification}")
                raise ValueError(f"Invalid classification result: {classification}")
        
        return classification == "yes"
    
    except OpenAIError as e:
        raise OpenAIError(f"Failed to classify paper: {e}")
    except Exception as e:
        print(f"Classification result: {classification}")
        raise ValueError(f"Unexpected error during classification: {e}")

def add_direction_classification_to_papers(direction: str = "LLM Safety"):
    """为所有论文添加方向分类标注"""
    # 读取JSON文件
    with open('./results/icml2025.json', 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    # 为每篇论文进行方向分类
    for i, paper in enumerate(papers):
        direction_key = f"is_{direction.lower().replace(' ', '_')}"
        
        if direction_key in paper:
            print(f"Skipping paper {i+1}/{len(papers)}: {paper['title']} (already classified)")
            continue
            
        if 'abstract' not in paper or not paper['abstract']:
            print(f"Skipping paper {i+1}/{len(papers)}: {paper['title']} (no abstract)")
            paper[direction_key] = False
            continue
            
        print(f"Classifying paper {i+1}/{len(papers)}: {paper['title']}")
        
        try:
            is_related = classify_paper_direction(
                title=paper['title'],
                abstract=paper['abstract'],
                direction=direction
            )
            paper[direction_key] = is_related
            print(f"Classification result: {'Related' if is_related else 'Not related'}")
        except Exception as e:
            print(f"Failed to classify paper: {e}")
            paper[direction_key] = False
        
        print("-" * 50)
    
    # 保存更新后的论文数据
    with open('./results/icml2025.json', 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=4, ensure_ascii=False)
    
    # 统计结果
    direction_key = f"is_{direction.lower().replace(' ', '_')}"
    related_count = sum(1 for paper in papers if paper.get(direction_key, False))
    print(f"Classification completed: {related_count}/{len(papers)} papers are related to {direction}")

def test_direction_classification():
    """测试方向分类功能"""
    # 读取JSON文件
    with open('./results/icml2025.json', 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    # 测试前几篇有摘要的论文
    test_papers = [paper for paper in papers if 'abstract' in paper and paper['abstract']][:3]
    
    for i, paper in enumerate(test_papers):
        print(f"Testing paper {i+1}: {paper['title']}")
        try:
            is_related = classify_paper_direction(
                title=paper['title'],
                abstract=paper['abstract']
            )
            print(f"Result: {'Related to LLM Safety' if is_related else 'Not related to LLM Safety'}")
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 50)

# 运行方向分类
# test_direction_classification()
add_direction_classification_to_papers("LLM Safety")


#%%
def filter_safety_related_papers():
    """筛选出与LLM Safety相关的论文并保存到单独文件"""
    # 读取原始JSON文件
    with open('./results/icml2025.json', 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    # 筛选出is_llm_safety为true的论文
    safety_related_papers = [paper for paper in papers if paper.get('is_llm_safety', False)]
    
    # 保存到新文件
    with open('./results/icml2025_safety_related.json', 'w', encoding='utf-8') as f:
        json.dump(safety_related_papers, f, indent=4, ensure_ascii=False)
    
    print(f"Filtered {len(safety_related_papers)} safety-related papers from {len(papers)} total papers")
    print(f"Results saved to ./results/icml2025_safety_related.json")

# 运行筛选功能
filter_safety_related_papers()

#%%
def filter_steering_related_papers():
    """筛选出与steering相关的论文并保存到单独文件"""
    # 读取原始JSON文件
    with open('./results/icml2025.json', 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    # 筛选出title或abstract中包含"steering"的论文
    steering_related_papers = []
    for paper in papers:
        title = paper.get('title', '').lower()
        abstract = paper.get('abstract', '').lower()
        
        if 'steering' in title or 'steering' in abstract:
            steering_related_papers.append(paper)
    
    # 保存到新文件
    with open('./results/icml2025_steering_related.json', 'w', encoding='utf-8') as f:
        json.dump(steering_related_papers, f, indent=4, ensure_ascii=False)
    
    print(f"Filtered {len(steering_related_papers)} steering-related papers from {len(papers)} total papers")
    print(f"Results saved to ./results/icml2025_steering_related.json")

# 运行筛选功能
filter_steering_related_papers()


#%%

import arxiv
import json
import time
from typing import Optional, Dict, Any
from difflib import SequenceMatcher

def calculate_title_similarity(title1: str, title2: str) -> float:
    """Calculate similarity between two titles using SequenceMatcher"""
    return SequenceMatcher(None, title1.lower(), title2.lower()).ratio()

def search_arxiv_with_library(title: str) -> Optional[Dict[str, Any]]:
    """Search for papers using arxiv library"""
    try:
        # Create search client
        client = arxiv.Client()
        
        # Build search query
        search = arxiv.Search(
            query=f'ti:"{title}"',
            max_results=5,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        # Execute search
        results = list(client.results(search))
        
        if results:
            # Find all results with high similarity
            similar_papers = []
            for paper in results:
                similarity = calculate_title_similarity(title, paper.title)
                if similarity > 0.6:  # Lower threshold to capture more potential matches
                    similar_papers.append({
                        'arxiv_id': paper.entry_id.split('/')[-1],
                        'arxiv_url': paper.entry_id,
                        'arxiv_title': paper.title,
                        'authors': [str(author) for author in paper.authors],
                        'published': paper.published.isoformat() if paper.published else None,
                        'similarity_score': similarity
                    })
            
            # Sort by similarity score
            similar_papers.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            if similar_papers:
                return {
                    'best_match': similar_papers[0],
                    'all_matches': similar_papers
                }
    except Exception as e:
        print(f"Search error: {e}")
        return None
    
    return None

def process_papers_with_arxiv_lib(json_file_path: str):
    """Process paper data using arxiv library"""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    processed_count = 0
    found_count = 0
    
    for i, paper in enumerate(papers):
        print(f"[{i+1}/{len(papers)}] Searching: {paper['title'][:60]}...")
        
        arxiv_info = search_arxiv_with_library(paper['title'])
        
        if arxiv_info:
            best_match = arxiv_info['best_match']
            all_matches = arxiv_info['all_matches']
            
            # Store the best match if similarity is high enough
            if best_match['similarity_score'] > 0.8:
                paper['arxiv_url'] = best_match['arxiv_url']
                paper['arxiv_id'] = best_match['arxiv_id']
                paper['arxiv_title'] = best_match['arxiv_title']
                paper['arxiv_authors'] = best_match['authors']
                paper['arxiv_published'] = best_match['published']
                paper['similarity_score'] = best_match['similarity_score']
                found_count += 1
                print(f"  ✓ Found: {best_match['arxiv_url']} (similarity: {best_match['similarity_score']:.3f})")
            else:
                print(f"  ~ Low similarity: {best_match['similarity_score']:.3f}")
            
            # Store all potential matches for review
            paper['arxiv_all_matches'] = all_matches
        else:
            print(f"  ✗ Not found")
        
        processed_count += 1
        time.sleep(1)  # Avoid API rate limits
    
    # Save results
    output_file = json_file_path.replace('.json', '_with_arxiv.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    
    print(f"\nProcessing completed:")
    print(f"- Total papers processed: {processed_count}")
    print(f"- Papers with high-similarity matches found: {found_count}")
    print(f"- Results saved to: {output_file}")

# Process the steering-related papers
# process_papers_with_arxiv_lib('./results/icml2025_steering_related.json')
process_papers_with_arxiv_lib('./results/icml2025_safety_related.json')


#%%
# Filter papers to keep only those with arxiv_url
def filter_papers_with_arxiv(json_file_path):
    """Filter papers to keep only those that have arxiv_url"""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    # Filter papers that have arxiv_url
    filtered_papers = [paper for paper in papers if 'arxiv_url' in paper and paper['arxiv_url']]
    
    # Save filtered results
    output_file = json_file_path.replace('.json', '_filtered.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_papers, f, indent=2, ensure_ascii=False)
    
    print(f"\nFiltering completed:")
    print(f"- Original papers: {len(papers)}")
    print(f"- Papers with arxiv_url: {len(filtered_papers)}")
    print(f"- Filtered results saved to: {output_file}")

# Filter the safety-related papers to keep only those with arxiv_url
filter_papers_with_arxiv('./results/icml2025_safety_related_with_arxiv.json')
filter_papers_with_arxiv('./results/icml2025_steering_related_with_arxiv.json')

#%%
import requests
import os
from urllib.parse import urlparse
import json
import time

def download_arxiv_pdfs(json_file_path, output_dir='./papers'):
    """Download PDF files from arXiv for papers in the JSON file"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    downloaded_count = 0
    skipped_count = 0
    failed_count = 0
    
    for i, paper in enumerate(papers):
        if 'arxiv_url' not in paper or not paper['arxiv_url']:
            print(f"Paper {i+1}: No arXiv URL found, skipping")
            skipped_count += 1
            continue
        
        arxiv_url = paper['arxiv_url']
        print(f"Processing paper {i+1}/{len(papers)}: {paper['title']}")
        print(f"  ArXiv URL: {arxiv_url}")
        
        # Extract arXiv ID from URL
        # URL format: http://arxiv.org/abs/2411.12768v1
        try:
            arxiv_id = arxiv_url.split('/')[-1]
            if not arxiv_id:
                print(f"  ✗ Could not extract arXiv ID from URL")
                failed_count += 1
                continue
            
            # Construct PDF URL
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            
            # Create filename
            # safe_title = "".join(c for c in paper['title'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            # safe_title = safe_title.replace(' ', '_')[:100]  # Limit length
            filename = f"{arxiv_id}.pdf"
            filepath = os.path.join(output_dir, filename)
            
            # Check if file already exists
            if os.path.exists(filepath):
                print(f"  ✓ File already exists: {filename}")
                skipped_count += 1
                continue
            
            # Download PDF
            print(f"  → Downloading from: {pdf_url}")
            response = requests.get(pdf_url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Save PDF file
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"  ✓ Downloaded: {filename}")
            downloaded_count += 1
            
        except requests.exceptions.RequestException as e:
            print(f"  ✗ Download failed: {e}")
            failed_count += 1
        except Exception as e:
            print(f"  ✗ Error processing paper: {e}")
            failed_count += 1
        
        # Add delay to be respectful to arXiv servers
        time.sleep(1)
        print("-" * 50)
    
    print(f"\nDownload completed:")
    print(f"- Total papers: {len(papers)}")
    print(f"- Successfully downloaded: {downloaded_count}")
    print(f"- Skipped (already exists or no URL): {skipped_count}")
    print(f"- Failed: {failed_count}")
    print(f"- PDFs saved to: {output_dir}")

# Download PDFs
# download_arxiv_pdfs('./results/icml2025_safety_related_with_arxiv_filtered.json')
download_arxiv_pdfs('./results/icml2025_steering_related_with_arxiv_filtered.json')

#%%
import json
import os

def generate_markdown_table(json_path, output_md_path, table_title):
    """Generate markdown table from JSON file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    markdown_content = f"# {table_title}\n\n"
    markdown_content += "| Title | ArXiv Link | PDF Link |\n"
    markdown_content += "|-------|------------|----------|\n"
    
    for paper in papers:
        title = paper.get('title', 'N/A')
        arxiv_url = paper.get('arxiv_url', '')
        arxiv_id = paper.get('arxiv_id', '')
        
        # Create links
        if arxiv_url:
            arxiv_link = f"[ArXiv]({arxiv_url})"
        else:
            arxiv_link = "N/A"
        
        if arxiv_id:
            pdf_link = f"[PDF](./papers/{arxiv_id}.pdf)"
        else:
            pdf_link = "N/A"
        
        markdown_content += f"| {title} | {arxiv_link} | {pdf_link} |\n"
    
    # Write to markdown file
    with open(output_md_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"Generated markdown table: {output_md_path}")

# Generate markdown tables
path1 = "/home/shenc/Desktop/NUS/crawl/results/icml2025_safety_related_with_arxiv_filtered.json"
path2 = "/home/shenc/Desktop/NUS/crawl/results/icml2025_steering_related_with_arxiv_filtered.json"

generate_markdown_table(path1, "icml2025_safety_papers.md", "ICML 2025 Safety Related Papers")
generate_markdown_table(path2, "icml2025_steering_papers.md", "ICML 2025 Steering Related Papers")
