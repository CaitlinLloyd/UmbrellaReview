import anthropic
import os
import glob
import argparse
from pathlib import Path

# Initialize the client
client = anthropic.Anthropic(api_key=$ANTHROPIC_KEY)


def debug_directory_contents(directory_path):
    """Debug function to show all files in a directory"""
    if not os.path.isdir(directory_path):
        print(f"DEBUG: '{directory_path}' is not a directory")
        return
        
    print(f"DEBUG: All files in directory '{directory_path}':")
    all_files = os.listdir(directory_path)
    
    if not all_files:
        print("DEBUG: Directory is empty")
        return
        
    for file in sorted(all_files):
        full_path = os.path.join(directory_path, file)
        if os.path.isfile(full_path):
            print(f"DEBUG:   FILE: {file}")
        else:
            print(f"DEBUG:   DIR:  {file}/")
    
    # Show specifically .txt files
    txt_files = [f for f in all_files if f.endswith('.txt')]
    print(f"DEBUG: Found {len(txt_files)} .txt files specifically:")
    for txt_file in sorted(txt_files):
        print(f"DEBUG:   - {txt_file}")

def find_article_files(input_path, file_pattern="*.txt"):
    """Find all article files based on input path and pattern"""
    print(f"DEBUG: Looking for files with input_path='{input_path}' and pattern='{file_pattern}'")
    
    if os.path.isfile(input_path):
        print(f"DEBUG: Single file provided: {input_path}")
        return [input_path]
    elif os.path.isdir(input_path):
        pattern = os.path.join(input_path, file_pattern)
        print(f"DEBUG: Directory provided. Full search pattern: {pattern}")
        files = sorted(glob.glob(pattern))
        print(f"DEBUG: Found {len(files)} files matching pattern")
        for f in files:
            print(f"DEBUG:   - {f}")
        return files
    else:
        print(f"DEBUG: Pattern provided: {input_path}")
        files = sorted(glob.glob(input_path))
        print(f"DEBUG: Found {len(files)} files matching pattern")
        for f in files:
            print(f"DEBUG:   - {f}")
        return files

def read_and_prepare_articles(file_paths):
    """Read all articles and prepare metadata"""
    articles_data = []
    successful_reads = 0
    
    print(f"Found {len(file_paths)} files to analyze:")
    for f in file_paths:
        print(f"  - {f}")
    
    for i, file_path in enumerate(file_paths, 1):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                filename = Path(file_path).stem
                
                article_data = {
                    'number': i,
                    'filename': filename,
                    'path': file_path,
                    'length': len(content),
                    'content': content
                }
                
                articles_data.append(article_data)
                successful_reads += 1
                print(f"  ✓ Read Article {i}: {filename} ({len(content)} characters)")
                
        except Exception as e:
            print(f"  ✗ Error reading {file_path}: {e}")
            continue
    
    print(f"\nSuccessfully read {successful_reads} out of {len(file_paths)} files.")
    
    # Show article size distribution
    if articles_data:
        total_chars = sum(article['length'] for article in articles_data)
        print(f"Total content: {total_chars:,} characters")
        print("Article sizes:")
        for article in articles_data:
            percentage = (article['length'] / total_chars) * 100
            print(f"  Article {article['number']}: {article['length']:,} chars ({percentage:.1f}%)")
    
    return articles_data, total_chars

def create_batches(articles_data, max_chars):
    """Create batches of articles that fit within character limits"""
    batches = []
    current_batch = []
    current_chars = 0
    
    # Reserve space for prompt overhead (approximately 3000 characters)
    effective_limit = max_chars - 3000
    
    for article in articles_data:
        # Estimate space needed for this article (content + formatting)
        article_overhead = len(f"=== ARTICLE {article['number']}: {article['filename']} ===\n") + 85
        total_needed = article['length'] + article_overhead
        
        if current_chars + total_needed > effective_limit and current_batch:
            # Start new batch
            batches.append(current_batch)
            current_batch = [article]
            current_chars = total_needed
        else:
            # Add to current batch
            current_batch.append(article)
            current_chars += total_needed
    
    # Add final batch
    if current_batch:
        batches.append(current_batch)
    
    return batches

def build_prompt_for_batch(articles_batch, batch_num=None, total_batches=None):
    """Build analysis prompt for a batch of articles"""
    batch_size = len(articles_batch)
    
    # Create content with separators
    articles_content = []
    for article in articles_batch:
        content_block = f"=== ARTICLE {article['number']}: {article['filename']} ===\n{article['content']}\n{'='*80}\n"
        articles_content.append(content_block)
    
    combined_content = "\n".join(articles_content)
    article_list = "\n".join([f"- Article {article['number']}: {article['filename']}" for article in articles_batch])
    
    # Build prompt
    if batch_num is not None and total_batches is not None:
        # This is a batch analysis
        prompt_parts = [
            f"This is BATCH {batch_num} of {total_batches} from a systematic umbrella review cluster.",
            f"Analyze these {batch_size} articles and provide:",
            "",
            "Articles in this batch:",
            article_list,
            "",
            "For this batch, provide:",
            "1. **Batch Summary**: Give an overview of the main themes for articles this batch. In particular, describe the study focus and aims",
            "2. **Population Data**: Exact Gender, SES, Race/Ethnicity for each article in this batch (or 'Not reported')",
            "3. **Study Details**: For each article, provide design, methodology, findings, and a brief 2-3 sentence summary (give the article name). Provide the effect size and confidence interval estimates for the primary analyses as well as subgroup and moderation analyses. You should describe the main findings and conclusions and the most clinically relevant findings",
            "4. **Batch Patterns**: Describe the common and discrepant features for the different articles in this batch?",
            "",
            "Articles:",
            "",
            combined_content
        ]
    else:
        # This is a single batch analysis (all articles together)
        prompt_parts = [
            f"I have {batch_size} articles from a systematic umbrella review that clustered together using BERTopic.",
            "",
            f"IMPORTANT: You must analyze ALL {batch_size} articles listed below.",
            "",
            "Articles to analyze:",
            article_list,
            "",
            "Please provide:",
            "1. **Cluster Label**: Descriptive label (max 6 words)",
            "2. **Overall Findings Summary**: Describe the main conclusion for each of the articles in this set. What are the main conclusions and themes across all studies? Please provide effect sizes and confidence intervals for the primary, sub-group and moderation analyses and describe which study they came from. You should describe the most scientifically important and clinically relevant findings in the set of articles",
            "3. **Exact Population Data Per Article**: Gender, SES, Race/Ethnicity for each article (or 'Not reported')",
            "4. **Subcategory Analysis**: Identify distinct subcategories that might warrant separate labels",
            "5. **Study Comparison**: What are the similarities and differences between each article in this set. Be specific when referring to findings from a specific article (provide the author name)",
            "6. **Article-Specific Information**: Brief summary of each article (provide its name). Give details of the statistical findings (i.e., effect size and confidence interval for the main meta-analysis as well as sub-group and moderation analyses",
            "7. **Synthesis**: How do these articles collectively further our understanding of a given topic"
            "",
            f"Here are the {batch_size} articles:",
            "",
            combined_content
        ]
    
    return "\n".join(prompt_parts)

def analyze_single_batch(articles_data):
    """Analyze all articles in a single batch"""
    print("Content size acceptable. Processing all articles together...")
    
    prompt = build_prompt_for_batch(articles_data)
    
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8000,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        return f"Error in API call: {e}"

def analyze_multiple_batches(articles_data, max_chars):
    """Analyze articles in multiple batches and synthesize"""
    print(f"Content exceeds {max_chars:,} characters. Using batching approach...")
    
    # Create batches
    batches = create_batches(articles_data, max_chars)
    print(f"Created {len(batches)} batches")
    
    batch_results = []
    
    # Analyze each batch
    for i, batch in enumerate(batches, 1):
        print(f"\nAnalyzing batch {i}/{len(batches)} ({len(batch)} articles)...")
        
        prompt = build_prompt_for_batch(batch, i, len(batches))
        
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=8000,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            batch_results.append({
                'batch_num': i,
                'articles': [article['filename'] for article in batch],
                'result': response.content[0].text
            })
            print(f"  ✓ Batch {i} complete")
        except Exception as e:
            print(f"  ✗ Error in batch {i}: {e}")
            batch_results.append({
                'batch_num': i,
                'articles': [article['filename'] for article in batch],
                'result': f"Error: {e}"
            })
    
    # Synthesize results
    print(f"\nSynthesizing results from {len(batches)} batches...")
    return synthesize_batch_results(batch_results, articles_data)

def synthesize_batch_results(batch_results, articles_data):
    """Synthesize results from multiple batches"""
    total_articles = len(articles_data)
    
    # Create summary of all articles
    article_summary = f"Total articles: {total_articles}\nArticle list:\n"
    for article in articles_data:
        article_summary += f"- Article {article['number']}: {article['filename']} ({article['length']:,} chars)\n"
    
    # Combine batch results
    synthesis_parts = [
        "=== SYNTHESIZED ANALYSIS FROM MULTIPLE BATCHES ===",
        "",
        article_summary,
        ""
    ]
    
    for batch in batch_results:
        synthesis_parts.append(f"--- BATCH {batch['batch_num']} RESULTS ---")
        synthesis_parts.append(f"Articles: {', '.join(batch['articles'])}")
        synthesis_parts.append("")
        synthesis_parts.append(batch['result'])
        synthesis_parts.append("")
    
    combined_results = "\n".join(synthesis_parts)
    
    # Create synthesis prompt
    synthesis_prompt = f"""I have batch analysis results from {len(batch_results)} batches covering {total_articles} total articles from a systematic umbrella review cluster. Please synthesize these into a final analysis.

Please provide:
1. **Overall Cluster Label**: Descriptive label for all articles (max 6 words)
Indicate whether there are subcategories of studies within this collection of articles.

2. **Overall Findings Summary**: 
What are the main conclusions and themes across all studies? 
Please provide effect sizes and confidence intervals for the main analyses, and indicate the findings of subgroup or moderation analyses (again using effect size and confidence interval statistics). 
Please provide information about the study or studies being referred to for each point. 
You should describe the most scientifically relevant as well as the most clinically relevant findings.

3. **Population Analysis**: 
   For each demographic category, specify which articles (by number/name) include this information:
   - Gender distribution and representation (specify which articles report gender data)
   - Socioeconomic status of participants (specify which articles include SES data)
   - Race and ethnicity breakdown (specify which articles report race/ethnicity data)
   - Any notable demographic gaps or biases across the studies

4. **Study Comparison**:
   - Similarities in study designs, methodologies, and approaches (specify which articles share these features)
   - Differences in populations, settings, and methods (specify which articles differ and how)
   - Convergent vs. divergent findings across the collection of studies (specify which articles support or contradict each finding)
   - Describe differences in the conclusions of the studies, and similarities in the conclusions.

5. **Article-Specific Information**:
   For each article, provide a brief summary including:
   - Main research question/objective
   - Study design and methodology
   - Key population characteristics
   - Primary findings (give effect size and confidence intervals)
   - Outcomes of subgroup or moderator analyses (give effect size and confidence intervals)
   - Notable limitations

Here are the batch results to synthesize:

{combined_results}

Please create a coherent synthesis that treats this as a single cluster analysis."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8000,
            temperature=0.3,
            messages=[{"role": "user", "content": synthesis_prompt}]
        )
        return response.content[0].text
    except Exception as e:
        return f"Error in synthesis: {e}\n\n=== RAW BATCH RESULTS ===\n{combined_results}"

def analyze_article_cluster(file_paths, cluster_name="articles", max_chars=100000):
    """
    Main function to analyze a cluster of articles
    Automatically handles batching if content is too large
    """
    if not file_paths:
        return "No files found to analyze."
    
    # Read and prepare all articles
    articles_data, total_chars = read_and_prepare_articles(file_paths)
    
    if not articles_data:
        return "No files could be read successfully."
    
    # Decide whether to batch or not
    if total_chars > max_chars:
        return analyze_multiple_batches(articles_data, max_chars)
    else:
        return analyze_single_batch(articles_data)

def main():
    """Main function with command line argument support"""
    parser = argparse.ArgumentParser(description="Analyze article clusters using Claude API")
    parser.add_argument("input", help="Input file, directory, or glob pattern")
    parser.add_argument("--pattern", default="*.txt", help="File pattern for directory input (default: *.txt)")
    parser.add_argument("--output", help="Output file name")
    parser.add_argument("--cluster-name", default="articles", help="Name for this cluster")
    parser.add_argument("--max-chars", type=int, default=100000, help="Maximum characters per batch (default: 100000)")
    
    args = parser.parse_args()
    
    # Debug directory contents first
    if os.path.isdir(args.input):
        debug_directory_contents(args.input)
    
    # Find files
    article_files = find_article_files(args.input, args.pattern)
    
    if not article_files:
        print(f"No files found matching: {args.input}")
        return
    
    # Run the analysis
    print(f"Analyzing {len(article_files)} articles in cluster '{args.cluster_name}'...")
    result = analyze_article_cluster(file_paths=article_files, cluster_name=args.cluster_name, max_chars=args.max_chars)
    
    # Determine output filename
    output_filename = args.output or f"{args.cluster_name}_analysis_results.txt"
    
    # Save results
    with open(output_filename, "w", encoding="utf-8") as output_file:
        output_file.write(f"Cluster Analysis Results: {args.cluster_name}\n")
        output_file.write(f"Number of articles: {len(article_files)}\n")
        output_file.write("="*60 + "\n\n")
        output_file.write(result)
    
    print(f"Analysis complete! Results saved to '{output_filename}'")
    print("\n" + "="*60)
    print(result)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        main()
    else:
        print("Usage examples:")
        print("python script.py /path/to/articles/")
        print("python script.py /path/to/articles/ --max-chars 600000")
        print("python script.py /path/to/articles/ --cluster-name 'my_cluster'")
