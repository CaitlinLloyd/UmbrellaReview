"""
Dual Extraction API Call - AMSTAR + Study Data (Enhanced with Supplement/Protocol Files)
Replicates the dual extraction approach for systematic review quality assessment
"""

import requests
import json
import pandas as pd
from datetime import datetime
import time
import sys
import os

article=sys.argv[1]
# Get article path from command line
my_key=$ANTHROPIC_KEY

file_name=sys.argv[2]

class DualExtractionAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": my_key,
            "anthropic-version": "2023-06-01"
        }

    def load_supplement_files(self, article_path):
        """Load supplement and protocol files if they exist"""
        base_name = os.path.splitext(os.path.basename(article_path))[0]
        base_dir = os.path.dirname(article_path)
        
        # Try different extensions for supplement file
        supp_content = ""
        supp_extensions = ['.txt', '.pdf', '.docx', '']
        for ext in supp_extensions:
            supp_path = os.path.join(base_dir,"Supplements",f"{base_name}_supp{ext}")
            if os.path.exists(supp_path):
                try:
                    with open(supp_path, 'r', encoding='utf-8', errors='ignore') as f:
                        supp_content = f.read()
                    print(f"Found supplement file: {supp_path}")
                    break
                except:
                    continue
        
        # Try different extensions for protocol file  
        protocol_content = ""
        for ext in supp_extensions:
            protocol_path = os.path.join(base_dir,"Protocols",f"{base_name}_protocol{ext}")
            if os.path.exists(protocol_path):
                try:
                    with open(protocol_path, 'r', encoding='utf-8', errors='ignore') as f:
                        protocol_content = f.read()
                    print(f"Found protocol file: {protocol_path}")
                    break
                except:
                    continue
                    
        return supp_content, protocol_content

    def extract_amstar_assessment(self, article_text, qc_questions, supp_content="", protocol_content=""):
        """
        First API call: Extract AMSTAR 2 quality assessments with supplement/protocol info
        """
        
        # Filter for AMSTAR-related questions
        amstar_questions = [q for q in qc_questions if self.is_amstar_question(q['Field'])]
        
        # Combine all available text
        combined_text = f"MAIN ARTICLE:\n{article_text}"
        if supp_content:
            combined_text += f"\n\nSUPPLEMENT MATERIAL:\n{supp_content}"
        if protocol_content:
            combined_text += f"\n\nPROTOCOL:\n{protocol_content}"
        
        amstar_prompt = f"""
You are conducting an AMSTAR 2 quality assessment of a systematic review/meta-analysis.

IMPORTANT: Check ALL provided documents (main article, supplements, and protocol) for information.

AMSTAR 2 has 16 items. For each item below, provide a response in this EXACT JSON format:
[
  {{
    "Section": "AMSTAR_Items",
    "Field": "Item_1", 
    "Value": "Yes/No/Partial Yes. [Provide a detailed explanation with evidence from the paper/supplement/protocol]"
  }}
]
Note that not all items can have a "Partial" response; this is limited to questions: 2,4,7,8,9. For information about how to differentiate "Partial Yes" from "Yes" see the SPECIAL ATTENTION section below.

AMSTAR 2 ITEMS TO ASSESS:
1. Did the research questions and inclusion criteria include components of PICO? Did this include: Population, Intervention, Comparator and Outcome?
2. Did the report contain an explicit statement that methods were established prior to conduct? Did the report justify any significant deviations from the protocol? (CHECK SUPPLEMENTS/PROTOCOL)
3. Did the authors explain their selection of study designs for inclusion? This should include reasons for including randomized controlled trials or non-randomized studies or both
4. Did the authors use a comprehensive literature search strategy? (CHECK SUPPLEMENTS)
5. Did the authors perform study selection in duplicate?
6. Did the authors perform data extraction in duplicate? This should involve either 1)  at least two reviewers achieved consensus on which data to extract from included studies; or 2) two reviewers extracted data from a sample of eligible studies and achieved good agreement (at least 80 percent), with the remainder extracted by one reviewer.
7. Did the authors provide a list of excluded studies and justify exclusions? (CHECK SUPPLEMENTS)
8. Did the authors describe included studies in adequate detail? This includes a description of populations, interventions, comparator interventions, outcomes, and research designs.
9. Did the authors use satisfactory technique for assessing risk of bias? (CHECK SUPPLEMENTS)
10. Did the authors report on sources of funding for included studies?  Must have reported on the sources of funding for individual studies included in the review.
11. If meta-analysis performed, did authors use appropriate statistical methods? This involves: 
justifying combining the data in a meta-analysis; using an appropriate weighted technique to combine study results and adjusted for heterogeneity if present; and investigating causes of heterogeneity (randomized controlled study only).
For a non-randomized study authors must justify combining raw estimates or only combine estimates adjusted for confounding, and report separate estimates from randomized controlled studies and non-randomized studies if applicable. 
12. If meta-analysis performed, did authors assess impact of risk of bias? This should include sub-analyses that examine whether findings are different when only low risk of bias studies are included, or examine the moderating effect of risk of bias on findings.
13. Did the authors account for risk of bias when interpreting results?
14. Did the authors provide satisfactory explanation for heterogeneity observed? If heterogeneity was present the authors performed an investigation of sources of any heterogeneity in the results and discussed the impact of this on the results of the review
15. If quantitative synthesis performed, did authors investigate publication bias? Performed graphical or statistical tests for publication bias and discussed the likelihood and magnitude of impact of publication bias
16. Did the authors report potential sources of conflict of interest? The authors reported no competing interests OR The authors described their funding sources and how they managed potential conflicts of interest

ASSESSMENT CRITERIA:
- "Yes" = Criterion clearly met
- "No" = Criterion clearly not met  
- "Partial Yes" = Criterion partially met or with limitations
- "Not applicable" = Does not apply to this study type
- Note that only items 2,4,7,8,9 can have the response "Partial Yes"; for other items only "Yes" or "No" is possible

SPECIAL ATTENTION: 

- Items 2, 4, 7, 9 often have details in supplements or protocols
- For Item 2, for the rating to be "Partial Yes" the authors must have stated that they had a written protocol or guide that included ALL the following: review question(s); a search strategy; inclusion/exclusion criteria; a risk of bias assessment
- For Item 2, for the rating to be "Yes" criteria for partial yes should be fulfilled and the protocol should be registered and should also have specified: a meta-analysis/synthesis plan; plan for investigating causes of heterogeneity
- For Item 4, for the rating to be "Partial Yes" the authors must have searched at least 2 databases (relevant to research question); provided key word and/or search strategy; and   justified publication restrictions (e.g. language)
- For Item 4, for the rating to be "Yes" criteria for partial yes should be fulfilled and the authors should have: searched the reference lists / bibliographies of included studies; searched trial/study registries; included/consulted content experts in the field; where relevant, searched for grey literature; conducted search within 24 months of completion of the review
- For Item 7; for the rating to be "Partial Yes" authors must have provided a list of all potentially relevant studies that were read in full-text form but excluded from the review
- For Item 7; for the rating to be "Yes" criteria for partial should be fulfilled, and the authors must have justified the exclusion from the review of each potentially relevant study 
- For Item 8; to move from a "Partial Yes" to a "Yes" the article should give a detailed description of population, intervention, comparator and describe each study’s setting and the timeframe for follow-up
- For Item 9; for the rating to be "Partial Yes", the following should be satisfied:
FOR RCT: risk of bias from unconcealed allocation, and lack of blinding of patients and assessors when assessing outcomes (unnecessary for objective outcomes such as all-cause mortality) should have been assessed
FOR non-randomized study: risk of bias from confounding and selection bias should have been assessed
For the rating to be "Yes", in addition to the criteria for "Partial Yes", the article should have assessed risk of bias from:
RCT:  allocation sequence that was not truly random, and selection of the reported result from among multiple measurements or analyses of a specified outcome
non-randomized study: methods used to ascertain exposures and outcomes, and selection of the reported result from among multiple measurements or analyses of a specified outcome
- Search supplement/protocol text carefully for registration info, search strategies, excluded study lists, and bias assessment details
- Cite which document (main/supplement/protocol) contains the evidence


COMBINED TEXT (Main Article + Supplements + Protocol):
{combined_text}

Return ONLY the JSON array with assessments for all 16 items.
"""

        return self._make_api_call(amstar_prompt)
    
    def extract_study_data(self, article_text, qc_questions):
        """
        Second API call: Extract specific study data and results
        """
        
        # Filter for study data questions  
        study_questions = [q for q in qc_questions if not self.is_amstar_question(q['Field'])]
        
        # Get the exact field names from QC sheet
        study_fields = [q['Field'] for q in study_questions]
        
        study_prompt = f"""
Extract specific data from this research study and format as JSON array.
Each object must have exactly these keys: "Section", "Field", and "Value".

CRITICAL: Use these EXACT field names in your response:
{json.dumps(study_fields, indent=2)}

For each field above, provide:
- "Section": The appropriate section name
- "Field": The EXACT field name from the list above  
- "Value": The extracted data from the article

EXTRACTION GUIDELINES:
- For country of article, report the country in which the corresponding author is based
- Report what the findings of the statistical analyses were for primary and second/third outcomes, the "Focus and main finding for outcome" field.
- For statistical results, provide EXACT numbers (OR, CI, p-values). You need to identify the primary outcome and the second and third outcomes of focus. For each one, provide the effect size along with the confidence interval and p-value. Be as comprehensive as you can.
- You do not need to report the statistical results of any meta-regression; only all meta-analytic findings.
- Provide information about the eating disorder or disorders of focus
- Describe the included population of the study, and specifically their sex, gender, race, ethnicity, age and socioeconomic status 
- Report the number of studies in each separate meta-analysis
- For sample sizes, provide exact participant counts
- For missing data, write "Not reported" or "Not available"
- For yes/no questions, write "Yes", "No", or "Unclear"


ARTICLE TEXT:
{article_text}

Return ONLY a valid JSON array with one object for each field listed above.
"""

        return self._make_api_call(study_prompt)
    
    def _make_api_call(self, prompt, max_retries=3):
        """Make API call to Claude with retry logic for rate limits"""
        payload = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 4000,
            "messages": [
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(self.base_url, headers=self.headers, json=payload)
                response.raise_for_status()
                content = response.json()['content'][0]['text']
                
                # Clean JSON response
                if content.startswith('```json'):
                    content = content.split('```json')[1].split('```')[0]
                elif content.startswith('```'):
                    content = content.split('```')[1].split('```')[0]
                    
                content = content.strip()
                if not content:
                    print("API call failed: Empty response content")
                    return []

                try:
                    return json.loads(content)
                except json.JSONDecodeError as json_err:
                    print(f"API call failed: Invalid JSON response - {str(json_err)}")
                    print(f"Response content: {content[:200]}...")
                    return []
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Rate limit error
                    wait_time = (attempt + 1) * 30  # Exponential backoff: 30, 60, 90 seconds
                    print(f"Rate limit exceeded. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"HTTP error: {str(e)}")
                    if hasattr(e, 'response'):
                        print(f"Response status: {e.response.status_code}")
                        print(f"Response text: {e.response.text}")
                    return []
            except Exception as e:
                print(f"API call failed: {str(e)}")
                return []
        
        print(f"Failed after {max_retries} attempts")
        return []
    
    def is_amstar_question(self, field):
        """Determine if a question is AMSTAR-related"""
        f = field.lower()
        
        # Check for explicit AMSTAR item patterns
        for i in range(1, 17):
            if f"item_{i}" in f or f"item {i}" in f:
                return True
        
        # Check for AMSTAR-specific content
        amstar_keywords = [
            'pico components',
            'protocol prior', 
            'comprehensive search',
            'selection duplicate',
            'extraction duplicate', 
            'excluded studies justify',
            'adequate detail',
            'risk of bias individual',
            'funding sources',
            'appropriate statistical',
            'impact risk of bias',
            'account risk of bias',
            'heterogeneity explanation',
            'publication bias investigation',
            'conflict interest'
        ]
        
        return any(keyword in f for keyword in amstar_keywords)
    
    def calculate_amstar_overall_rating(self, amstar_results):
        """Calculate AMSTAR 2 overall confidence rating based on critical domains"""
        
        # Critical domains (items 2, 4, 7, 9, 11, 13, 15)
        critical_domains = {
            'Item_2': 2,   # Protocol registered before commencement
            'Item_4': 4,   # Adequacy of literature search
            'Item_7': 7,   # Justification for excluding studies
            'Item_9': 9,   # Risk of bias from individual studies
            'Item_11': 11, # Appropriateness of meta-analytical methods
            'Item_13': 13, # Consideration of risk of bias when interpreting
            'Item_15': 15  # Assessment of publication bias
        }
        
        # Create lookup for AMSTAR results
        amstar_lookup = {item['Field']: item['Value'] for item in amstar_results}
        
        # Count critical flaws and non-critical weaknesses
        critical_flaws = 0
        non_critical_weaknesses = 0
        
        # Check critical domains - "No" or "Partial Yes" counts as flaw
        for item_key, item_num in critical_domains.items():
            value = amstar_lookup.get(item_key, "").lower()
            if "No." in value or "Partial Yes." in value:
                critical_flaws += 1
        
        # Check non-critical domains (items 1, 3, 5, 6, 8, 10, 12, 14, 16)
        non_critical_items = ['Item_1', 'Item_3', 'Item_5', 'Item_6', 'Item_8', 'Item_10', 'Item_12', 'Item_14', 'Item_16']
        for item_key in non_critical_items:
            value = amstar_lookup.get(item_key, "").lower()
            if "No." or "Partial Yes." in value:
                non_critical_weaknesses += 1
        
        # Determine overall confidence rating based on your framework
        if critical_flaws == 0:
            if non_critical_weaknesses <= 1:
                overall_rating = "HIGH"
                confidence_description = "No or one non-critical weakness: the systematic review provides an accurate and comprehensive summary of the results of the available studies that address the question of interest"
            else:
                overall_rating = "MODERATE" 
                confidence_description = "More than one non-critical weakness: the systematic review has more than one weakness but no critical flaws. It may provide an accurate summary of the results of the available studies that were included in the review"
        elif critical_flaws == 1:
            overall_rating = "LOW"
            confidence_description = "One critical flaw with or without non-critical weaknesses: the review has a critical flaw and may not provide an accurate and comprehensive summary of the available studies that address the question of interest"
        else:  # critical_flaws > 1
            overall_rating = "CRITICALLY LOW"
            confidence_description = "More than one critical flaw with or without non-critical weaknesses: the review has more than one critical flaw and should not be relied on to provide an accurate and comprehensive summary of the available studies"
        
        return {
            'overall_rating': overall_rating,
            'confidence_description': confidence_description,
            'critical_flaws': critical_flaws,
            'non_critical_weaknesses': non_critical_weaknesses,
            'critical_domains_assessed': list(critical_domains.values())
        }

    def combine_extractions(self, amstar_results, study_results, qc_questions):
        """Combine AMSTAR and study data extractions"""
        # Create lookup dictionaries  
        amstar_lookup = {item['Field']: item['Value'] for item in amstar_results}
        study_lookup = {item['Field']: item['Value'] for item in study_results}
        
        # Create AMSTAR item mapping (Item_1 = question 1, etc.)
        amstar_questions = [q for q in qc_questions if q.get('Section') == 'AMSTAR2_Items']
        
        combined_results = []
        for question in qc_questions:
            
            # Check if this is an AMSTAR question
            if question.get('Section') == 'AMSTAR2_Items':
                # Find corresponding Item_X in amstar_results
                question_index = amstar_questions.index(question) + 1
                item_key = f"Item_{question_index}"
                value = amstar_lookup.get(item_key, f"AMSTAR assessment needed for: {question['Field']}")
                extraction_type = "AMSTAR"
            else:
                # For study data, try to match by field name
                field_name = question['Field']
                value = study_lookup.get(field_name, f"Study data needed for: {question['Field']}")
                extraction_type = "Study Data"
            
            combined_results.append({
                'Section': question.get('Section', 'Unknown'),
                'Field': question['Field'],
                'Value': value,
                'ExtractionType': extraction_type,
                'ProcessedAt': datetime.now().isoformat()
            })
        
        # Calculate and add AMSTAR overall rating
        if amstar_results:
            overall_rating_data = self.calculate_amstar_overall_rating(amstar_results)
            
            # Add overall rating as a new entry
            combined_results.append({
                'Section': 'AMSTAR2_Overall',
                'Field': 'Overall_Confidence_Rating',
                'Value': f"{overall_rating_data['overall_rating']} - {overall_rating_data['confidence_description']}",
                'ExtractionType': "AMSTAR",
                'ProcessedAt': datetime.now().isoformat()
            })
            
            # Add summary of flaws/weaknesses
            combined_results.append({
                'Section': 'AMSTAR2_Overall', 
                'Field': 'Critical_Flaws_Count',
                'Value': str(overall_rating_data['critical_flaws']),
                'ExtractionType': "AMSTAR",
                'ProcessedAt': datetime.now().isoformat()
            })
            
            combined_results.append({
                'Section': 'AMSTAR2_Overall',
                'Field': 'Non_Critical_Weaknesses_Count', 
                'Value': str(overall_rating_data['non_critical_weaknesses']),
                'ExtractionType': "AMSTAR",
                'ProcessedAt': datetime.now().isoformat()
            })
        
        return combined_results
    
    def process_article_with_qc_sheet(self, article_text, qc_csv_path, article_path):
        """
        Complete workflow: Load QC sheet, load supplements/protocol, run dual extraction, combine results
        """
        
        # Load supplement and protocol files
        supp_content, protocol_content = self.load_supplement_files(article_path)
        
        # Load QC questions
        qc_df = pd.read_csv(qc_csv_path)
        qc_questions = qc_df.to_dict('records')
        
        print(f"Loaded {len(qc_questions)} QC questions")
        print(f"AMSTAR questions: {len([q for q in qc_questions if self.is_amstar_question(q['Field'])])}")
        print(f"Study data questions: {len([q for q in qc_questions if not self.is_amstar_question(q['Field'])])}")
        if supp_content:
            print("Including supplement file in AMSTAR assessment")
        if protocol_content:
            print("Including protocol file in AMSTAR assessment")
        
        # First API call: AMSTAR assessment (with supplements/protocol)
        print("\nRunning AMSTAR assessment...")
        amstar_results = self.extract_amstar_assessment(article_text, qc_questions, supp_content, protocol_content)
        print("Waiting 60 seconds between API calls to avoid rate limits...")
        time.sleep(60)  # Increased rate limiting
        
        # Second API call: Study data extraction  
        print("Running study data extraction...")
        study_results = self.extract_study_data(article_text, qc_questions)
        time.sleep(5)  # Rate limiting
        
        # Combine results
        print("Combining extractions...")
        final_results = self.combine_extractions(amstar_results, study_results, qc_questions)
        
        return final_results
    
    def save_results(self, results, output_file):
        """Save combined results to CSV"""
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        
        # Print summary
        amstar_count = len([r for r in results if r['ExtractionType'] == 'AMSTAR'])
        study_count = len([r for r in results if r['ExtractionType'] == 'Study Data'])
        
        print(f"\nEXTRACTION SUMMARY:")
        print(f"Total fields: {len(results)}")
        print(f"AMSTAR assessments: {amstar_count}")
        print(f"Study data fields: {study_count}")

# USAGE EXAMPLE
def main():
    # Initialize the dual extraction API
    extractor = DualExtractionAPI(my_key)
    
    # Load your article text
    with open(article, "r", encoding='utf-8') as f:
        article_text = f.read()
    
    # Process with your QC sheet (now includes supplement/protocol checking)
    results = extractor.process_article_with_qc_sheet(
        article_text=article_text,
        qc_csv_path="/Users/emilylloyd/Documents/systematic_review_extraction/DataExtract_QC.csv",
        article_path=article
    )
    
    # Save results
    extractor.save_results(results, file_name)
    
    return results

if __name__ == "__main__":
    # Single article processing
    results = main()
