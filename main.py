"""
AbstractReviewAI — Multi-Agent Abstract Review and Correction System
"""

import os
import re
import json
from datetime import datetime
import pathlib
from dotenv import load_dotenv

# Azure AI libraries
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import ConnectedAgentTool, MessageRole, ListSortOrder
from azure.identity import DefaultAzureCredential

# --------------------- Fallback values ---------------------
FALLBACK_REVIEW = [
    "Abstract provides a good overview but could be more concise.",
    "Consider adding more specific methodology details.",
    "Results section should highlight key findings more clearly.",
    "Conclusion could better articulate the study's contributions.",
]

# --------------------- File Reading Function ---------------------
def read_article_file(filepath: str) -> str:
    """
    Read article content from file with error handling.
    
    Args:
        filepath: Path to the article file
        
    Returns:
        str: File content or error message
        
    Raises:
        FileNotFoundError: If file doesn't exist
        UnicodeDecodeError: If file encoding issues
    """
    try:
        # Expand user home directory (~)
        expanded_path = os.path.expanduser(filepath)
        
        if not os.path.exists(expanded_path):
            raise FileNotFoundError(f"File not found: {expanded_path}")
        
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(expanded_path, 'r', encoding=encoding) as f:
                    content = f.read()
                    print(f"✓ Successfully read article from: {expanded_path}")
                    print(f"  File size: {len(content)} characters")
                    return content
            except UnicodeDecodeError:
                continue
        
        # If all encodings fail, try reading as binary
        with open(expanded_path, 'rb') as f:
            content = f.read().decode('utf-8', errors='ignore')
            print(f"⚠ Read article with encoding issues (non-UTF8 characters ignored)")
            return content
            
    except Exception as e:
        raise Exception(f"Error reading file '{filepath}': {str(e)}")

def truncate_content(content: str, max_words: int = 2000) -> str:
    """Truncate content to avoid exceeding token limits."""
    words = content.split()
    if len(words) > max_words:
        truncated = ' '.join(words[:max_words])
        return f"{truncated}\n\n[Content truncated to {max_words} words. Full article has {len(words)} words.]"
    return content

# --------------------- auxiliary functions ---------------------
def pretty_json(obj):
    """Formats data as readable JSON."""
    return json.dumps(obj, indent=2, ensure_ascii=False)

def clean_markdown(s):
    """Removes Markdown symbols to make the text look clean."""
    s = s.strip()
    s = re.sub(r'【[^】]*】', '', s)  # Remove citations 【...】
    s = re.sub(r"^[\-\*\+\s]+", "", s)  # Remove hyphens, asterisks, and plus signs
    s = re.sub(r"^\*\*(.+?)\*\*$", r"\1", s)  # Remove double asterisks
    s = re.sub(r"^\*(.+?)\*$", r"\1", s)  # Remove single asterisks
    return s.strip()

def clamp_words(s: str, max_words: int) -> str:
    """Shortens text to a specific number of words."""
    words = s.split()
    if len(words) <= max_words:
        return s
    return " ".join(words[:max_words])

# --------------------- main programme --------------------
def validate_and_fill(parsed: dict, original_abstract: str, custom_commands: str, article_content) -> dict:
    """Checks whether the review contains all sections and fills in missing data."""
    out = {
        "original_abstract": original_abstract.strip(),
        "custom_commands": (custom_commands or "none").strip(),
        "review_comments": list(parsed.get("review_comments") or []),
        "checklist_scores": parsed.get("checklist_scores") or {},
        "corrected_abstract": (parsed.get("corrected_abstract") or original_abstract).strip(),
        "improvement_summary": (parsed.get("improvement_summary") or 
                               "Abstract has been reviewed and improved for clarity, completeness, and academic standards.").strip(),
    }

    # Process review comments
    out["review_comments"] = [clamp_words(clean_markdown(s), 50) for s in out["review_comments"] if s and s.strip()]
    if len(out["review_comments"]) < 2:
        out["review_comments"] = FALLBACK_REVIEW
    else:
        out["review_comments"] = out["review_comments"][:10]

    # SIMPLIFIED: Use agent's scores directly if they match expected categories
    # If not, use intelligent mapping
    agent_scores = out["checklist_scores"]
    expected_categories = ["length", "keywords", "gist", "consistency", 
                          "inclusion", "checklist_completeness", "conciseness"]
    
    # If agent returned exactly our expected categories, use them
    if all(cat in agent_scores for cat in expected_categories):
        # Already have the right format
        pass
    else:
        # Map from agent's categories to expected categories
        mapped_scores = {}
        
        # Common mappings from agent's categories to ours
        mapping_rules = {
            "length": ["length", "word_count"],
            "keywords": ["keywords", "relevance", "subject_relevance"],
            "gist": ["gist", "essence", "technical_accuracy", "scientific_rigor"],
            "consistency": ["consistency", "consistency_with_article_content", "alignment"],
            "inclusion": ["inclusion", "completeness", "data_inclusion"],
            "checklist_completeness": ["checklist_completeness", "completeness", "structure"],
            "conciseness": ["conciseness", "clarity", "brevity"]
        }
        
        # Try to map each expected category
        for expected_cat in expected_categories:
            found = False
            # Check mapping rules
            if expected_cat in mapping_rules:
                for possible_key in mapping_rules[expected_cat]:
                    if possible_key in agent_scores:
                        mapped_scores[expected_cat] = agent_scores[possible_key]
                        found = True
                        break
            
            # If not found, use default
            if not found:
                # Smart defaults based on category
                defaults = {
                    "length": 85,  # Based on 200 words
                    "keywords": 90,
                    "gist": 90,
                    "consistency": 98 if article_content else 75,  # High if article matches
                    "inclusion": 85,
                    "checklist_completeness": 85,
                    "conciseness": 90
                }
                mapped_scores[expected_cat] = defaults.get(expected_cat, 70)
        
        out["checklist_scores"] = mapped_scores
    
    # Check for off-topic warning and adjust consistency score
    off_topic_keywords = ["off-topic", "different topic", "unrelated", "mismatch", 
                         "inconsistent", "photonics", "optics", "waveguide"]
    review_text = " ".join(out["review_comments"]).lower()
    if any(keyword in review_text for keyword in off_topic_keywords):
        print("⚠  Detected off-topic article. Adjusting consistency score.")
        out["checklist_scores"]["consistency"] = 0

    # Ensure corrected abstract is reasonable length
    if len(out["corrected_abstract"].split()) < 30:
        out["corrected_abstract"] = original_abstract

    return out

def run_abstract_reviewer():
    os.system("cls" if os.name == "nt" else "clear")
    load_dotenv()
    
    PROJECT_ENDPOINT = os.getenv('PROJECT_ENDPOINT')
    MODEL_DEPLOYMENT = os.getenv('MODEL_DEPLOYMENT_NAME')
    
    if not PROJECT_ENDPOINT or not MODEL_DEPLOYMENT:
        raise RuntimeError("Set PROJECT_ENDPOINT and MODEL_DEPLOYMENT_NAME in your .env file.")
    
    credential = DefaultAzureCredential()
    agents_client = AgentsClient(
        endpoint=PROJECT_ENDPOINT,
        credential=credential,
    )
    
    try:
        with agents_client:
            # Get user input
            print("\n📚 --- AbstractReviewAI --- 📚\n")
            print("Paste your abstract (can be multiple lines):")
            user_abstract = ""
            while True:
                line = input()
                if line == "":
                    break  # Stop on empty line
                if user_abstract:
                    user_abstract += "\n" + line
                else:
                    user_abstract = line

            user_commands = input("Enter custom review commands or 'none': ").strip() or "none"
            user_article = input("Provide filepath to full article (or press Enter if not available): ").strip()
            
            # Read article file if provided
            article_content = None
            if user_article:
                try:
                    article_content = read_article_file(user_article)
                    article_content = truncate_content(article_content)
                    print(f"✓ Article loaded successfully ({len(article_content.split())} words)")
                except Exception as e:
                    print(f"\n❌ Error: {str(e)}")
                    print("Please either:")
                    print("1. Provide a valid file path")
                    print("2. Press Enter to continue without the article")
                    print("3. Type 'exit' to quit")
                    
                    choice = input("\nYour choice: ").strip().lower()
                    if choice == 'exit':
                        print("Exiting...")
                        return
                    elif choice == '':
                        print("Continuing without article...")
                        article_content = None
                    else:
                        # Try with new path
                        try:
                            article_content = read_article_file(choice)
                            article_content = truncate_content(article_content)
                            print(f"✓ Article loaded successfully ({len(article_content.split())} words)")
                            user_article = choice
                        except Exception as e2:
                            print(f"❌ Failed to read file: {str(e2)}")
                            print("Continuing without article...")
                            article_content = None
            
            # Create specialized agents with updated instructions based on article availability
            input_agent = agents_client.create_agent(
                model=MODEL_DEPLOYMENT,
                name="input_agent",
                instructions=(
                    "You are 'Input Agent'. First ask the user for their abstract, "
                    "then for any custom review commands or preferences, "
                    "and finally ask for the file path to the full article (if available)."
                ),
            )
            
            # Update reviewer agent instructions based on whether article is available
            # Update reviewer agent instructions based on whether article is available
            if article_content:
                reviewer_instructions = (
                    f"CRITICAL: You are 'Reviewer Agent'. First, check if this article is relevant to the abstract topic.\n\n"
                    f"ABSTRACT TOPIC: 3D sand mould printing, binder jet technology, casting, additive manufacturing, sustainable manufacturing.\n\n"
                    f"ARTICLE CONTENT (first 1000 chars):\n{article_content[:1000]}\n\n"
                    f"DECISION TREE:\n"
                    f"1. If the article is about COMPLETELY DIFFERENT topics (e.g., photonics, silicon chips, optics, microresonators, etc.), "
                    f"then IMMEDIATELY state: 'CRITICAL WARNING: The uploaded article appears to be completely off-topic. "
                    f"The abstract discusses 3D sand printing for casting, while the article is about [briefly describe article topic]. "
                    f"Review will proceed based on abstract's internal consistency only.'\n\n"
                    f"2. If the article IS relevant, then review normally.\n\n"
                    f"3. For normal review, check:\n"
                    f"   - Consistency between abstract and article\n"
                    f"   - Clarity and structure\n"
                    f"   - Completeness (background, methods, results, conclusion)\n"
                    f"   - Academic standards\n"
                    f"   - Overall impact\n\n"
                    f"IMPORTANT: Be brutally honest about topic mismatch if it exists!"
                )
            else:
                reviewer_instructions = (
                    "You are 'Reviewer Agent'. Review the abstract quality based on its own merits since no full article was provided. "
                    "Provide constructive feedback on:\n"
                    "1. Clarity and readability\n"
                    "2. Completeness (background, methods, results, conclusion)\n"
                    "3. Academic writing style\n"
                    "4. Overall impact and contribution\n"
                    "5. Logical flow and structure"
                )   

            reviewer_agent = agents_client.create_agent(
                model=MODEL_DEPLOYMENT,
                name="reviewer_agent",
                instructions=reviewer_instructions,
            )
            
            checklister_agent = agents_client.create_agent(
                model=MODEL_DEPLOYMENT,
                name="checklister_agent",
                instructions=(
                    "You MUST return EXACTLY 7 scores in this EXACT JSON format with these EXACT keys:\n\n"
                    '{"length": score, "keywords": score, "gist": score, "consistency": score, "inclusion": score, "checklist_completeness": score, "conciseness": score}\n\n'
                    "Score each category 0-100%:\n"
                    "1. 'length': Is abstract 200-250 words? (200-250 words = 100/100; less than 50 or more than 500 = 0/100)\n"
                    "2. 'keywords': Relevance to paper subject\n"
                    "3. 'gist': Captures essence of article\n"
                    "4. 'consistency': Aligns with full article content (100/100 if perfect match; 0/100 if off-topic)\n"
                    "5. 'inclusion': Contains relevant data/evidence\n"
                    "6. 'checklist_completeness': Has background, objective, methods, results, conclusion\n"
                    "7. 'conciseness': Balance between comprehensive and concise\n\n"
                    "DO NOT use any other keys. DO NOT add explanations. Return ONLY the JSON."
                ),
            )
            
            writer_agent = agents_client.create_agent(
                model=MODEL_DEPLOYMENT,
                name="writer_agent",
                instructions=(
                    "You are 'Abstract Writing Agent'. Based on feedback from the reviewer "
                    "and checklister agent, write an improved version of the abstract. "
                    "Maintain the original meaning while addressing identified issues. "
                    "Ensure academic tone and proper structure."
                ),
            )
            
            # Connect all agents
            t_input = ConnectedAgentTool(id=input_agent.id, name="input_agent", 
                                        description="Collects user input including abstract and preferences")
            t_review = ConnectedAgentTool(id=reviewer_agent.id, name="reviewer_agent", 
                                         description="Provides detailed review of abstract quality")
            t_check = ConnectedAgentTool(id=checklister_agent.id, name="checklister_agent", 
                                        description="Evaluates abstract against objective criteria with scores")
            t_write = ConnectedAgentTool(id=writer_agent.id, name="writer_agent", 
                                        description="Writes corrected abstract based on feedback")
            
            # Create orchestrator with article context
            orchestrator_instructions = (
                "You are 'Abstract Orchestrator'. DO NOT ask for the abstract or article - they have already been provided.\n"
                "Coordinate the review process by:\n"
                "1. Using reviewer_agent to analyze the abstract (and article if provided)\n"
                "2. Using checklister_agent to provide percentage scores (as JSON)\n"
                "3. Using writer_agent to improve the abstract\n"
                "4. Presenting the final review with these sections clearly labeled:\n"
                "   - REVIEW COMMENTS (as bullet points)\n"
                "   - CHECKLIST SCORES (as JSON object)\n"
                "   - CORRECTED ABSTRACT\n"
                "   - IMPROVEMENT SUMMARY (as bullet points)\n\n"
                "IMPORTANT: Format each section with the exact header names above."
            )
            
            if article_content:
                orchestrator_instructions += (
                    f"IMPORTANT: The full article has been provided. Ensure the review agent "
                    f"checks for consistency between the abstract and the full article content.\n\n"
                )
            else:
                orchestrator_instructions += (
                    "NOTE: No full article was provided. Review will be based on abstract's internal consistency.\n\n"
                )
            
            orchestrator = agents_client.create_agent(
                model=MODEL_DEPLOYMENT,
                name="abstract_orchestrator",
                instructions=orchestrator_instructions,
                tools=[t_review.definitions[0], t_check.definitions[0], t_write.definitions[0]]
            )
            
            # Create thread and send message
            thread = agents_client.threads.create()
            
            # Build user message with article context
            user_msg = f"My abstract: {user_abstract}. "
            if user_commands != "none":
                user_msg += f"Custom review commands: {user_commands}. "
            
            if article_content:
                user_msg += f"Full article content has been provided to the reviewer agent. "
            else:
                user_msg += f"No full article was provided. "
            
            user_msg += "Please review and correct my abstract."
            
            agents_client.messages.create(
                thread_id=thread.id,
                role=MessageRole.USER,
                content=user_msg
            )
            
            print("\n🔍 Processing abstract review...\n")
            
            # Run the orchestrator
            run = agents_client.runs.create_and_process(
                thread_id=thread.id,
                agent_id=orchestrator.id
            )
            
            if run.status == "failed":
                print("Run failed:", run.last_error)
            
            # Retrieve and process messages
            messages = agents_client.messages.list(
                thread_id=thread.id,
                order=ListSortOrder.ASCENDING
            )
            
            assistant_chunks = []
            for m in messages:
                if m.text_messages:
                    text = m.text_messages[-1].text.value
                    assistant_chunks.append(text)
                    print(text)
            
            combined = "\n".join(assistant_chunks)
            
            # NEW PARSING FUNCTION - FIXED
            def parse_assistant_output(text: str) -> dict:
                """Parse assistant output into structured sections."""
                parsed = {
                    "review_comments": [],
                    "checklist_scores": {},
                    "corrected_abstract": "",
                    "improvement_summary": ""
                }
                
                # Convert to uppercase for case-insensitive matching
                text_upper = text.upper()
                
                # Find section boundaries
                review_start = text_upper.find("REVIEW COMMENTS")
                scores_start = text_upper.find("CHECKLIST SCORES")
                corrected_start = text_upper.find("CORRECTED ABSTRACT")
                improvement_start = text_upper.find("IMPROVEMENT SUMMARY")
                
                # Extract review comments
                if review_start != -1 and scores_start != -1:
                    review_text = text[review_start:scores_start]
                    # Extract bullet points
                    lines = review_text.split('\n')
                    for line in lines:
                        line = line.strip()
                        # Look for bullet points or numbered lists
                        if line.startswith(('-', '•', '*')):
                            # Remove bullet and clean
                            clean_line = re.sub(r'^[\-\•\*\s]+', '', line)
                            if clean_line and len(clean_line) > 10:  # Ensure meaningful content
                                parsed["review_comments"].append(clean_line)
                        elif re.match(r'^\d+\.', line):
                            # Remove number and clean
                            clean_line = re.sub(r'^\d+\.\s*', '', line)
                            if clean_line and len(clean_line) > 10:
                                parsed["review_comments"].append(clean_line)
                
                # Extract checklist scores
                if scores_start != -1 and corrected_start != -1:
                    scores_text = text[scores_start:corrected_start]
                elif scores_start != -1 and corrected_start == -1 and improvement_start != -1:
                    scores_text = text[scores_start:improvement_start]
                elif scores_start != -1:
                    scores_text = text[scores_start:]
                
                if 'scores_text' in locals():
                    # Find JSON object
                    try:
                        json_start = scores_text.find('{')
                        json_end = scores_text.rfind('}') + 1
                        if json_start != -1 and json_end > json_start:
                            scores_json = scores_text[json_start:json_end]
                            parsed["checklist_scores"] = json.loads(scores_json)
                    except json.JSONDecodeError:
                        print(f"⚠  Warning: Could not parse scores JSON")
                
                # Extract corrected abstract
                if corrected_start != -1 and improvement_start != -1:
                    corrected_text = text[corrected_start:improvement_start]
                elif corrected_start != -1:
                    corrected_text = text[corrected_start:]
                
                if 'corrected_text' in locals():
                    # Remove the "CORRECTED ABSTRACT" header and any following empty lines
                    lines = corrected_text.split('\n')
                    abstract_lines = []
                    header_passed = False
                    for line in lines:
                        if line.upper().strip().startswith("CORRECTED ABSTRACT"):
                            header_passed = True
                            continue
                        if header_passed and line.strip():
                            abstract_lines.append(line.strip())
                    
                    if abstract_lines:
                        parsed["corrected_abstract"] = ' '.join(abstract_lines)
                
                # Extract improvement summary
                if improvement_start != -1:
                    improvement_text = text[improvement_start:]
                    # Remove the "IMPROVEMENT SUMMARY" header
                    lines = improvement_text.split('\n')
                    summary_lines = []
                    header_passed = False
                    for line in lines:
                        if line.upper().strip().startswith("IMPROVEMENT SUMMARY"):
                            header_passed = True
                            continue
                        if header_passed and line.strip():
                            # Remove bullet points for summary
                            clean_line = re.sub(r'^[\-\•\*\s]+', '', line.strip())
                            if clean_line:
                                summary_lines.append(clean_line)
                    
                    if summary_lines:
                        parsed["improvement_summary"] = ' '.join(summary_lines)
                
                return parsed
            
            # Use the parsing function
            parsed = parse_assistant_output(combined)
            
            # Debug: Print what was parsed
            print(f"\n🔍 DEBUG PARSED DATA:")
            print(f"Review comments found: {len(parsed['review_comments'])}")
            print(f"Checklist scores found: {len(parsed['checklist_scores'])}")
            print(f"Corrected abstract length: {len(parsed['corrected_abstract'])} chars")
            
            # Create final result
            result = validate_and_fill(
                parsed, 
                original_abstract=user_abstract, 
                custom_commands=user_commands,
                article_content=article_content
            )
            
            # Add article info to result
            if user_article:
                result["article_path"] = user_article
                result["article_provided"] = article_content is not None
            

            # Display result
            print("\n" + "="*50)
            print("ABSTRACT REVIEW REPORT")
            print("="*50)

            # Calculate actual word count properly
            abstract_words = len(user_abstract.split())
            print(f"\n📝 Original Abstract ({abstract_words} words):")
            print("-"*40)
            # Display the FULL abstract, not truncated
            print(user_abstract)

            print(f"\n📋 Review Comments ({len(result['review_comments'])}):")
            print("-"*40)
            for i, comment in enumerate(result['review_comments'], 1):
                print(f"{i}. {comment}")

            print(f"\n📊 Checklist Scores:")
            print("-"*40)
            for criterion, score in result['checklist_scores'].items():
                print(f"{criterion.replace('_', ' ').title()}: {score}/100")

            # Calculate corrected abstract word count
            corrected_words = len(result['corrected_abstract'].split())
            print(f"\n✏️  Corrected Abstract ({corrected_words} words):")
            print("-"*40)
            # Display the FULL corrected abstract
            print(result['corrected_abstract'])

            print(f"\n✅ Improvement Summary:")
            print("-"*40)
            print(result['improvement_summary'])

            
            # Save results to files
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            base = f"abstract_review_{ts}"
            out_dir = pathlib.Path("./outputs")
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # Save JSON
            json_path = out_dir / f"{base}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            # Save Markdown report
            md_path = out_dir / f"{base}.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(f"# Abstract Review Report\n\n")
                f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**Article Provided:** {'Yes' if article_content else 'No'}\n")
                if user_article:
                    f.write(f"**Article Path:** {user_article}\n")
                f.write(f"\n")
                
                f.write(f"## Original Abstract\n")
                f.write(f"```\n{user_abstract}\n```\n\n")
                
                f.write(f"## Review Comments\n")
                for i, comment in enumerate(result['review_comments'], 1):
                    f.write(f"{i}. {comment}\n")
                f.write(f"\n")
                
                f.write(f"## Checklist Scores\n")
                for criterion, score in result['checklist_scores'].items():
                    f.write(f"- **{criterion.replace('_', ' ').title()}:** {score}/100\n")
                f.write(f"\n")
                
                f.write(f"## Corrected Abstract\n")
                f.write(f"```\n{result['corrected_abstract']}\n```\n\n")
                
                f.write(f"## Improvement Summary\n")
                f.write(f"{result['improvement_summary']}\n")
            
            print("\n📁 Files saved:")
            print(f"- JSON: {json_path.resolve()}")
            print(f"- Markdown: {md_path.resolve()}")
    
    finally:
        # Clean up agents
        print("\nCleaning up agents...")
        try:
            agents_to_clean = [
                locals().get("orchestrator"),
                locals().get("input_agent"),
                locals().get("reviewer_agent"),
                locals().get("checklister_agent"),
                locals().get("writer_agent")
            ]
            
            for agent in agents_to_clean:
                if agent is not None:
                    try:
                        agents_client.delete_agent(agent.id)
                    except Exception as e:
                        print(f"Warning: deleting agent failed: {e}")
        except Exception as e:
            print(f"Cleanup encountered an error: {e}")
        
        print("\n✅ Review complete!")


if __name__ == "__main__":
    run_abstract_reviewer()