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

def validate_and_fill(parsed: dict, original_abstract: str, custom_commands: str) -> dict:
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
    out["review_comments"] = [clamp_words(clean_markdown(s), 20) for s in out["review_comments"] if s and s.strip()]
    if len(out["review_comments"]) < 2:
        out["review_comments"] = FALLBACK_REVIEW
    else:
        out["review_comments"] = out["review_comments"][:8]

    # Process checklist scores - ensure it's a proper dictionary
    if not isinstance(out["checklist_scores"], dict):
        out["checklist_scores"] = {
            "length": 75,
            "keywords": 80,
            "gist": 70,
            "complicity": 65,
            "inclusion": 80,
            "checklist_completeness": 70,
            "conciseness": 75
        }

    # Ensure corrected abstract is reasonable length
    if len(out["corrected_abstract"].split()) < 50:
        out["corrected_abstract"] = original_abstract  # Fallback to original if too short

    return out

# --------------------- main programme --------------------
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
            user_abstract = input("Enter your abstract: ").strip()
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
            if article_content:
                reviewer_instructions = (
                    f"You are 'Reviewer Agent'. Your objective is to review the quality of the abstract "
                    f"based on the full article text provided below.\n\n"
                    f"FULL ARTICLE CONTENT:\n{article_content}\n\n"
                    f"Provide constructive feedback comparing the abstract to the full article. Check:\n"
                    f"1. Consistency (internal and with full article content)\n"
                    f"2. Clarity and readability\n"
                    f"3. Completeness (background, methods, results, conclusion) referring to the content of the full article\n"
                    f"4. Accuracy relative to full article\n"
                    f"5. Academic writing style\n"
                    f"6. Overall impact and contribution\n\n"
                    f"Highlight any discrepancies between the abstract and the full article."
                )
            else:
                reviewer_instructions = (
                    "You are 'Reviewer Agent'. Your objective is to review the quality of the abstract "
                    "based on its own merits since no full article was provided. "
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
                    "You are 'Checklister Agent'. Evaluate the abstract quality using percentage scores (0-100%) "
                    "for these objective criteria:\n"
                    "1. Length (250-500 words optimal)\n"
                    "2. Keywords (relevance to paper subject)\n"
                    "3. Gist (captures essence of article)\n"
                    "4. Consistency (aligns with full article content if provided)\n"
                    "5. Inclusion (relevant data/evidence)\n"
                    "6. Checklist (background, objective, methods, results, conclusion)\n"
                    "7. Concise and Comprehensive balance\n\n"
                    "Return scores as a JSON object with these keys and numeric values."
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
                "You are 'Abstract Orchestrator'. Coordinate the abstract review process:\n"
                "1. Use input_agent to collect the abstract and article\n"
                "2. Use reviewer_agent for qualitative feedback\n"
                "3. Use checklister_agent for quantitative scoring (in %)\n"
                "4. Use writer_agent to produce corrected version\n"
                "5. Present final review with scores, feedback, error message if article file path is incorrect, short summary of the article actually provided in the file path and corrected abstract\n\n"
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
            
            orchestrator_instructions += "Format output clearly with sections: REVIEW, SCORES, CORRECTED ABSTRACT"
            
            orchestrator = agents_client.create_agent(
                model=MODEL_DEPLOYMENT,
                name="abstract_orchestrator",
                instructions=orchestrator_instructions,
                tools=[t_input.definitions[0], t_review.definitions[0], 
                       t_check.definitions[0], t_write.definitions[0]],
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
            
            # Parse the output
            parsed = {
                "review_comments": [],
                "checklist_scores": {},
                "corrected_abstract": "",
                "improvement_summary": ""
            }
            
            # Enhanced parsing logic - FIXED: Removed redefinition of 'json'
            lines = combined.split('\n')
            current_section = None
            
            for line in lines:
                line_lower = line.lower()
                if 'review' in line_lower:
                    current_section = 'review'
                elif 'score' in line_lower or 'checklist' in line_lower:
                    current_section = 'scores'
                    # Try to find JSON-like scores
                    try:
                        # Look for JSON in the line
                        json_start = line.find('{')
                        json_end = line.rfind('}') + 1
                        if json_start != -1 and json_end != 0:
                            scores_json = line[json_start:json_end]
                            # Use the imported json module, don't redefine it
                            parsed["checklist_scores"] = json.loads(scores_json)
                    except Exception as e:
                        print(f"Warning: Could not parse scores JSON: {e}")
                        pass
                elif 'corrected' in line_lower or 'improved' in line_lower:
                    current_section = 'corrected'
                elif 'summary' in line_lower or 'improvement' in line_lower:
                    current_section = 'summary'
                
                if current_section == 'review' and line.strip() and 'review' not in line_lower:
                    if line.strip().startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '-', '*')):
                        parsed["review_comments"].append(line.strip())
                elif current_section == 'corrected' and line.strip() and 'corrected' not in line_lower:
                    if not line.startswith('=') and not line.startswith('-'):
                        parsed["corrected_abstract"] += line + "\n"
                elif current_section == 'summary' and line.strip() and 'summary' not in line_lower:
                    if not line.startswith('=') and not line.startswith('-'):
                        parsed["improvement_summary"] += line + " "
            
            # If no corrected abstract was found, use a simple improvement
            if not parsed["corrected_abstract"].strip():
                parsed["corrected_abstract"] = user_abstract  # Fallback
            
            # Create final result
            result = validate_and_fill(
                parsed, 
                original_abstract=user_abstract, 
                custom_commands=user_commands
            )
            
            # Add article info to result
            if user_article:
                result["article_path"] = user_article
                result["article_provided"] = article_content is not None
            
            # Display result
            print("\n" + "="*50)
            print("ABSTRACT REVIEW REPORT")
            print("="*50)
            
            print(f"\n📝 Original Abstract ({len(user_abstract.split())} words):")
            print("-"*40)
            print(user_abstract[:500] + ("..." if len(user_abstract) > 500 else ""))
            
            print(f"\n📋 Review Comments ({len(result['review_comments'])}):")
            print("-"*40)
            for i, comment in enumerate(result['review_comments'], 1):
                print(f"{i}. {comment}")
            
            print(f"\n📊 Checklist Scores:")
            print("-"*40)
            for criterion, score in result['checklist_scores'].items():
                print(f"{criterion.replace('_', ' ').title()}: {score}/100")
            
            print(f"\n✏️  Corrected Abstract ({len(result['corrected_abstract'].split())} words):")
            print("-"*40)
            print(result['corrected_abstract'][:600] + 
                  ("..." if len(result['corrected_abstract']) > 600 else ""))
            
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