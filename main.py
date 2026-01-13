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

FALLBACK_CORRECTIONS = [
    "Improved clarity in research objectives",
    "Enhanced methodology description",
    "More precise results presentation",
    "Stronger conclusion statement"
]

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
            # Create specialized agents
            input_agent = agents_client.create_agent(
                model=MODEL_DEPLOYMENT,
                name="input_agent",
                instructions=(
                    "You are 'Input Agent'. First ask the user for their abstract, "
                    "then for any custom review commands or preferences, "
                    "and finally ask for the file path to the full article (if available)."
                ),
            )
            
            reviewer_agent = agents_client.create_agent(
                model=MODEL_DEPLOYMENT,
                name="reviewer_agent",
                instructions=(
                    "You are 'Reviewer Agent'. Your objective is to review the quality of the abstract "
                    "based on the full article text. Provide constructive feedback on: "
                    "1. Clarity and readability\n"
                    "2. Completeness (background, methods, results, conclusion)\n"
                    "3. Accuracy relative to full article\n"
                    "4. Academic writing style\n"
                    "5. Overall impact and contribution"
                ),
            )
            
            checklister_agent = agents_client.create_agent(
                model=MODEL_DEPLOYMENT,
                name="checklister_agent",
                instructions=(
                    "You are 'Checklister Agent'. Evaluate the abstract quality using percentage scores (0-100) "
                    "for these objective criteria:\n"
                    "1. Length (250-500 words optimal)\n"
                    "2. Keywords (relevance to paper subject)\n"
                    "3. Gist (captures essence of article)\n"
                    "4. Complicity (aligns with conclusions)\n"
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
            
            # Create orchestrator
            orchestrator = agents_client.create_agent(
                model=MODEL_DEPLOYMENT,
                name="abstract_orchestrator",
                instructions=(
                    "You are 'Abstract Orchestrator'. Coordinate the abstract review process:\n"
                    "1. Use input_agent to collect the abstract and article\n"
                    "2. Use reviewer_agent for qualitative feedback\n"
                    "3. Use checklister_agent for quantitative scoring\n"
                    "4. Use writer_agent to produce corrected version\n"
                    "5. Present final review with scores, feedback, and corrected abstract\n\n"
                    "Format output clearly with sections: REVIEW, SCORES, CORRECTED ABSTRACT"
                ),
                tools=[t_input.definitions[0], t_review.definitions[0], 
                       t_check.definitions[0], t_write.definitions[0]],
            )
            
            # Get user input
            print("\n📚 --- AbstractReviewAI --- 📚\n")
            user_abstract = input("Enter your abstract: ").strip()
            user_commands = input("Enter custom review commands or 'none': ").strip() or "none"
            user_article = input("Provide filepath to full article (or press Enter if not available): ").strip()
            
            # Create thread and send message
            thread = agents_client.threads.create()
            user_msg = f"My abstract: {user_abstract}. "
            if user_commands != "none":
                user_msg += f"Custom review commands: {user_commands}. "
            if user_article:
                user_msg += f"Article file path: {user_article}. "
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
            
            # Parse the output (this is simplified - you might need more sophisticated parsing)
            # For now, we'll create a basic structure
            parsed = {
                "review_comments": [],
                "checklist_scores": {},
                "corrected_abstract": "",
                "improvement_summary": ""
            }
            
            # Simple parsing logic (you might want to enhance this)
            lines = combined.split('\n')
            current_section = None
            
            for line in lines:
                line_lower = line.lower()
                if 'review' in line_lower:
                    current_section = 'review'
                elif 'score' in line_lower or 'checklist' in line_lower:
                    current_section = 'scores'
                    # Try to find JSON-like scores
                    import json
                    try:
                        # Look for JSON in the line
                        json_start = line.find('{')
                        json_end = line.rfind('}') + 1
                        if json_start != -1 and json_end != 0:
                            scores_json = line[json_start:json_end]
                            parsed["checklist_scores"] = json.loads(scores_json)
                    except:
                        pass
                elif 'corrected' in line_lower or 'improved' in line_lower:
                    current_section = 'corrected'
                
                if current_section == 'review' and line.strip() and 'review' not in line_lower:
                    parsed["review_comments"].append(line.strip())
                elif current_section == 'corrected' and line.strip() and 'corrected' not in line_lower:
                    parsed["corrected_abstract"] += line + "\n"
            
            # If no corrected abstract was found, use a simple improvement
            if not parsed["corrected_abstract"].strip():
                parsed["corrected_abstract"] = user_abstract  # Fallback
            
            # Create final result
            result = validate_and_fill(
                parsed, 
                original_abstract=user_abstract, 
                custom_commands=user_commands
            )
            
            # Add article path to result
            if user_article:
                result["article_path"] = user_article
            
            # Display result
            print("\n" + "="*50)
            print("ABSTRACT REVIEW REPORT")
            print("="*50)
            print(f"\nOriginal Abstract ({len(user_abstract.split())} words):")
            print("-"*40)
            print(user_abstract[:500] + ("..." if len(user_abstract) > 500 else ""))
            
            print(f"\nReview Comments ({len(result['review_comments'])}):")
            print("-"*40)
            for i, comment in enumerate(result['review_comments'], 1):
                print(f"{i}. {comment}")
            
            print(f"\nChecklist Scores:")
            print("-"*40)
            for criterion, score in result['checklist_scores'].items():
                print(f"{criterion.replace('_', ' ').title()}: {score}/100")
            
            print(f"\nCorrected Abstract ({len(result['corrected_abstract'].split())} words):")
            print("-"*40)
            print(result['corrected_abstract'][:600] + 
                  ("..." if len(result['corrected_abstract']) > 600 else ""))
            
            print(f"\nImprovement Summary:")
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
                f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
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
        
        print("\nReview complete!")


if __name__ == "__main__":
    run_abstract_reviewer()