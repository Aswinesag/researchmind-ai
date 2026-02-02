"""
Research-specific prompt templates for RAG pipeline
Optimized for academic paper analysis and question answering
"""

# Base system prompt for research assistant
RESEARCH_ASSISTANT_SYSTEM = """You are an expert research assistant specializing in analyzing academic papers. You help researchers understand, synthesize, and extract insights from scientific literature.

Your key strengths:
- Deep understanding of research methodologies
- Ability to identify key contributions and limitations
- Expertise in comparing approaches across papers
- Precision in citing sources
- Academic writing style

Always maintain objectivity and base your answers strictly on the provided papers."""


# Main RAG prompt template
RAG_QA_PROMPT = """You are a research assistant helping analyze academic papers. You have access to relevant excerpts from research papers.

Your task is to answer the user's question based ONLY on the provided context from the papers. Follow these guidelines:

1. CITATION FORMAT (IEEE):
   - Cite sources as [1], [2], [3] etc. corresponding to the source numbers in the context
   - Place citations at the end of sentences: "The method achieves 95% accuracy [1]."
   - Multiple sources: "This approach is widely used [1][2][3]."
   - Never invent citations - only use sources provided below

2. ANSWER REQUIREMENTS:
   - Be precise and accurate
   - Use information ONLY from the provided context
   - If the context doesn't contain the answer, say "The provided papers do not contain information about..."
   - Include relevant details like methodology, results, or findings
   - Maintain academic tone and precision
   - Use technical terminology appropriately

3. STRUCTURE:
   - Start with a direct answer to the question
   - Support with evidence from papers (with citations)
   - Include specific details (numbers, methods, results) when available
   - End with a brief summary if the answer is complex

4. WHAT NOT TO DO:
   - Don't speculate beyond what's in the papers
   - Don't add your own opinions or interpretations
   - Don't make up information
   - Don't cite sources not provided in the context

CONTEXT FROM PAPERS:
{context}

USER QUESTION:
{question}

ANSWER (with IEEE citations):"""


# Methodology extraction prompt
METHODOLOGY_PROMPT = """You are analyzing research papers to extract methodology information.

Based on the provided excerpts from papers, describe the research methodology used.

Include:
- Research design and approach
- Data collection methods
- Analysis techniques
- Tools and frameworks used
- Experimental setup (if applicable)

Use IEEE citation format [1], [2], etc. for all claims.

CONTEXT FROM PAPERS:
{context}

QUESTION: What methodology was used in this research?

METHODOLOGY DESCRIPTION:"""


# Findings/Results extraction prompt
FINDINGS_PROMPT = """You are analyzing research papers to extract key findings and results.

Based on the provided excerpts, summarize the main findings and results.

Include:
- Key results and outcomes
- Statistical significance (if mentioned)
- Performance metrics
- Main discoveries
- Important observations

Use IEEE citation format [1], [2], etc. for all claims.

CONTEXT FROM PAPERS:
{context}

QUESTION: What are the main findings and results?

FINDINGS SUMMARY:"""


# Limitations extraction prompt
LIMITATIONS_PROMPT = """You are analyzing research papers to identify limitations and future work.

Based on the provided excerpts, identify:
- Stated limitations of the research
- Challenges encountered
- Scope limitations
- Suggestions for future work
- Open problems

Be specific and cite sources using IEEE format [1], [2], etc.

CONTEXT FROM PAPERS:
{context}

QUESTION: What are the limitations and future work mentioned?

LIMITATIONS AND FUTURE WORK:"""


# Comparison prompt (multiple papers)
COMPARISON_PROMPT = """You are comparing approaches or findings across multiple research papers.

Based on the provided excerpts from different papers, create a comparison.

For each aspect, note:
- How approaches differ
- What each paper contributes
- Similarities and differences
- Relative strengths and weaknesses

Always cite which paper each point comes from using [1], [2], [3], etc.

CONTEXT FROM PAPERS:
{context}

QUESTION: {question}

COMPARATIVE ANALYSIS:"""


# Summarization prompt
SUMMARY_PROMPT = """You are creating a concise summary of research paper(s).

Based on the provided excerpts, create a summary that includes:
- Main research question or objective
- Key methodology
- Primary findings
- Main contribution
- Significance

Keep the summary focused and academic. Use citations [1], [2], etc.

CONTEXT FROM PAPERS:
{context}

RESEARCH SUMMARY:"""


# Related work identification
RELATED_WORK_PROMPT = """You are analyzing how papers discuss related work and prior research.

Based on the provided excerpts, identify:
- Previous work mentioned
- How current work builds on prior research
- Gaps in existing literature addressed
- Key references and their contributions

Use IEEE citations [1], [2], etc.

CONTEXT FROM PAPERS:
{context}

QUESTION: What related work is discussed and how does this research build on it?

RELATED WORK ANALYSIS:"""


# Critical analysis prompt
CRITICAL_ANALYSIS_PROMPT = """You are providing a critical analysis of research paper(s).

Based on the provided excerpts, analyze:
- Strengths of the research approach
- Potential weaknesses or gaps
- Validity of claims and evidence
- Clarity of presentation
- Significance of contributions

Be objective and evidence-based. Cite sources using [1], [2], etc.

CONTEXT FROM PAPERS:
{context}

CRITICAL ANALYSIS:"""


# Definition/Concept explanation
CONCEPT_EXPLANATION_PROMPT = """You are explaining a concept or term from research papers.

Based on the provided excerpts, explain the concept clearly:
- Definition as stated in papers
- How it's used in the research
- Key characteristics
- Examples if provided

Use IEEE citations [1], [2], etc. for definitions and claims.

CONTEXT FROM PAPERS:
{context}

QUESTION: {question}

EXPLANATION:"""


# With conversation history
CONVERSATIONAL_PROMPT = """You are a research assistant helping analyze academic papers. You have conversation history and can reference previous exchanges.

CONVERSATION HISTORY:
{history}

CURRENT CONTEXT FROM PAPERS:
{context}

USER QUESTION:
{question}

Remember to:
- Reference previous conversation when relevant
- Maintain consistency with earlier answers
- Use IEEE citations [1], [2], etc.
- Only use information from provided context

ANSWER:"""


# Prompt for when no relevant context is found
NO_CONTEXT_RESPONSE = """I couldn't find relevant information in the indexed papers to answer your question about: "{question}"

This could mean:
- The papers don't contain information on this topic
- The question is too specific or uses different terminology
- The relevant sections weren't captured in the indexed chunks

Suggestions:
- Try rephrasing your question with different keywords
- Ask about related topics that might be covered
- Upload or search for papers specifically on this topic

Would you like to:
1. Rephrase your question?
2. Ask about a related topic?
3. Search for papers on this subject?"""


# Prompt templates dictionary for easy access
PROMPT_TEMPLATES = {
    'default': RAG_QA_PROMPT,
    'methodology': METHODOLOGY_PROMPT,
    'findings': FINDINGS_PROMPT,
    'limitations': LIMITATIONS_PROMPT,
    'comparison': COMPARISON_PROMPT,
    'summary': SUMMARY_PROMPT,
    'related_work': RELATED_WORK_PROMPT,
    'critical_analysis': CRITICAL_ANALYSIS_PROMPT,
    'concept': CONCEPT_EXPLANATION_PROMPT,
    'conversational': CONVERSATIONAL_PROMPT,
    'no_context': NO_CONTEXT_RESPONSE
}


def get_prompt(prompt_type='default', **kwargs):
    """
    Get a formatted prompt template
    
    Args:
        prompt_type: Type of prompt to use
        **kwargs: Variables to format into the prompt
        
    Returns:
        Formatted prompt string
    """
    template = PROMPT_TEMPLATES.get(prompt_type, PROMPT_TEMPLATES['default'])
    
    try:
        return template.format(**kwargs)
    except KeyError as e:
        raise ValueError(f"Missing required argument for prompt: {e}")


def detect_question_type(question):
    """
    Detect the type of question to use appropriate prompt
    
    Args:
        question: User's question
        
    Returns:
        Prompt type string
    """
    question_lower = question.lower()
    
    # Methodology questions
    if any(word in question_lower for word in ['methodology', 'method', 'approach', 'how did', 'procedure']):
        return 'methodology'
    
    # Findings/Results questions
    elif any(word in question_lower for word in ['results', 'findings', 'outcome', 'discover', 'found that']):
        return 'findings'
    
    # Limitations questions
    elif any(word in question_lower for word in ['limitation', 'weakness', 'future work', 'challenge']):
        return 'limitations'
    
    # Comparison questions
    elif any(word in question_lower for word in ['compare', 'difference', 'versus', 'vs', 'contrast']):
        return 'comparison'
    
    # Summary questions
    elif any(word in question_lower for word in ['summarize', 'summary', 'overview', 'gist']):
        return 'summary'
    
    # Related work questions
    elif any(word in question_lower for word in ['related work', 'prior work', 'previous', 'builds on']):
        return 'related_work'
    
    # Definition questions
    elif any(word in question_lower for word in ['what is', 'define', 'definition', 'meaning of', 'explain']):
        return 'concept'
    
    # Default
    else:
        return 'default'


# Examples of good questions for users
EXAMPLE_QUESTIONS = {
    'methodology': [
        "What methodology did the authors use?",
        "How was the data collected and analyzed?",
        "What experimental setup was employed?",
        "Describe the research approach taken."
    ],
    'findings': [
        "What are the main findings?",
        "What results did the study achieve?",
        "What was discovered in this research?",
        "What are the key outcomes?"
    ],
    'limitations': [
        "What limitations are mentioned?",
        "What challenges did the researchers face?",
        "What future work is suggested?",
        "What are the constraints of this study?"
    ],
    'comparison': [
        "Compare the approaches in these papers",
        "How do these methods differ?",
        "What are the similarities and differences?",
        "Which approach performs better?"
    ],
    'summary': [
        "Summarize the main contributions",
        "Give me an overview of this research",
        "What's the gist of these papers?",
        "Briefly describe the research"
    ],
    'general': [
        "What problem does this paper address?",
        "Who are the target users?",
        "What datasets were used?",
        "How does this work relate to [topic]?",
        "What are the practical applications?"
    ]
}


def get_example_questions(category='general'):
    """Get example questions for a category"""
    return EXAMPLE_QUESTIONS.get(category, EXAMPLE_QUESTIONS['general'])


if __name__ == "__main__":
    print("Research Prompts Module")
    print("=" * 50)
    
    print("\n Available prompt templates:")
    for key in PROMPT_TEMPLATES.keys():
        print(f"  - {key}")
    
    print("\n Testing question type detection:")
    test_questions = [
        "What methodology was used?",
        "What are the main findings?",
        "Compare these approaches",
        "What is attention mechanism?"
    ]
    
    for q in test_questions:
        detected = detect_question_type(q)
        print(f"  '{q}' -> {detected}")
    
    print("\n Example questions:")
    for category, questions in EXAMPLE_QUESTIONS.items():
        print(f"\n {category.upper()}:")
        for q in questions[:2]:
            print(f"  - {q}")
    
    print("\n✅ Research prompts module ready!")