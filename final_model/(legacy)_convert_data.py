"""
Convert QandA.txt and courses_clean.json to evaluation format for Ass5-style metrics
"""
import json
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

print("Loading course data...")
with open("courses_clean.json", "r", encoding="utf-8") as f:
    courses = json.load(f)

# Create course documents with full context
course_docs = {}
for course in courses:
    # Combine all sections into full text
    sections_text = "\n\n".join([
        f"{section['section_name_original']}: {section['content']}"
        for section in course.get('sections', [])
    ])
    
    full_text = (
        f"{course['course_code']}\n"
        f"Course name: {course['course_name']}\n"
        f"Type: {course['course_type']}\n\n"
        f"{sections_text}"
    )
    course_docs[course['course_code']] = full_text

print(f"Loaded {len(course_docs)} courses")

# Setup text splitter for chunking (matching Ass5 approach)
print("\nChunking course documents...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

# Chunk each course document
course_chunks = {}
for course_code, full_text in course_docs.items():
    doc = Document(page_content=full_text, metadata={"course_code": course_code})
    chunks = text_splitter.split_documents([doc])
    # Store chunks but keep full text for backward compatibility
    course_chunks[course_code] = [chunk.page_content for chunk in chunks]

print(f"Created {sum(len(chunks) for chunks in course_chunks.values())} chunks from {len(course_docs)} courses")

# Parse QandA.txt
print("\nParsing QandA.txt...")
with open("QandA.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

qa_pairs = []
i = 0
while i < len(lines):
    line = lines[i].strip()
    if line.startswith("Q") and ":" in line:
        # Extract question
        question = line.split(":", 1)[1].strip()
        
        # Get answer from next line
        if i + 1 < len(lines):
            ans_line = lines[i + 1].strip()
            if ans_line.startswith("A") and ":" in ans_line:
                answer = ans_line.split(":", 1)[1].strip()
                
                # Extract course code from question
                match = re.search(r'\b([A-Z]{3}\d{3})\b', question)
                course_code = match.group(1) if match else None
                
                # Determine yes/no label (for applicable questions)
                answer_lower = answer.lower()
                gold_label = None
                if re.match(r'^yes\b', answer_lower):
                    gold_label = 'yes'
                elif re.match(r'^no\b', answer_lower):
                    gold_label = 'no'
                
                # Get context if course code exists
                context = course_docs.get(course_code, "")
                
                qa_pairs.append({
                    "question": question,
                    "answer": answer,
                    "gold_label": gold_label,
                    "course_code": course_code,
                    "context": context,
                    "has_context": len(context) > 0
                })
                i += 2
                continue
    i += 1

# Filter to only yes/no questions with valid context (like Ass5)
yes_no_pairs = [qa for qa in qa_pairs if qa['gold_label'] in ['yes', 'no'] and qa['has_context']]

print(f"\nTotal Q&A pairs: {len(qa_pairs)}")
print(f"Yes/No questions: {len([qa for qa in qa_pairs if qa['gold_label'] in ['yes', 'no']])}")
print(f"Yes/No with valid context: {len(yes_no_pairs)}")

# Save in evaluation format
with open("qa_eval_format.json", "w", encoding="utf-8") as f:
    json.dump(yes_no_pairs, f, indent=2, ensure_ascii=False)

print(f"\n✓ Saved {len(yes_no_pairs)} questions to qa_eval_format.json")

# Show statistics
yes_count = sum(1 for qa in yes_no_pairs if qa['gold_label'] == 'yes')
no_count = sum(1 for qa in yes_no_pairs if qa['gold_label'] == 'no')
print(f"\nLabel distribution:")
print(f"  Yes: {yes_count} ({100*yes_count/len(yes_no_pairs):.1f}%)")
print(f"  No:  {no_count} ({100*no_count/len(yes_no_pairs):.1f}%)")

print(f"\nSample questions:")
for i, qa in enumerate(yes_no_pairs[:5], 1):
    print(f"{i}. {qa['question'][:60]}... -> {qa['gold_label']}")
