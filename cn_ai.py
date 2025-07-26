import streamlit as st
import os
from datetime import datetime
import pymupdf as fitz
import docx
import logging
from dotenv import load_dotenv
from transformers import pipeline
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st
from typing import List, Dict, Tuple, Any

st.set_page_config(
    page_title="Career Navigator AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
    }
    
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
    
    .info-box {
        background: #e7f3ff;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
        color: #000000;
    }
    
    .question-box {
        background: #fff3cd;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #ffeaa7;
        margin: 1rem 0;
        color: #000000;
    }
    
    .evaluation-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
        color: #000000;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    .upload-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #667eea;
        text-align: center;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'backend' not in st.session_state:
    st.session_state.backend = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'interview_questions' not in st.session_state:
    st.session_state.interview_questions = []
if 'interview_answers' not in st.session_state:
    st.session_state.interview_answers = []
if 'interview_evaluations' not in st.session_state:
    st.session_state.interview_evaluations = []
if 'current_question_index' not in st.session_state:
    st.session_state.current_question_index = 0
if 'interview_active' not in st.session_state:
    st.session_state.interview_active = False

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Career Navigator AI</h1>
        <p>Your AI-Powered Career Advancement Platform</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### 🔧 Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Enter GROQ API Key",
            type="password",
            help="Your GROQ API key for AI services"
        )
        
        if api_key:
            if st.session_state.backend is None:
                try:
                    with st.spinner("Initializing AI backend..."):
                        st.session_state.backend = CareerNavigatorBackend(api_key)
                    st.success("✅ AI Backend initialized successfully!")
                except Exception as e:
                    st.error(f"❌ Failed to initialize: {str(e)}")
                    return
        
        st.markdown("---")
        
        # Navigation
        st.markdown("### 📋 Navigation")
        page = st.radio(
            "Select Feature",
            ["📄 Resume Analysis", "🎯 ATS Feedback", "📝 Resume Summary", "🎤 Interview Simulation"]
        )
        
        st.markdown("---")
        
        # Info section
        st.markdown("### ℹ️ Features")
        st.markdown("""
        - **Resume Analysis**: Complete resume evaluation
        - **ATS Feedback**: Applicant Tracking System scoring
        - **Resume Summary**: Professional summary generation
        - **Interview Simulation**: AI-powered interview practice
        """)

    # Main content area
    if not api_key:
        st.markdown("""
        <div class="info-box">
            <h3>🔑 Welcome to Career Navigator AI</h3>
            <p>Please enter your GROQ API key in the sidebar to get started.</p>
            <p>This platform provides comprehensive career services including:</p>
            <ul>
                <li>Resume Analysis & Optimization</li>
                <li>ATS Compatibility Scoring</li>
                <li>Professional Resume Summaries</li>
                <li>AI-Powered Interview Simulations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        return

    if st.session_state.backend is None:
        st.warning("⚠️ Please wait for the AI backend to initialize...")
        return

    # File upload section (common for most features)
    if page in ["📄 Resume Analysis", "🎯 ATS Feedback", "📝 Resume Summary", "🎤 Interview Simulation"]:
        st.markdown("### 📁 Upload Your Resume")
        
        uploaded_file = st.file_uploader(
            "Choose your resume file",
            type=['pdf', 'docx'],
            help="Upload PDF or DOCX format resume"
        )
        
        if uploaded_file is not None:
            with st.expander("📋 File Details", expanded=False):
                st.write(f"**Filename:** {uploaded_file.name}")
                st.write(f"**File Size:** {uploaded_file.size} bytes")
                st.write(f"**File Type:** {uploaded_file.type}")

    # Page routing
    if page == "📄 Resume Analysis":
        show_resume_analysis(uploaded_file)
    elif page == "🎯 ATS Feedback":
        show_ats_feedback(uploaded_file)
    elif page == "📝 Resume Summary":
        show_resume_summary(uploaded_file)
    elif page == "🎤 Interview Simulation":
        show_interview_simulation(uploaded_file)

def show_resume_analysis(uploaded_file):
    st.markdown("### 📊 Complete Resume Analysis")
    
    if uploaded_file is not None:
        col1, col2 = st.columns([3, 1])
        
        with col2:
            if st.button("🔍 Analyze Resume", key="analyze_btn"):
                with st.spinner("🔄 Analyzing your resume..."):
                    results = st.session_state.backend.analyze_resume(uploaded_file)
                    st.session_state.analysis_results = results
        
        if st.session_state.analysis_results:
            results = st.session_state.analysis_results
            
            if results.get('success'):
                st.markdown('<div class="success-message">✅ Resume analysis completed successfully!</div>', unsafe_allow_html=True)
                
                # Display results in organized sections
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Role identification
                    st.markdown("#### 🎯 Identified Role")
                    st.markdown(f"**{results['role']}**")
                    
                    # ATS Feedback
                    st.markdown("#### 📈 ATS Feedback")
                    st.markdown(f"```\n{results['ats_feedback']}\n```")
                    
                    # Resume Summary
                    st.markdown("#### 📝 Professional Summary")
                    st.markdown(f"```\n{results['summary']}\n```")
                
                with col2:
                    # Keywords
                    st.markdown("#### 🔑 Extracted Keywords")
                    if results['keywords']:
                        for keyword in results['keywords'][:10]:  # Show top 10
                            st.markdown(f"• {keyword}")
                    else:
                        st.markdown("No keywords extracted")
                    
                    # Quick stats
                    st.markdown("#### 📊 Quick Stats")
                    st.markdown(f"**Text Length:** {len(results['resume_text'])} characters")
                    st.markdown(f"**Keywords Found:** {len(results['keywords'])}")
                    st.markdown(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
                
            else:
                st.markdown(f'<div class="error-message">❌ Analysis failed: {results.get("error", "Unknown error")}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">📁 Please upload a resume file to begin analysis.</div>', unsafe_allow_html=True)

def show_ats_feedback(uploaded_file):
    st.markdown("### 🎯 ATS Compatibility Analysis")
    
    if uploaded_file is not None:
        col1, col2 = st.columns([3, 1])
        
        with col2:
            if st.button("📊 Get ATS Score", key="ats_btn"):
                with st.spinner("🔄 Analyzing ATS compatibility..."):
                    resume_text = st.session_state.backend.extract_text_from_file(uploaded_file)
                    ats_feedback = st.session_state.backend.get_ats_feedback(resume_text)
                    st.session_state.ats_result = ats_feedback
        
        if hasattr(st.session_state, 'ats_result'):
            st.markdown("#### 📈 ATS Analysis Results")
            st.markdown(f"```\n{st.session_state.ats_result}\n```")
            
            # Additional tips
            st.markdown("#### 💡 ATS Optimization Tips")
            st.markdown("""
            - Use standard section headings (Experience, Education, Skills)
            - Include relevant keywords from job descriptions
            - Use simple, clean formatting
            - Avoid images, tables, and complex layouts
            - Save in ATS-friendly formats (PDF or DOCX)
            - Use standard fonts (Arial, Calibri, Times New Roman)
            """)
    else:
        st.markdown('<div class="info-box">📁 Please upload a resume file to get ATS feedback.</div>', unsafe_allow_html=True)

def show_resume_summary(uploaded_file):
    st.markdown("### 📝 Professional Resume Summary")
    
    if uploaded_file is not None:
        col1, col2 = st.columns([3, 1])
        
        with col2:
            if st.button("📄 Generate Summary", key="summary_btn"):
                with st.spinner("🔄 Generating professional summary..."):
                    resume_text = st.session_state.backend.extract_text_from_file(uploaded_file)
                    summary = st.session_state.backend.summarize_resume(resume_text)
                    st.session_state.summary_result = summary
        
        if hasattr(st.session_state, 'summary_result'):
            st.markdown("#### 📋 Generated Summary")
            st.markdown(f"```\n{st.session_state.summary_result}\n```")
            
            # Copy to clipboard button
            if st.button("📋 Copy Summary"):
                st.success("Summary copied to clipboard! (Feature simulated)")
            
            # Usage tips
            st.markdown("#### 💡 How to Use This Summary")
            st.markdown("""
            - Add to your LinkedIn profile
            - Use in cover letters
            - Include in email signatures
            - Adapt for different job applications
            - Use as elevator pitch talking points
            """)
    else:
        st.markdown('<div class="info-box">📁 Please upload a resume file to generate a summary.</div>', unsafe_allow_html=True)

def show_interview_simulation(uploaded_file):
    st.markdown("### 🎤 AI Interview Simulation")
    
    if uploaded_file is not None:
        # First, analyze resume to get role
        if st.session_state.analysis_results is None:
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("🔍 Analyze for Interview", key="interview_analyze_btn"):
                    with st.spinner("🔄 Analyzing resume for interview preparation..."):
                        results = st.session_state.backend.analyze_resume(uploaded_file)
                        st.session_state.analysis_results = results
        
        if st.session_state.analysis_results and st.session_state.analysis_results.get('success'):
            role = st.session_state.analysis_results['role']
            
            st.markdown(f"#### 🎯 Interview Role: **{role}**")
            
            # Interview controls
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                if st.button("🎤 Start Interview", key="start_interview"):
                    st.session_state.interview_active = True
                    st.session_state.current_question_index = 0
                    st.session_state.interview_questions = []
                    st.session_state.interview_answers = []
                    st.session_state.interview_evaluations = []
            
            with col2:
                if st.button("📝 Generate New Question", key="new_question"):
                    if st.session_state.interview_active:
                        with st.spinner("🔄 Generating new question..."):
                            question_result = st.session_state.backend.conduct_interview_session(role)
                            if question_result['success']:
                                st.session_state.interview_questions.append(question_result['question'])
            
            with col3:
                if st.button("🛑 End Interview", key="end_interview"):
                    st.session_state.interview_active = False
            
            # Interview session
            if st.session_state.interview_active:
                st.markdown("---")
                
                # Current question
                if st.session_state.interview_questions:
                    current_q_index = len(st.session_state.interview_questions) - 1
                    current_question = st.session_state.interview_questions[current_q_index]
                    
                    st.markdown(f"""
                    <div class="question-box">
                        <h4>❓ Interview Question {current_q_index + 1}</h4>
                        <p><strong>{current_question}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Answer input
                    answer = st.text_area(
                        "Your Answer:",
                        key=f"answer_{current_q_index}",
                        height=100,
                        placeholder="Type your answer here..."
                    )
                    
                    if st.button("✅ Submit Answer", key=f"submit_{current_q_index}"):
                        if answer.strip():
                            with st.spinner("🔄 Evaluating your answer..."):
                                evaluation = st.session_state.backend.evaluate_answer(role, current_question, answer)
                                st.session_state.interview_answers.append(answer)
                                st.session_state.interview_evaluations.append(evaluation)
                        else:
                            st.warning("Please provide an answer before submitting.")
                
                # Show previous Q&As
                if st.session_state.interview_evaluations:
                    st.markdown("#### 📊 Previous Questions & Evaluations")
                    
                    for i, (q, a, e) in enumerate(zip(
                        st.session_state.interview_questions[:-1] if len(st.session_state.interview_questions) > 1 else [],
                        st.session_state.interview_answers,
                        st.session_state.interview_evaluations
                    )):
                        with st.expander(f"Question {i+1}: {q[:50]}..."):
                            st.markdown(f"**Question:** {q}")
                            st.markdown(f"**Your Answer:** {a}")
                            st.markdown(f"""
                            <div class="evaluation-box">
                                <h5>📋 Evaluation</h5>
                                <p>{e}</p>
                            </div>
                            """, unsafe_allow_html=True)
            
            # Interview tips
            st.markdown("#### 💡 Interview Tips")
            st.markdown("""
            - **Be Specific**: Use concrete examples and numbers
            - **STAR Method**: Situation, Task, Action, Result
            - **Ask Questions**: Show genuine interest in the role
            - **Stay Calm**: Take your time to think before answering
            - **Practice**: The more you practice, the more confident you'll become
            """)
    else:
        st.markdown('<div class="info-box">📁 Please upload a resume file to start interview simulation.</div>', unsafe_allow_html=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CareerNavigatorBackend:
    def __init__(self, groq_api_key: str):
        """Initialize the Career Navigator with API key"""
        self.groq_api_key = groq_api_key
        self.llm = None
        self.chains = {}
        self.ner_model = None
        self._setup_llm()
        self._setup_chains()
        self._setup_ner_model()

    def _setup_llm(self):
        """Configure the LLM"""
        try:
            self.llm = ChatGroq(
                model_name="llama3-8b-8192",
                temperature=0.3,
                api_key=self.groq_api_key
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise

    def _setup_chains(self):
        """Setup all LLM chains"""
        role_prompt = PromptTemplate.from_template(
            """Analyze this resume and identify the most likely job role/position this person is seeking or qualified for.
            Consider their experience, skills, and background.
            
            Resume:
            {resume}
            
            Return only the specific job role title (e.g., "Software Engineer", "Data Scientist", "Marketing Manager"):"""
        )

        interview_question_prompt = PromptTemplate.from_template(
            """Generate a technical interview question for a {role} position.
            The question should be:
            - Relevant to the role
            - Moderately challenging
            - Practical and realistic
            
            
            Return only the question without any additional text:"""
        )

        evaluate_prompt = PromptTemplate.from_template(
            """Evaluate this interview answer for a {role} position:
            
            Question: {question}
            Answer: {answer}
            
            Provide:
            1. Score out of 10
            2. Brief explanation of strengths and weaknesses
            3. Suggestions for improvement
            
            Format your response as:
            Score: X/10
            Evaluation: [Your detailed feedback]"""
        )

        ats_prompt = PromptTemplate.from_template(
            """You are an ATS (Applicant Tracking System) analyzing this resume.
            
            Resume:
            {resume}
            
            Provide:
            1. Overall ATS score out of 100
            2. Key strengths identified
            3. Areas needing improvement
            4. Specific recommendations to improve ATS compatibility
            
            Format your response as:
            ATS Score: X/100
            Strengths: [List key strengths]
            Areas for Improvement: [List improvement areas]
            Recommendations: [Specific actionable recommendations]"""
        )

        summarize_prompt = PromptTemplate.from_template(
            """Create a concise professional summary of this resume:
            
            Resume:
            {resume}
            
            Provide:
            1. Professional summary (2-3 sentences)
            2. Key skills and expertise
            3. Years of experience
            4. Notable achievements
            
            Format your response clearly and professionally."""
        )

        self.chains = {
            'role': LLMChain(prompt=role_prompt, llm=self.llm),
            'question': LLMChain(prompt=interview_question_prompt, llm=self.llm),
            'evaluate': LLMChain(prompt=evaluate_prompt, llm=self.llm),
            'ats': LLMChain(prompt=ats_prompt, llm=self.llm),
            'summarize': LLMChain(prompt=summarize_prompt, llm=self.llm)
        }

    @st.cache_resource
    def _setup_ner_model(_self):
        """Setup NER model for keyword extraction"""
        try:
            return pipeline("ner", model="dslim/bert-base-NER")
        except Exception as e:
            logger.error(f"Failed to load NER model: {e}")
            return None

    def extract_text_from_file(self, uploaded_file) -> str:
        """Extract text from uploaded file"""
        try:
            if uploaded_file.type == "application/pdf":
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                doc = fitz.open("temp.pdf")
                text = "\n".join([page.get_text() for page in doc])
                doc.close()
                os.remove("temp.pdf")
                return text
                
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                with open("temp.docx", "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                doc = docx.Document("temp.docx")
                text = "\n".join([p.text for p in doc.paragraphs])
                os.remove("temp.docx")
                return text
                
            else:
                raise ValueError("Unsupported file format. Please upload PDF or DOCX files.")
                
        except Exception as e:
            logger.error(f"Failed to extract text from file: {e}")
            raise

    def extract_keywords(self, resume_text: str) -> List[str]:
        """Extract keywords using NER model"""
        try:
            if self.ner_model is None:
                self.ner_model = self._setup_ner_model()
            
            if self.ner_model is None:
                return []
                
            entities = self.ner_model(resume_text)
            keywords = list(set(ent["word"] for ent in entities if ent["entity"].startswith("B-")))
            
            keywords = [kw.replace("##", "").strip() for kw in keywords if len(kw) > 2]
            return keywords[:20]  
        except Exception as e:
            logger.error(f"Failed to extract keywords: {e}")
            return []

    def identify_role(self, resume_text: str) -> str:
        """Identify the most likely job role from resume"""
        try:
            role = self.chains['role'].run({"resume": resume_text}).strip()
            return role
        except Exception as e:
            logger.error(f"Failed to identify role: {e}")
            return "General Professional"

    def get_ats_feedback(self, resume_text: str) -> str:
        """Get ATS feedback and scoring"""
        try:
            feedback = self.chains['ats'].run({"resume": resume_text}).strip()
            return feedback
        except Exception as e:
            logger.error(f"Failed to get ATS feedback: {e}")
            return "Unable to generate ATS feedback at this time."

    def summarize_resume(self, resume_text: str) -> str:
        """Generate resume summary"""
        try:
            summary = self.chains['summarize'].run({"resume": resume_text}).strip()
            return summary
        except Exception as e:
            logger.error(f"Failed to summarize resume: {e}")
            return "Unable to generate resume summary at this time."

    def generate_interview_question(self, role: str) -> str:
        """Generate interview question for specific role"""
        try:
            question = self.chains['question'].run({"role": role}).strip()
            return question
        except Exception as e:
            logger.error(f"Failed to generate interview question: {e}")
            return f"Tell me about your experience in {role}?"

    def evaluate_answer(self, role: str, question: str, answer: str) -> str:
        """Evaluate interview answer"""
        try:
            evaluation = self.chains['evaluate'].run({
                "role": role,
                "question": question,
                "answer": answer
            }).strip()
            return evaluation
        except Exception as e:
            logger.error(f"Failed to evaluate answer: {e}")
            return "Score: 5/10\nEvaluation: Unable to evaluate answer at this time."

    def analyze_resume(self, uploaded_file) -> Dict[str, Any]:
        """Complete resume analysis pipeline"""
        try:
            resume_text = self.extract_text_from_file(uploaded_file)
            
            role = self.identify_role(resume_text)
            ats_feedback = self.get_ats_feedback(resume_text)
            summary = self.summarize_resume(resume_text)
            keywords = self.extract_keywords(resume_text)
            
            return {
                "resume_text": resume_text,
                "role": role,
                "ats_feedback": ats_feedback,
                "summary": summary,
                "keywords": keywords,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze resume: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def conduct_interview_session(self, role: str) -> Dict[str, str]:
        """Generate a single interview question for the session"""
        try:
            question = self.generate_interview_question(role)
            return {
                "question": question,
                "success": True
            }
        except Exception as e:
            logger.error(f"Failed to conduct interview: {e}")
            return {
                "success": False,
                "error": str(e)
            }

if __name__ == "__main__":
    main()
