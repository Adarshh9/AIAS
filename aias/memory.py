"""
AIAS Memory System
Personalized memory that learns and remembers user information across sessions
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict
from loguru import logger


@dataclass
class UserProfile:
    """User's personal information"""
    name: str = ""
    nickname: str = ""
    occupation: str = ""
    location: str = ""
    timezone: str = ""
    language: str = "English"
    interests: List[str] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    contacts: Dict[str, str] = field(default_factory=dict)  # name -> relationship
    preferences: Dict[str, Any] = field(default_factory=dict)
    custom_facts: List[str] = field(default_factory=list)  # Free-form facts about user
    
    def to_context(self) -> str:
        """Convert profile to context string for LLM"""
        lines = []
        
        if self.name:
            lines.append(f"User's name: {self.name}")
        if self.nickname:
            lines.append(f"Nickname/preferred name: {self.nickname}")
        if self.occupation:
            lines.append(f"Occupation: {self.occupation}")
        if self.location:
            lines.append(f"Location: {self.location}")
        if self.interests:
            lines.append(f"Interests: {', '.join(self.interests)}")
        if self.skills:
            lines.append(f"Skills: {', '.join(self.skills)}")
        if self.contacts:
            contacts_str = ", ".join([f"{name} ({rel})" for name, rel in self.contacts.items()])
            lines.append(f"Known contacts: {contacts_str}")
        if self.custom_facts:
            lines.append("Other facts about user:")
            for fact in self.custom_facts[-20:]:  # Last 20 facts
                lines.append(f"  - {fact}")
        
        return "\n".join(lines) if lines else "No user information stored yet."


@dataclass
class MemoryEntry:
    """A single memory entry"""
    timestamp: str
    query: str
    response: str
    extracted_facts: List[str] = field(default_factory=list)
    importance: int = 1  # 1-5 scale
    tags: List[str] = field(default_factory=list)


class PersonalMemory:
    """
    Persistent memory system that learns and remembers user information
    
    Features:
    - User profile storage (name, preferences, contacts, etc.)
    - Conversation history with fact extraction
    - Automatic context injection for personalized responses
    - Persistent storage across sessions
    """
    
    def __init__(self, memory_dir: str = "memory"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        
        self.profile_file = self.memory_dir / "user_profile.json"
        self.memories_file = self.memory_dir / "memories.json"
        self.facts_file = self.memory_dir / "learned_facts.json"
        
        # Load existing data
        self.profile = self._load_profile()
        self.memories: List[MemoryEntry] = self._load_memories()
        self.learned_facts: List[Dict[str, Any]] = self._load_facts()
        
        logger.info(f"Memory system initialized: {len(self.memories)} memories, {len(self.learned_facts)} facts")
    
    def _load_profile(self) -> UserProfile:
        """Load user profile from disk"""
        if self.profile_file.exists():
            try:
                with open(self.profile_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return UserProfile(**data)
            except Exception as e:
                logger.error(f"Failed to load profile: {e}")
        return UserProfile()
    
    def _save_profile(self):
        """Save user profile to disk"""
        try:
            with open(self.profile_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.profile), f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save profile: {e}")
    
    def _load_memories(self) -> List[MemoryEntry]:
        """Load conversation memories from disk"""
        if self.memories_file.exists():
            try:
                with open(self.memories_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return [MemoryEntry(**m) for m in data]
            except Exception as e:
                logger.error(f"Failed to load memories: {e}")
        return []
    
    def _save_memories(self):
        """Save conversation memories to disk"""
        try:
            # Keep last 500 memories to prevent file from growing too large
            recent_memories = self.memories[-500:]
            with open(self.memories_file, 'w', encoding='utf-8') as f:
                json.dump([asdict(m) for m in recent_memories], f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save memories: {e}")
    
    def _load_facts(self) -> List[Dict[str, Any]]:
        """Load learned facts from disk"""
        if self.facts_file.exists():
            try:
                with open(self.facts_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load facts: {e}")
        return []
    
    def _save_facts(self):
        """Save learned facts to disk"""
        try:
            # Keep last 200 facts
            recent_facts = self.learned_facts[-200:]
            with open(self.facts_file, 'w', encoding='utf-8') as f:
                json.dump(recent_facts, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save facts: {e}")
    
    def update_profile(self, **kwargs):
        """Update user profile with new information"""
        for key, value in kwargs.items():
            if hasattr(self.profile, key):
                if isinstance(getattr(self.profile, key), list) and not isinstance(value, list):
                    # Append to list fields
                    getattr(self.profile, key).append(value)
                elif isinstance(getattr(self.profile, key), dict) and isinstance(value, dict):
                    # Merge dict fields
                    getattr(self.profile, key).update(value)
                else:
                    setattr(self.profile, key, value)
        
        self._save_profile()
        logger.info(f"Profile updated: {list(kwargs.keys())}")
    
    def add_contact(self, name: str, relationship: str):
        """Add or update a contact"""
        self.profile.contacts[name] = relationship
        self._save_profile()
        logger.info(f"Contact added: {name} ({relationship})")
    
    def add_fact(self, fact: str, category: str = "general"):
        """Add a learned fact about the user"""
        self.learned_facts.append({
            "fact": fact,
            "category": category,
            "timestamp": datetime.now().isoformat(),
            "source": "conversation"
        })
        self.profile.custom_facts.append(fact)
        self._save_facts()
        self._save_profile()
        logger.debug(f"Fact learned: {fact}")
    
    def add_memory(self, query: str, response: str, extracted_facts: List[str] = None):
        """Add a conversation memory"""
        memory = MemoryEntry(
            timestamp=datetime.now().isoformat(),
            query=query,
            response=response[:500],  # Truncate long responses
            extracted_facts=extracted_facts or []
        )
        self.memories.append(memory)
        
        # Save extracted facts
        if extracted_facts:
            for fact in extracted_facts:
                self.add_fact(fact)
        
        self._save_memories()
    
    def is_remember_command(self, query: str) -> bool:
        """Check if user is asking to remember something from screen"""
        query_lower = query.lower().strip()
        remember_triggers = [
            "remember this",
            "remember that",
            "save this",
            "store this",
            "note this",
            "keep this",
            "memorize this",
            "its my resume",
            "it's my resume",
            "this is my resume",
            "my resume",
        ]
        return any(trigger in query_lower for trigger in remember_triggers)
    
    def extract_facts_from_response(self, query: str, response: str) -> List[str]:
        """
        Extract facts from LLM response when user says 'remember this'
        This parses the LLM's analysis of what it saw on screen
        """
        import re
        facts = []
        response_lower = response.lower()
        
        # ===== RESUME/PROFILE PATTERNS =====
        # Name
        name_patterns = [
            r'\*\*name\*\*[:\s]+([^\n*]+)',
            r'name[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            r'resume.*?of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
        ]
        for pattern in name_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                name = match.group(1).strip().rstrip(',.')
                if name and len(name) > 2:
                    facts.append(f"Full name: {name}")
                    # Update profile with full name
                    self.update_profile(name=name.split()[0])  # First name
                    self.profile.custom_facts.append(f"Full name: {name}")
                    break
        
        # Email
        email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
        emails = re.findall(email_pattern, response)
        for email in emails:
            facts.append(f"Email: {email}")
            self.update_profile(preferences={"email": email})
        
        # Phone
        phone_patterns = [
            r'\+?\d{1,3}[-.\s]?\d{4,5}[-.\s]?\d{4,5}',
            r'\*\*phone[^*]*\*\*[:\s]+([+\d\s-]+)',
        ]
        for pattern in phone_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for phone in matches:
                phone = phone.strip()
                if len(phone) >= 10:
                    facts.append(f"Phone: {phone}")
                    self.update_profile(preferences={"phone": phone})
                    break
        
        # Experience/Work
        exp_patterns = [
            r'\*\*([^*]+)\*\*[:\s]*(AI Intern|Intern|Developer|Engineer|Manager)[^\n]*',
            r'(Supermaya|Google|Microsoft|Amazon|Meta|Apple)[^\n]*',
        ]
        for pattern in exp_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in matches[:3]:  # Limit to 3
                if isinstance(match, tuple):
                    exp = ' '.join(match).strip()
                else:
                    exp = match.strip()
                if exp and len(exp) > 5:
                    facts.append(f"Experience: {exp[:100]}")
        
        # Skills/Technologies
        tech_keywords = ['python', 'javascript', 'react', 'node', 'tensorflow', 'pytorch', 
                         'llm', 'ai', 'ml', 'deep learning', 'computer vision', 'nlp',
                         'docker', 'kubernetes', 'aws', 'gcp', 'azure']
        found_skills = []
        for tech in tech_keywords:
            if tech in response_lower:
                found_skills.append(tech)
        if found_skills:
            facts.append(f"Skills: {', '.join(found_skills[:10])}")
            for skill in found_skills[:5]:
                if skill not in self.profile.skills:
                    self.profile.skills.append(skill)
        
        # Education
        edu_patterns = [
            r'(B\.?Tech|M\.?Tech|Bachelor|Master|PhD|B\.?S\.?|M\.?S\.?)[^\n,]*',
        ]
        for pattern in edu_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for edu in matches[:2]:
                facts.append(f"Education: {edu}")
        
        # Certifications
        cert_patterns = [
            r'(certified|certification|certificate)[^\n]*',
            r'(postman|google|aws|azure)[^\n]*(expert|certified|certificate)[^\n]*',
        ]
        for pattern in cert_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for cert in matches[:3]:
                if isinstance(cert, tuple):
                    cert = ' '.join(cert)
                facts.append(f"Certification: {cert[:80]}")
        
        # Save all facts
        if facts:
            self._save_profile()
            for fact in facts:
                self.add_fact(fact, category="resume")
            logger.info(f"Extracted {len(facts)} facts from screen content")
        
        return facts
    
    def get_recent_memories(self, n: int = 5) -> List[MemoryEntry]:
        """Get n most recent memories"""
        return self.memories[-n:]
    
    def search_memories(self, keyword: str) -> List[MemoryEntry]:
        """Search memories for keyword"""
        keyword_lower = keyword.lower()
        return [
            m for m in self.memories
            if keyword_lower in m.query.lower() or keyword_lower in m.response.lower()
        ]
    
    def get_context_for_query(self, query: str) -> str:
        """
        Build context string to inject into LLM prompt
        Includes user profile and relevant memories
        """
        context_parts = []
        
        # User profile
        profile_context = self.profile.to_context()
        if profile_context and profile_context != "No user information stored yet.":
            context_parts.append("=== USER PROFILE ===")
            context_parts.append(profile_context)
        
        # Recent conversation history (last 3)
        recent = self.get_recent_memories(3)
        if recent:
            context_parts.append("\n=== RECENT CONVERSATION ===")
            for m in recent:
                context_parts.append(f"User asked: {m.query[:100]}")
                context_parts.append(f"You answered: {m.response[:150]}...")
        
        # Relevant learned facts (search by keywords in query)
        relevant_facts = []
        query_words = set(query.lower().split())
        for fact_entry in self.learned_facts[-50:]:  # Check recent facts
            fact_lower = fact_entry["fact"].lower()
            if any(word in fact_lower for word in query_words if len(word) > 3):
                relevant_facts.append(fact_entry["fact"])
        
        if relevant_facts:
            context_parts.append("\n=== RELEVANT FACTS ===")
            for fact in relevant_facts[-5:]:
                context_parts.append(f"- {fact}")
        
        return "\n".join(context_parts) if context_parts else ""
    
    def extract_facts_from_conversation(self, query: str, response: str) -> List[str]:
        """
        Extract potential facts from conversation to remember
        This is a simple heuristic-based extraction
        """
        facts = []
        query_lower = query.lower()
        
        # ========== EXPLICIT MEMORY COMMANDS ==========
        # User can explicitly tell AIAS to remember something
        explicit_patterns = [
            "remember that",
            "remember this:",
            "remember my",
            "store this:",
            "save this:",
            "note that",
            "don't forget",
            "keep in mind",
            "fyi",
            "for your info",
        ]
        
        for pattern in explicit_patterns:
            if pattern in query_lower:
                # Extract everything after the pattern
                idx = query_lower.find(pattern)
                value = query[idx + len(pattern):].strip()
                # Clean up common prefixes
                for prefix in [":", "-", "that", "this"]:
                    if value.lower().startswith(prefix):
                        value = value[len(prefix):].strip()
                
                if value and len(value) > 3:
                    fact = f"User explicitly noted: {value}"
                    facts.append(fact)
                    self.profile.custom_facts.append(value)
                    self._save_profile()
                    logger.info(f"Explicit memory stored: {value}")
                return facts  # Return early for explicit commands
        
        # ========== URL DETECTION ==========
        # Detect URLs and what they belong to
        import re
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, query)
        
        for url in urls:
            # Try to determine what the URL is for
            url_context = query_lower
            
            if any(word in url_context for word in ["portfolio", "my site", "my website", "personal site"]):
                fact = f"User's portfolio site: {url}"
                facts.append(fact)
                self.update_profile(preferences={"portfolio_url": url})
                
            elif any(word in url_context for word in ["github", "repo", "repository"]):
                fact = f"User's GitHub/repo: {url}"
                facts.append(fact)
                self.update_profile(preferences={"github_url": url})
                
            elif any(word in url_context for word in ["linkedin"]):
                fact = f"User's LinkedIn: {url}"
                facts.append(fact)
                self.update_profile(preferences={"linkedin_url": url})
                
            elif any(word in url_context for word in ["its mine", "it's mine", "my ", "belongs to me", "i own"]):
                fact = f"User's URL: {url}"
                facts.append(fact)
                self.update_profile(preferences={"saved_url": url})
            else:
                # Generic URL mention
                fact = f"User mentioned URL: {url}"
                facts.append(fact)
        
        # ========== PROJECT/WORK PATTERNS ==========
        project_patterns = [
            ("working on", "project"),
            ("building", "project"),
            ("my project", "project"),
            ("created", "project"),
            ("developed", "project"),
        ]
        
        for pattern, category in project_patterns:
            if pattern in query_lower:
                idx = query_lower.find(pattern)
                value_start = idx + len(pattern)
                rest = query[value_start:].strip()
                for delimiter in [",", ".", "!", "?"]:
                    if delimiter in rest:
                        rest = rest.split(delimiter)[0]
                if rest and len(rest) > 2:
                    fact = f"User {pattern}: {rest.strip()}"
                    facts.append(fact)
        
        # ========== ORIGINAL SELF-INTRODUCTION PATTERNS ==========
        patterns = [
            ("my name is", "name"),
            ("i am called", "name"),
            ("i'm called", "name"),
            ("call me", "nickname"),
            ("i work as", "occupation"),
            ("i'm a ", "occupation"),
            ("i am a ", "occupation"),
            ("i work at", "occupation"),
            ("i live in", "location"),
            ("i'm from", "location"),
            ("i am from", "location"),
            ("i like", "interest"),
            ("i love", "interest"),
            ("i enjoy", "interest"),
            ("i'm interested in", "interest"),
            ("i know", "skill"),
            ("i can", "skill"),
            ("my email is", "email"),
            ("my phone is", "phone"),
            ("my friend", "contact"),
            ("my brother", "contact"),
            ("my sister", "contact"),
            ("my mom", "contact"),
            ("my dad", "contact"),
            ("my wife", "contact"),
            ("my husband", "contact"),
            ("my girlfriend", "contact"),
            ("my boyfriend", "contact"),
            ("my boss", "contact"),
            ("my colleague", "contact"),
        ]
        
        for pattern, category in patterns:
            if pattern in query_lower:
                # Extract the value after the pattern
                idx = query_lower.find(pattern)
                value_start = idx + len(pattern)
                # Get the rest of the sentence (up to punctuation or end)
                rest = query[value_start:].strip()
                # Take first meaningful chunk
                for delimiter in [",", ".", "!", "?", " and ", " but "]:
                    if delimiter in rest:
                        rest = rest.split(delimiter)[0]
                
                if rest and len(rest) > 1:
                    fact = f"User {pattern} {rest.strip()}"
                    facts.append(fact)
                    
                    # Also update profile directly for known categories
                    if category == "name":
                        self.update_profile(name=rest.strip())
                    elif category == "nickname":
                        self.update_profile(nickname=rest.strip())
                    elif category == "occupation":
                        self.update_profile(occupation=rest.strip())
                    elif category == "location":
                        self.update_profile(location=rest.strip())
                    elif category == "interest":
                        self.update_profile(interests=rest.strip())
                    elif category == "skill":
                        self.update_profile(skills=rest.strip())
        
        return facts
    
    def get_system_prompt_addition(self) -> str:
        """Get additional system prompt with memory context"""
        return """
MEMORY INSTRUCTIONS:
- You have access to the user's personal information and conversation history below.
- Use this information to personalize your responses.
- Remember details the user has shared and reference them when relevant.
- If the user shares new information about themselves, acknowledge it naturally.
- Address the user by name when appropriate.
- Build on previous conversations and maintain continuity.
"""
    
    def clear_all(self):
        """Clear all memory (use with caution)"""
        self.profile = UserProfile()
        self.memories = []
        self.learned_facts = []
        self._save_profile()
        self._save_memories()
        self._save_facts()
        logger.info("All memory cleared")
    
    def export_all(self) -> dict:
        """Export all memory data"""
        return {
            "profile": asdict(self.profile),
            "memories": [asdict(m) for m in self.memories],
            "learned_facts": self.learned_facts
        }
    
    def handle_memory_command(self, query: str) -> Optional[str]:
        """
        Handle special memory-related commands from user
        Returns a response if command was handled, None otherwise
        """
        query_lower = query.lower().strip()
        
        # Show what AIAS knows about user
        if query_lower in ["what do you know about me", "show my profile", "my info", "what do you remember"]:
            profile_str = self.profile.to_context()
            facts_count = len(self.learned_facts)
            memory_count = len(self.memories)
            return f"📋 **Your Profile:**\n{profile_str}\n\n📊 Stats: {facts_count} facts learned, {memory_count} conversations remembered"
        
        # Clear memory
        if query_lower in ["clear my memory", "forget everything", "reset memory"]:
            self.clear_all()
            return "🗑️ Memory cleared. Starting fresh!"
        
        # Show recent facts
        if query_lower in ["show facts", "what facts", "list facts"]:
            if not self.learned_facts:
                return "No facts stored yet. Tell me about yourself!"
            recent = self.learned_facts[-10:]
            facts_str = "\n".join([f"• {f['fact']}" for f in recent])
            return f"📝 **Recent Facts ({len(self.learned_facts)} total):**\n{facts_str}"
        
        # Show preferences/URLs
        if query_lower in ["show urls", "my urls", "my links", "show preferences"]:
            prefs = self.profile.preferences
            if not prefs:
                return "No URLs/preferences saved yet."
            prefs_str = "\n".join([f"• {k}: {v}" for k, v in prefs.items()])
            return f"🔗 **Saved Preferences:**\n{prefs_str}"
        
        return None  # Not a memory command
    
    def force_save_fact(self, fact: str, category: str = "manual") -> str:
        """Manually add a fact - called when user explicitly asks to remember"""
        self.learned_facts.append({
            "fact": fact,
            "category": category,
            "timestamp": datetime.now().isoformat(),
            "source": "user_explicit"
        })
        self.profile.custom_facts.append(fact)
        self._save_facts()
        self._save_profile()
        logger.info(f"Manual fact saved: {fact}")
        return f"✅ Saved: {fact}"
