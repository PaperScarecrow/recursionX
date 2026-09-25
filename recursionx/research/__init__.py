"""Research loop: gather candidate examples for a new skill from knowledge
sources, verify them, and package them as a SkillRecord for the wake phase."""
from .loop import EpisodeTask, ResearchLoop, ResearchReport, encode
from .sources import (KnowledgeSource, OracleSource, ProgramSource, ResearchResult, SkillRequest,
                      TeacherLLMSource, parse_examples)
from .verify import AgreementVerifier, ConsistencyVerifier, ExecutionVerifier, run_verifiers
