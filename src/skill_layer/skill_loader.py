"""
Skill Loader — loads Skill YAML files from disk and caches them in memory.

Usage:
    loader = SkillLoader()
    skill = loader.get("search.pico_term_generation")
    skills = loader.get_by_stage("search")
    skills = loader.get_by_trigger(stage="search", step="step_1_pico_generation")
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .skill_schema import Skill

logger = logging.getLogger(__name__)


class SkillLoader:
    """
    Loads and caches Skill objects from the skills/ directory.

    Directory layout expected:
        skills/
          search/         ← stage-level subdirectory
            pico_term_generation.yaml
            ...
          screening/
          extraction/
          synthesis/
          domain/
          templates/
    """

    def __init__(self, skills_dir: Optional[Path] = None):
        if skills_dir is None:
            from config import SKILLS_DIR
            skills_dir = SKILLS_DIR

        self._skills_dir = Path(skills_dir)
        self._cache: Dict[str, Skill] = {}  # skill_id → Skill
        self._loaded = False

    # ── Public API ────────────────────────────────────────────────────────────

    def load_all(self) -> None:
        """Scan skills_dir recursively and load every YAML file."""
        if self._loaded:
            return
        for yaml_file in self._skills_dir.rglob("*.yaml"):
            self._load_file(yaml_file)
        self._loaded = True
        logger.info("SkillLoader: loaded %d skills from %s", len(self._cache), self._skills_dir)

    def get(self, skill_id: str) -> Skill:
        """Return a Skill by its skill_id. Raises KeyError if not found."""
        if not self._loaded:
            self.load_all()
        if skill_id not in self._cache:
            raise KeyError(f"Skill '{skill_id}' not found. Available: {list(self._cache.keys())}")
        return self._cache[skill_id]

    def get_many(self, skill_ids: List[str]) -> List[Skill]:
        """Return multiple Skills by their IDs."""
        return [self.get(sid) for sid in skill_ids]

    def get_by_stage(self, stage: str) -> List[Skill]:
        """Return all Skills whose trigger.stage matches."""
        if not self._loaded:
            self.load_all()
        return [s for s in self._cache.values() if s.trigger.stage == stage]

    def get_by_trigger(self, stage: str, step: str) -> Optional[Skill]:
        """Return the Skill whose trigger matches the given stage+step pair."""
        if not self._loaded:
            self.load_all()
        for skill in self._cache.values():
            if skill.trigger.stage == stage and skill.trigger.step == step:
                return skill
        return None

    def list_ids(self) -> List[str]:
        """Return all loaded skill IDs."""
        if not self._loaded:
            self.load_all()
        return list(self._cache.keys())

    def reload(self) -> None:
        """Clear cache and reload from disk (useful during development)."""
        self._cache.clear()
        self._loaded = False
        self.load_all()

    # ── Internals ─────────────────────────────────────────────────────────────

    def _load_file(self, path: Path) -> None:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            if not isinstance(data, dict) or "skill_id" not in data:
                logger.warning("Skipping %s: missing skill_id", path)
                return
            skill = Skill.from_dict(data)
            if skill.skill_id in self._cache:
                logger.warning(
                    "Duplicate skill_id '%s' from %s — overwriting previous entry.",
                    skill.skill_id,
                    path,
                )
            self._cache[skill.skill_id] = skill
            logger.debug("Loaded skill: %s (v%s)", skill.skill_id, skill.version)
        except Exception as exc:
            logger.error("Failed to load skill from %s: %s", path, exc)


# Module-level singleton for convenience
_default_loader: Optional[SkillLoader] = None


def get_loader() -> SkillLoader:
    """Return the module-level singleton SkillLoader (lazy-initialized)."""
    global _default_loader
    if _default_loader is None:
        _default_loader = SkillLoader()
    return _default_loader
