from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AgentResult:
    name: str
    message: str


class BaseAgent:
    name = "base_agent"

    def log(self, message: str) -> AgentResult:
        return AgentResult(name=self.name, message=message)
