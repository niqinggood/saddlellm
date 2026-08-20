"""Aggregate pretraining artifacts into one report."""
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class ReportFinding:
    severity: str
    area: str
    message: str

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PretrainReportResult:
    output_dir: str
    artifacts: Dict[str, str]
    sections: Dict[str, Dict]
    findings: List[ReportFinding] = field(default_factory=list)
    score: float = 0.0
    grade: str = "unknown"

    def to_dict(self) -> Dict:
        return {
            "output_dir": self.output_dir,
            "artifacts": self.artifacts,
            "sections": self.sections,
            "findings": [f.to_dict() for f in self.findings],
            "score": self.score,
            "grade": self.grade,
        }

    def to_markdown(self) -> str:
        lines = [
            "# Pretraining Report",
            "",
            f"- Output dir: `{self.output_dir}`",
            f"- Grade: **{self.grade}**",
            f"- Score: **{self.score:.1f}/100**",
            "",
            "## Findings",
        ]
        if self.findings:
            for item in self.findings:
                lines.append(f"- **{item.severity.upper()}** `{item.area}`: {item.message}")
        else:
            lines.append("- No major issues detected from available artifacts.")

        for name, section in self.sections.items():
            lines.extend(["", f"## {name.replace('_', ' ').title()}"])
            if not section:
                lines.append("- Not available")
                continue
            for key, value in section.items():
                if isinstance(value, (dict, list)):
                    compact = json.dumps(value, ensure_ascii=False)
                    if len(compact) > 500:
                        compact = compact[:500] + "..."
                    lines.append(f"- `{key}`: {compact}")
                else:
                    lines.append(f"- `{key}`: {value}")
        return "\n".join(lines)


class PretrainReport:
    """Build a run-level report from SaddleLLM pretraining outputs."""

    DEFAULT_FILES = {
        "summary": "summary.json",
        "tokenizer_eval": os.path.join("tokenizer", "tokenizer_eval.json"),
        "stability": os.path.join("stability", "stability_summary.json"),
        "model_metrics": os.path.join("model_metrics", "model_metrics_summary.json"),
        "eval_suite": os.path.join("eval_suite", "suite_results.json"),
        "data_mix": "data_mix_plan.json",
        "contamination": "contamination_report.json",
    }

    @classmethod
    def from_output_dir(cls, output_dir: str, extra_artifacts: Optional[Dict[str, str]] = None) -> PretrainReportResult:
        artifacts = cls._discover(output_dir, extra_artifacts or {})
        sections = {name: cls._load_json(path) for name, path in artifacts.items() if path and os.path.exists(path)}
        findings = cls._findings(sections)
        score = cls._score(findings, sections)
        return PretrainReportResult(
            output_dir=output_dir,
            artifacts=artifacts,
            sections=sections,
            findings=findings,
            score=score,
            grade=cls._grade(score),
        )

    @classmethod
    def save(
        cls,
        output_dir: str,
        report: PretrainReportResult,
        json_name: str = "pretrain_report.json",
        markdown_name: str = "pretrain_report.md",
    ) -> Dict[str, str]:
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, json_name)
        md_path = os.path.join(output_dir, markdown_name)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(report.to_markdown())
        return {"json": json_path, "markdown": md_path}

    @classmethod
    def generate(cls, output_dir: str, extra_artifacts: Optional[Dict[str, str]] = None) -> PretrainReportResult:
        report = cls.from_output_dir(output_dir, extra_artifacts=extra_artifacts)
        cls.save(output_dir, report)
        return report

    @classmethod
    def _discover(cls, output_dir: str, extra: Dict[str, str]) -> Dict[str, str]:
        artifacts = {}
        for name, rel_path in cls.DEFAULT_FILES.items():
            path = rel_path if os.path.isabs(rel_path) else os.path.join(output_dir, rel_path)
            if os.path.exists(path):
                artifacts[name] = path
        for name, path in extra.items():
            if path:
                artifacts[name] = path
        return artifacts

    @staticmethod
    def _load_json(path: str) -> Dict:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            return {"error": str(exc), "path": path}

    @classmethod
    def _findings(cls, sections: Dict[str, Dict]) -> List[ReportFinding]:
        findings: List[ReportFinding] = []
        tok = sections.get("tokenizer_eval", {})
        if tok:
            if tok.get("unk_rate", 0) > 0.001:
                findings.append(ReportFinding("warning", "tokenizer", f"UNK rate is high: {tok.get('unk_rate'):.4f}"))
            if tok.get("avg_tokens_per_domain_term", 0) > 6:
                findings.append(ReportFinding("warning", "tokenizer", "domain terms fragment heavily"))
            if tok.get("samples", 0) < 100:
                findings.append(ReportFinding("info", "tokenizer", "tokenizer evaluation used fewer than 100 samples"))
        else:
            findings.append(ReportFinding("info", "tokenizer", "tokenizer_eval.json not found"))

        stability = sections.get("stability", {})
        if stability:
            event_count = stability.get("event_count", len(stability.get("events", [])))
            if event_count:
                critical = any(e.get("severity") == "critical" for e in stability.get("events", []))
                findings.append(ReportFinding("critical" if critical else "warning", "stability", f"{event_count} stability events detected"))
            if stability.get("trend") == "increasing":
                findings.append(ReportFinding("warning", "stability", "loss trend is increasing"))
        else:
            findings.append(ReportFinding("info", "stability", "stability_summary.json not found"))

        model_metrics = sections.get("model_metrics", {})
        if model_metrics:
            arch = model_metrics.get("architecture", {})
            router = model_metrics.get("router", {})
            cache = model_metrics.get("cache", {})
            if arch.get("ffn_kind") == "moe":
                balance = router.get("mean_load_balance")
                if isinstance(balance, (int, float)) and balance < 0.80:
                    findings.append(ReportFinding("warning", "model", f"MoE load balance is low: {balance:.3f}"))
                else:
                    findings.append(ReportFinding("info", "model", "MoE router metrics are being tracked"))
            if cache.get("mode") == "mla_latent":
                findings.append(ReportFinding("info", "model", "MLA latent cache is enabled"))
            if arch.get("multi_token_prediction"):
                findings.append(ReportFinding("info", "model", "MTP objective is enabled"))

        eval_suite = sections.get("eval_suite", {})
        if eval_suite:
            for key, value in eval_suite.get("summary", {}).items():
                if key.endswith("perplexity") and isinstance(value, (int, float)) and value > 100:
                    findings.append(ReportFinding("warning", "eval", f"{key} perplexity is high: {value}"))
            for task, result in eval_suite.get("results", {}).items():
                if result.get("error"):
                    findings.append(ReportFinding("warning", "eval", f"{task} failed: {result.get('error')}"))
        else:
            findings.append(ReportFinding("info", "eval", "suite_results.json not found"))

        contamination = sections.get("contamination", {})
        if contamination:
            rate = contamination.get("contamination_rate", 0)
            if rate > 0.01:
                findings.append(ReportFinding("critical", "contamination", f"contamination rate {rate:.2%} exceeds 1%"))
            elif rate > 0:
                findings.append(ReportFinding("warning", "contamination", f"contamination rate {rate:.2%}"))
        return findings

    @staticmethod
    def _score(findings: List[ReportFinding], sections: Dict[str, Dict]) -> float:
        score = 100.0
        for finding in findings:
            if finding.severity == "critical":
                score -= 25
            elif finding.severity == "warning":
                score -= 10
            elif finding.severity == "info":
                score -= 2
        if "summary" not in sections:
            score -= 5
        return max(0.0, min(100.0, score))

    @staticmethod
    def _grade(score: float) -> str:
        if score >= 90:
            return "A"
        if score >= 75:
            return "B"
        if score >= 60:
            return "C"
        return "D"
