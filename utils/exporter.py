"""Export utilities for downloading session Q&A history as PDF or TXT."""
import datetime
import io
from typing import Any, Dict, List

from fpdf import FPDF

from utils.logger import setup_logger

logger = setup_logger("exporter")


class QAExporter:
    """Exports chat messages, evidence scores, and citations to downloadable PDF and TXT formats."""

    @classmethod
    def export_as_txt(cls, session_id: str, chat_history: List[Dict[str, Any]]) -> str:
        """Formats conversation history as a readable plain text file."""
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            "=" * 70,
            f"AskAnything — Q&A Session Export",
            f"Session ID : {session_id}",
            f"Exported At: {now}",
            "=" * 70,
            "",
        ]

        for idx, msg in enumerate(chat_history, start=1):
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            lines.append(f"[{role} - Message #{idx}]")
            lines.append(content)

            evidence_score = msg.get("evidence_score")
            if evidence_score:
                lines.append(f"-> Evidence Score: {evidence_score.get('label', '')}")

            citations = msg.get("citations", [])
            if citations:
                lines.append("-> Sources Cited:")
                for c in citations:
                    lines.append(f"   * {c.get('label', '')}")

            lines.append("-" * 70)
            lines.append("")

        return "\n".join(lines)

    @classmethod
    def export_as_pdf(cls, session_id: str, chat_history: List[Dict[str, Any]]) -> bytes:
        """Generates a styled PDF report using fpdf2."""
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # Header
        pdf.set_font("Helvetica", "B", 18)
        pdf.set_text_color(99, 102, 241)  # Indigo
        pdf.cell(0, 10, "AskAnything — Q&A Session Report", ln=True, align="C")

        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(100, 116, 139)  # Slate
        now_str = datetime.datetime.now().strftime("%B %d, %Y - %H:%M")
        pdf.cell(0, 6, f"Session: {session_id[:16]}... | Date: {now_str}", ln=True, align="C")
        pdf.ln(8)

        for idx, msg in enumerate(chat_history, start=1):
            role = msg.get("role", "User").capitalize()
            content = msg.get("content", "")

            # Role Title
            pdf.set_font("Helvetica", "B", 12)
            if role.lower() == "user":
                pdf.set_text_color(30, 41, 59)
                pdf.cell(0, 7, f"Question #{idx}", ln=True)
            else:
                pdf.set_text_color(79, 70, 229)
                pdf.cell(0, 7, f"Answer #{idx}", ln=True)

            # Message body (clean unicode for standard Latin-1 Helvetica in FPDF)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(51, 65, 85)

            # Normalize non-latin characters to prevent fpdf encoding crash
            safe_content = content.encode("latin-1", "replace").decode("latin-1")
            pdf.multi_cell(0, 6, safe_content)
            pdf.ln(3)

            # Evidence Score
            evidence = msg.get("evidence_score")
            if evidence and isinstance(evidence, dict):
                pdf.set_font("Helvetica", "I", 9)
                pdf.set_text_color(16, 185, 129)  # Emerald
                label = evidence.get("label", "").encode("latin-1", "replace").decode("latin-1")
                pdf.cell(0, 5, f"[Score] {label}", ln=True)

            # Citations
            citations = msg.get("citations", [])
            if citations:
                pdf.set_font("Helvetica", "B", 9)
                pdf.set_text_color(71, 85, 105)
                pdf.cell(0, 5, "Citations:", ln=True)
                pdf.set_font("Helvetica", "", 8)
                for cit in citations:
                    clabel = cit.get("label", "").encode("latin-1", "replace").decode("latin-1")
                    pdf.cell(0, 4, f"  * {clabel}", ln=True)

            pdf.ln(5)
            # Divider line
            pdf.set_draw_color(226, 232, 240)
            pdf.line(15, pdf.get_y(), 195, pdf.get_y())
            pdf.ln(5)

        return bytes(pdf.output())
