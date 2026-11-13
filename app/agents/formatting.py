"""
Shared output-formatting rules for every AI feature that produces text a student reads (chat, quiz, flashcards, summary).

The Web app renders this text with one renderer (Markdown + KaTeX + highlighted code), so the model is asked for exactly one
syntax. The renderer also repairs older content, so these rules improve consistency; they are not what makes rendering work.
"""

MARKDOWN_RULES = r"""
FORMATTING (the answer is rendered as Markdown with mathematics):
- Use valid Markdown: headings (## then ###), bullet and numbered lists, **bold**, and tables with a header row and a | --- | separator row.
- Mathematics: inline maths between single dollar signs, for example $x^2 + y^2 = z^2$ or $\frac{2}{5}$; display equations between double dollar
  signs on their own lines. Never use \( \) or \[ \] delimiters, and never put plain-language words inside maths without \text{...}.
- Write currency as plain text such as 5 dollars or USD 5, not with a dollar sign next to a number.
- Chemical formulas and reactions: always use Unicode (H₂O, CO₂, SO₄²⁻, Na⁺, 2H₂ + O₂ → 2H₂O, ⇌), never \ce{...}.
- Keep maths simple: fractions, powers, roots, Greek letters and integrals. Avoid \begin{...} environments, matrices and \text blocks.
- Everyday symbols may be written directly in Unicode (π, θ, Δ, °C, ≤, ≥, ≈, ±, ×, √).
- Code goes in fenced blocks with a language tag, for example ```python.
- Worked solutions use numbered steps. Do not use HTML tags.
- Ordinary prose (history, literature, business) must stay ordinary prose: no maths delimiters around normal sentences.
"""

JSON_RULES = r"""
FORMATTING of every text value inside the JSON:
- The text is rendered as Markdown with mathematics: inline maths between single dollar signs ($\frac{2}{5}$), display maths between double
  dollar signs. Never use \( \) or \[ \].
- Because this is JSON, every backslash inside a string MUST be written twice, for example "$\\frac{2}{5} + \\frac{3}{10}$" and "$6.02 \\times 10^{23}$".
- Write chemical formulas in Unicode (H₂O, CO₂, 2H₂ + O₂ → 2H₂O). Ordinary prose stays ordinary prose. Do not use HTML tags.
"""


def with_markdown_rules(prompt: str) -> str:
    return f"{prompt}\n{MARKDOWN_RULES}"


def with_json_rules(prompt: str) -> str:
    return f"{prompt}\n{JSON_RULES}"
