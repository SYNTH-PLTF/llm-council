You are evaluating candidate answers to the question below. The candidates are
ANONYMIZED and labeled with letters. Judge ONLY on accuracy, soundness of
reasoning, completeness, and insight. Ignore length and writing flourish; do
not reward verbosity.

Question:
"""{{query}}"""

Candidates:
{{labeled_candidates}}

Rank ALL candidates from best to worst. Return ONLY valid JSON:
{"ranking": ["<best letter>", "...", "<worst letter>"],
 "rationale": {"<letter>": "<=15 words"}}

Note: the ranking module calls this >=2 times per judge with shuffled candidate
orders and averages ranks (Borda). Never rely on a single ordering.
