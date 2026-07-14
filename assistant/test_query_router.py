from __future__ import annotations

import unittest
from typing import Any

from assistant.query_router import analyse_query


DOCUMENTS: list[dict[str, Any]] = [
    {
        "document_id": (
            "01_urinary_tract_infections_in_children"
        ),
        "title": (
            "Urinary Tract Infections in Children"
        ),
    },
    {
        "document_id": (
            "03_adrenal_insufficiency"
        ),
        "title": (
            "Adrenal Insufficiency and Addison's Disease"
        ),
    },
    {
        "document_id": (
            "04_amyloidosis_and_kidney_disease"
        ),
        "title": (
            "Amyloidosis and Kidney Disease"
        ),
    },
    {
        "document_id": (
            "00_short_bowel_syndrome"
        ),
        "title": "Short Bowel Syndrome",
    },
    {
        "document_id": (
            "02_abdominal_adhesions"
        ),
        "title": "Abdominal Adhesions",
    },
]


class QueryRouterTests(unittest.TestCase):
    """Verify that evidence requirements match question intent."""

    def test_general_text_route(self) -> None:
        """General questions should use ordinary hybrid retrieval."""
        route = analyse_query(
            question=(
                "What is Short Bowel Syndrome?"
            ),
            documents=DOCUMENTS,
        )

        self.assertEqual(
            route.intent,
            "general_text",
        )
        self.assertEqual(
            route.required_facts,
            ["clinical_answer"],
        )
        self.assertFalse(
            route.requires_figure
        )

    def test_classification_code_route(self) -> None:
        """Classification lookup must never require a figure."""
        route = analyse_query(
            question=(
                "What is the classification code for the "
                "Urinary Tract Infections in Children monograph?"
            ),
            documents=DOCUMENTS,
        )

        self.assertEqual(
            route.intent,
            "classification_code",
        )
        self.assertEqual(
            route.required_facts,
            ["classification_code"],
        )
        self.assertTrue(
            route.requires_metadata
        )
        self.assertTrue(
            route.requires_structured_table
        )
        self.assertFalse(
            route.requires_figure
        )

    def test_review_body_route(self) -> None:
        """Approval questions should request footnote evidence."""
        route = analyse_query(
            question=(
                "Which review body approved the Adrenal "
                "Insufficiency and Addison's Disease monograph?"
            ),
            documents=DOCUMENTS,
        )

        self.assertEqual(
            route.intent,
            "review_body",
        )
        self.assertIn(
            "footnote",
            route.retrieval_channels,
        )
        self.assertEqual(
            route.required_facts,
            ["review_body"],
        )
        self.assertFalse(
            route.requires_figure
        )

    def test_figure_route(self) -> None:
        """Explicit Figure 1 questions must require figure evidence."""
        route = analyse_query(
            question=(
                "According to Figure 1 in the Abdominal "
                "Adhesions monograph, what is the 2024 "
                "reported prevalence per 100,000?"
            ),
            documents=DOCUMENTS,
        )

        self.assertEqual(
            route.intent,
            "figure_value",
        )
        self.assertTrue(
            route.requires_figure
        )
        self.assertEqual(
            route.required_facts,
            ["figure_value"],
        )

    def test_scanned_appendix_route(self) -> None:
        """Scanned appendices should accept OCR text or tables."""
        route = analyse_query(
            question=(
                "In the scanned Appendix C of the Amyloidosis "
                "and Kidney Disease monograph, what is listed "
                "as the formulary agent of record and its "
                "induction dose?"
            ),
            documents=DOCUMENTS,
        )

        self.assertEqual(
            route.intent,
            "scanned_appendix",
        )
        self.assertFalse(
            route.requires_figure
        )
        self.assertEqual(
            set(route.required_facts),
            {
                "drug",
                "induction_dose",
            },
        )

    def test_landscape_table_route(self) -> None:
        """Appendix A maintenance questions should use tables."""
        route = analyse_query(
            question=(
                "In Appendix A, what maintenance dose of "
                "Hematoril is listed?"
            ),
            documents=DOCUMENTS,
        )

        self.assertEqual(
            route.intent,
            "maintenance_dose",
        )
        self.assertTrue(
            route.requires_structured_table
        )
        self.assertFalse(
            route.requires_figure
        )

    def test_cross_document_route(self) -> None:
        """Cross-document dosing should require a resolved link."""
        route = analyse_query(
            question=(
                "The Short Bowel Syndrome monograph tells you "
                "not to use local values for a drug's induction "
                "dose and points to another monograph. What is "
                "the dose?"
            ),
            documents=DOCUMENTS,
        )

        self.assertEqual(
            route.intent,
            "cross_document_dose",
        )
        self.assertTrue(
            route.requires_cross_document
        )
        self.assertFalse(
            route.requires_figure
        )
        self.assertEqual(
            set(route.required_facts),
            {
                "cross_document_link",
                "drug",
                "induction_dose",
            },
        )

    def test_corpus_tier_route(self) -> None:
        """Corpus tier questions should use deterministic aggregation."""
        route = analyse_query(
            question=(
                "List every condition in the corpus whose "
                "Document Control Record assigns Monitoring Tier 3."
            ),
            documents=DOCUMENTS,
        )

        self.assertEqual(
            route.intent,
            "corpus_monitoring_tier",
        )
        self.assertTrue(
            route.requires_corpus_aggregation
        )
        self.assertFalse(
            route.requires_figure
        )

    def test_reverse_registry_route(self) -> None:
        """Reverse registry lookup should use exact metadata."""
        route = analyse_query(
            question=(
                "What is the registry code CDR-1500 assigned to?"
            ),
            documents=DOCUMENTS,
        )

        self.assertEqual(
            route.intent,
            "reverse_registry_code",
        )
        self.assertTrue(
            route.requires_metadata
        )
        self.assertFalse(
            route.requires_figure
        )

    def test_non_figure_regression_questions(self) -> None:
        """Previously misrouted questions must not require figures."""
        questions = [
            (
                "What is the classification code for the "
                "Urinary Tract Infections in Children monograph?"
            ),
            (
                "Which review body approved the Adrenal "
                "Insufficiency and Addison's Disease monograph?"
            ),
            (
                "In the scanned Appendix C of the Amyloidosis "
                "and Kidney Disease monograph, what is the "
                "formulary agent of record and induction dose?"
            ),
            (
                "In Appendix A, what maintenance dose of "
                "Hematoril is listed?"
            ),
            (
                "The Short Bowel Syndrome monograph points to "
                "another monograph for an induction dose."
            ),
        ]

        for question in questions:
            with self.subTest(question=question):
                route = analyse_query(
                    question=question,
                    documents=DOCUMENTS,
                )

                self.assertFalse(
                    route.requires_figure,
                    msg=(
                        "Non-figure question was incorrectly "
                        f"routed to figures: {question}"
                    ),
                )


if __name__ == "__main__":
    unittest.main()