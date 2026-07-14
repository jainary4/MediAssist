"""Create figure evidence while avoiding duplicate table images."""

from __future__ import annotations

import re
from pathlib import Path
from statistics import median
from typing import Any

import pytesseract
from PIL import Image, ImageOps

from retrieval.config import EvidenceBuilderConfig
from retrieval.models import EvidenceChunk
from retrieval.utils import (
    item_pages,
    label_value,
    normalize_text,
    source_reference,
    stable_id,
)


MINIMUM_TABLE_OVERLAP = 0.65
MINIMUM_TABLE_TEXT_OVERLAP = 0.55
MINIMUM_OCR_CONFIDENCE = 30.0


class FigureEvidenceBuilder:
    """Build evidence from true figures and merge table visuals."""

    def __init__(
        self,
        config: EvidenceBuilderConfig,
    ) -> None:
        """Initialize the figure evidence builder.

        Args:
            config (EvidenceBuilderConfig): Builder settings.
        """
        self.config = config

    def build(
        self,
        document: Any,
        document_record: dict[str, Any],
        final_directory: Path,
        table_records: list[dict[str, Any]],
    ) -> tuple[
        list[dict[str, Any]],
        list[EvidenceChunk],
        list[dict[str, Any]],
    ]:
        """Build valid figure records and picture audit results.

        Args:
            document (Any): Loaded DoclingDocument.
            document_record (dict[str, Any]): Source metadata.
            final_directory (Path): Final ingestion directory.
            table_records (list[dict[str, Any]]): Tables that may
                receive a visual asset from a PictureItem.

        Returns:
            tuple: Figure records, figure chunks, and audit records.
        """
        figures: list[dict[str, Any]] = []
        chunks: list[EvidenceChunk] = []
        audit_records: list[dict[str, Any]] = []

        pictures = list(
            getattr(document, "pictures", []) or []
        )

        docling_tables = list(
            getattr(document, "tables", []) or []
        )

        context_by_reference = (
            self._build_context_map(document)
        )

        available_assets = sorted(
            (
                final_directory / "figures"
            ).glob("figure_*.png")
        )

        for figure_number, picture in enumerate(
            pictures,
            start=1,
        ):
            source_ref = source_reference(picture)
            pages = item_pages(picture)

            context = context_by_reference.get(
                source_ref,
                {
                    "section": "",
                    "nearby_text": "",
                },
            )

            caption = self._caption_text(
                picture=picture,
                document=document,
            )

            asset_path = self._find_asset(
                final_directory=final_directory,
                figure_number=figure_number,
                available_assets=available_assets,
            )

            asset_status = self._inspect_asset(
                asset_path
            )

            overlapping_table_index = (
                self._find_overlapping_table(
                    picture=picture,
                    tables=docling_tables,
                )
            )

            if overlapping_table_index is not None:
                table_record = table_records[
                    overlapping_table_index
                ]

                self._attach_asset_to_table(
                    table_record=table_record,
                    asset_path=asset_path,
                )

                audit_records.append({
                    "document_id": document_record[
                        "document_id"
                    ],
                    "picture_number": figure_number,
                    "page_numbers": pages,
                    "source_ref": source_ref,
                    "decision": "merged_into_table",
                    "reason": (
                        "Picture region overlaps a "
                        "structured Docling table."
                    ),
                    "matched_table_id": table_record[
                        "table_id"
                    ],
                    "asset_path": (
                        str(asset_path)
                        if asset_path
                        else None
                    ),
                    "asset_found": asset_status[
                        "asset_found"
                    ],
                    "ocr_attempted": False,
                    "ocr_quality": "not_required",
                })

                continue

            ocr_result = self._run_figure_ocr(
                asset_path
            )

            text_matched_table_index = (
                self._find_table_by_ocr_text(
                    ocr_text=ocr_result["ocr_text"],
                    table_records=table_records,
                    picture_pages=pages,
                )
            )

            if (
                not caption
                and text_matched_table_index
                is not None
            ):
                table_record = table_records[
                    text_matched_table_index
                ]

                self._attach_asset_to_table(
                    table_record=table_record,
                    asset_path=asset_path,
                )

                audit_records.append({
                    "document_id": document_record[
                        "document_id"
                    ],
                    "picture_number": figure_number,
                    "page_numbers": pages,
                    "source_ref": source_ref,
                    "decision": "merged_into_table",
                    "reason": (
                        "Picture OCR substantially "
                        "matches structured table text."
                    ),
                    "matched_table_id": table_record[
                        "table_id"
                    ],
                    "asset_path": (
                        str(asset_path)
                        if asset_path
                        else None
                    ),
                    "asset_found": asset_status[
                        "asset_found"
                    ],
                    "ocr_attempted": ocr_result[
                        "ocr_attempted"
                    ],
                    "ocr_quality": "table_duplicate",
                })

                continue

            meaningful_content = bool(
                caption
                or ocr_result["ocr_text"]
            )

            if not meaningful_content:
                audit_records.append({
                    "document_id": document_record[
                        "document_id"
                    ],
                    "picture_number": figure_number,
                    "page_numbers": pages,
                    "source_ref": source_ref,
                    "decision": "rejected",
                    "reason": (
                        "No independent caption, OCR "
                        "text, or structured meaning."
                    ),
                    "asset_path": (
                        str(asset_path)
                        if asset_path
                        else None
                    ),
                    "asset_found": asset_status[
                        "asset_found"
                    ],
                    "ocr_attempted": ocr_result[
                        "ocr_attempted"
                    ],
                    "ocr_quality": ocr_result[
                        "ocr_quality"
                    ],
                })

                continue

            requires_visual_check = (
                not asset_status["asset_found"]
                or not pages
                or ocr_result[
                    "requires_visual_check"
                ]
            )

            figure_id = stable_id(
                "figure",
                document_record["document_id"],
                source_ref or figure_number,
            )

            figure_record = {
                "figure_id": figure_id,
                "document_id": document_record[
                    "document_id"
                ],
                "figure_number": self._printed_number(
                    caption=caption,
                    fallback=figure_number,
                ),
                "caption": caption,
                "section": context["section"],
                "nearby_text": context[
                    "nearby_text"
                ],
                "ocr_text": ocr_result["ocr_text"],
                "ocr_tokens": ocr_result[
                    "ocr_tokens"
                ],
                "ocr_attempted": ocr_result[
                    "ocr_attempted"
                ],
                "ocr_quality": ocr_result[
                    "ocr_quality"
                ],
                "ocr_median_confidence": ocr_result[
                    "median_confidence"
                ],
                "numeric_tokens": ocr_result[
                    "numeric_tokens"
                ],
                "page_numbers": pages,
                "source_ref": source_ref,
                "asset_path": (
                    str(asset_path)
                    if asset_path
                    else None
                ),
                "requires_visual_check": (
                    requires_visual_check
                ),
            }

            figures.append(figure_record)

            search_text = self._search_text(
                document_title=document_record[
                    "title"
                ],
                section=context["section"],
                caption=caption,
                nearby_text=context[
                    "nearby_text"
                ],
                ocr_text=ocr_result["ocr_text"],
            )

            chunks.append(
                EvidenceChunk(
                    chunk_id=stable_id(
                        "figure-chunk",
                        figure_id,
                    ),
                    document_id=document_record[
                        "document_id"
                    ],
                    content_type="figure",
                    title=document_record["title"],
                    section=context["section"],
                    search_text=search_text,
                    display_text=search_text,
                    page_numbers=pages,
                    source_refs=(
                        [source_ref]
                        if source_ref
                        else []
                    ),
                    parent_id=figure_id,
                    asset_path=(
                        str(asset_path)
                        if asset_path
                        else None
                    ),
                    ingestion_quality=document_record[
                        "ingestion_quality"
                    ],
                    metadata={
                        "figure_id": figure_id,
                        "figure_number": (
                            figure_record[
                                "figure_number"
                            ]
                        ),
                        "caption": caption,
                        "has_ocr_text": bool(
                            ocr_result["ocr_text"]
                        ),
                        "ocr_attempted": ocr_result[
                            "ocr_attempted"
                        ],
                        "ocr_quality": ocr_result[
                            "ocr_quality"
                        ],
                        "ocr_median_confidence": (
                            ocr_result[
                                "median_confidence"
                            ]
                        ),
                        "numeric_tokens": ocr_result[
                            "numeric_tokens"
                        ],
                        "ocr_tokens": ocr_result[
                            "ocr_tokens"
                        ],
                        "requires_visual_check": (
                            requires_visual_check
                        ),
                    },
                )
            )

            audit_records.append({
                "document_id": document_record[
                    "document_id"
                ],
                "picture_number": figure_number,
                "page_numbers": pages,
                "source_ref": source_ref,
                "decision": "indexed_as_figure",
                "reason": (
                    "Independent figure with caption "
                    "or meaningful OCR content."
                ),
                "asset_path": (
                    str(asset_path)
                    if asset_path
                    else None
                ),
                "asset_found": asset_status[
                    "asset_found"
                ],
                "ocr_attempted": ocr_result[
                    "ocr_attempted"
                ],
                "ocr_quality": ocr_result[
                    "ocr_quality"
                ],
                "requires_visual_check": (
                    requires_visual_check
                ),
            })

        return figures, chunks, audit_records

    @staticmethod
    def _find_asset(
        final_directory: Path,
        figure_number: int,
        available_assets: list[Path],
    ) -> Path | None:
        """Find a flat zero-padded figure image.

        Args:
            final_directory (Path): Final ingestion directory.
            figure_number (int): One-based PictureItem order.
            available_assets (list[Path]): Discovered figure files.

        Returns:
            Path | None: Figure image path.
        """
        expected_path = (
            final_directory
            / "figures"
            / f"figure_{figure_number:03d}.png"
        )

        if expected_path.is_file():
            return expected_path

        fallback_index = figure_number - 1

        if fallback_index < len(available_assets):
            candidate = available_assets[
                fallback_index
            ]

            if candidate.is_file():
                return candidate

        return None

    @staticmethod
    def _inspect_asset(
        asset_path: Path | None,
    ) -> dict[str, Any]:
        """Check whether an image asset is readable.

        Args:
            asset_path (Path | None): Candidate image path.

        Returns:
            dict[str, Any]: Asset quality information.
        """
        if asset_path is None:
            return {
                "asset_found": False,
                "image_readable": False,
                "width": 0,
                "height": 0,
            }

        try:
            with Image.open(asset_path) as image:
                width, height = image.size
                image.verify()

            return {
                "asset_found": True,
                "image_readable": True,
                "width": width,
                "height": height,
            }
        except OSError:
            return {
                "asset_found": False,
                "image_readable": False,
                "width": 0,
                "height": 0,
            }

    def _run_figure_ocr(
        self,
        asset_path: Path | None,
    ) -> dict[str, Any]:
        """Run confidence-aware OCR on a true figure candidate.

        Args:
            asset_path (Path | None): Figure image path.

        Returns:
            dict[str, Any]: OCR text, tokens, numbers, and quality.
        """
        empty_result = {
            "ocr_attempted": False,
            "ocr_text": "",
            "ocr_tokens": [],
            "numeric_tokens": [],
            "median_confidence": None,
            "ocr_quality": "not_attempted",
            "requires_visual_check": True,
        }

        if (
            not self.config.enable_figure_ocr
            or asset_path is None
            or not asset_path.is_file()
        ):
            return empty_result

        try:
            with Image.open(asset_path) as source:
                image = source.convert("RGB")

            image = ImageOps.grayscale(image)
            image = ImageOps.autocontrast(image)

            image = image.resize(
                (
                    image.width * 2,
                    image.height * 2,
                ),
                Image.Resampling.LANCZOS,
            )

            data = pytesseract.image_to_data(
                image,
                config="--psm 11",
                output_type=pytesseract.Output.DICT,
            )

        except (
            OSError,
            pytesseract.TesseractError,
        ):
            return {
                **empty_result,
                "ocr_attempted": True,
                "ocr_quality": "ocr_error",
            }

        tokens: list[dict[str, Any]] = []
        accepted_text: list[str] = []
        confidences: list[float] = []

        for index, raw_text in enumerate(
            data.get("text", [])
        ):
            text = normalize_text(raw_text)

            if not text:
                continue

            try:
                confidence = float(
                    data["conf"][index]
                )
            except (
                KeyError,
                TypeError,
                ValueError,
            ):
                confidence = -1.0

            token = {
                "text": text,
                "confidence": confidence,
                "left": int(
                    data["left"][index]
                ),
                "top": int(
                    data["top"][index]
                ),
                "width": int(
                    data["width"][index]
                ),
                "height": int(
                    data["height"][index]
                ),
            }

            tokens.append(token)

            if confidence >= MINIMUM_OCR_CONFIDENCE:
                accepted_text.append(text)
                confidences.append(confidence)

        ocr_text = normalize_text(
            " ".join(accepted_text)
        )

        numeric_tokens = re.findall(
            r"\b\d[\d,]*(?:\.\d+)?%?\b",
            ocr_text,
        )

        median_confidence = (
            float(median(confidences))
            if confidences
            else None
        )

        if not ocr_text:
            ocr_quality = "no_meaningful_text"
        elif (
            median_confidence is not None
            and median_confidence < 50.0
        ):
            ocr_quality = "low_confidence"
        else:
            ocr_quality = "pass"

        return {
            "ocr_attempted": True,
            "ocr_text": ocr_text,
            "ocr_tokens": tokens,
            "numeric_tokens": numeric_tokens,
            "median_confidence": median_confidence,
            "ocr_quality": ocr_quality,
            "requires_visual_check": (
                ocr_quality != "pass"
            ),
        }

    @staticmethod
    def _find_overlapping_table(
        picture: Any,
        tables: list[Any],
    ) -> int | None:
        """Find a table occupying the same page region.

        Args:
            picture (Any): Docling PictureItem.
            tables (list[Any]): Docling TableItems.

        Returns:
            int | None: Matching table index.
        """
        picture_pages = set(item_pages(picture))
        picture_bbox = (
            FigureEvidenceBuilder._bbox(picture)
        )

        if not picture_pages or picture_bbox is None:
            return None

        best_index: int | None = None
        best_overlap = 0.0

        for table_index, table in enumerate(tables):
            if not picture_pages.intersection(
                item_pages(table)
            ):
                continue

            table_bbox = (
                FigureEvidenceBuilder._bbox(table)
            )

            if table_bbox is None:
                continue

            overlap = (
                FigureEvidenceBuilder._overlap_ratio(
                    picture_bbox,
                    table_bbox,
                )
            )

            if overlap > best_overlap:
                best_overlap = overlap
                best_index = table_index

        if best_overlap >= MINIMUM_TABLE_OVERLAP:
            return best_index

        return None

    @staticmethod
    def _bbox(
        item: Any,
    ) -> tuple[float, float, float, float] | None:
        """Read the first provenance bounding box.

        Args:
            item (Any): Docling document item.

        Returns:
            tuple | None: Left, bottom, right, top coordinates.
        """
        provenance = getattr(item, "prov", []) or []

        if not provenance:
            return None

        bbox = getattr(
            provenance[0],
            "bbox",
            None,
        )

        if bbox is None:
            return None

        try:
            left = float(getattr(bbox, "l"))
            bottom = float(getattr(bbox, "b"))
            right = float(getattr(bbox, "r"))
            top = float(getattr(bbox, "t"))
        except (
            AttributeError,
            TypeError,
            ValueError,
        ):
            return None

        return left, bottom, right, top

    @staticmethod
    def _overlap_ratio(
        first: tuple[float, float, float, float],
        second: tuple[float, float, float, float],
    ) -> float:
        """Calculate overlap relative to the smaller region.

        Args:
            first (tuple): First bounding box.
            second (tuple): Second bounding box.

        Returns:
            float: Overlap ratio from zero to one.
        """
        first_left, first_bottom, first_right, first_top = (
            first
        )

        second_left, second_bottom, second_right, second_top = (
            second
        )

        intersection_width = max(
            0.0,
            min(first_right, second_right)
            - max(first_left, second_left),
        )

        intersection_height = max(
            0.0,
            min(first_top, second_top)
            - max(first_bottom, second_bottom),
        )

        intersection_area = (
            intersection_width
            * intersection_height
        )

        first_area = max(
            0.0,
            first_right - first_left,
        ) * max(
            0.0,
            first_top - first_bottom,
        )

        second_area = max(
            0.0,
            second_right - second_left,
        ) * max(
            0.0,
            second_top - second_bottom,
        )

        smaller_area = min(
            first_area,
            second_area,
        )

        if smaller_area <= 0:
            return 0.0

        return intersection_area / smaller_area

    @staticmethod
    def _find_table_by_ocr_text(
        ocr_text: str,
        table_records: list[dict[str, Any]],
        picture_pages: list[int],
    ) -> int | None:
        """Find a table whose text substantially matches picture OCR.

        This is a fallback when bounding boxes do not overlap
        reliably.

        Args:
            ocr_text (str): OCR text from the picture.
            table_records (list[dict[str, Any]]): Table records.
            picture_pages (list[int]): Picture page numbers.

        Returns:
            int | None: Matching table index.
        """
        ocr_tokens = (
            FigureEvidenceBuilder._comparison_tokens(
                ocr_text
            )
        )

        if not ocr_tokens:
            return None

        picture_page_set = set(picture_pages)
        best_index: int | None = None
        best_score = 0.0

        for table_index, table in enumerate(
            table_records
        ):
            if not picture_page_set.intersection(
                table["page_numbers"]
            ):
                continue

            table_tokens = (
                FigureEvidenceBuilder._comparison_tokens(
                    table["markdown"]
                )
            )

            if not table_tokens:
                continue

            shared_tokens = (
                ocr_tokens.intersection(table_tokens)
            )

            denominator = min(
                len(ocr_tokens),
                len(table_tokens),
            )

            score = (
                len(shared_tokens) / denominator
                if denominator
                else 0.0
            )

            if score > best_score:
                best_score = score
                best_index = table_index

        if best_score >= MINIMUM_TABLE_TEXT_OVERLAP:
            return best_index

        return None

    @staticmethod
    def _comparison_tokens(text: str) -> set[str]:
        """Create normalized tokens for table-image comparison.

        Args:
            text (str): OCR or Markdown text.

        Returns:
            set[str]: Meaningful lowercase tokens.
        """
        return {
            token.lower()
            for token in re.findall(
                r"[A-Za-z0-9.-]+",
                text,
            )
            if len(token) >= 3
        }

    @staticmethod
    def _attach_asset_to_table(
        table_record: dict[str, Any],
        asset_path: Path | None,
    ) -> None:
        """Attach a picture asset to its structured table.

        Args:
            table_record (dict[str, Any]): Table to update.
            asset_path (Path | None): Visual table image.
        """
        if (
            asset_path is not None
            and asset_path.is_file()
            and not table_record.get("asset_path")
        ):
            table_record["asset_path"] = str(
                asset_path
            )

        table_record["has_picture_visual"] = True

    @staticmethod
    def _build_context_map(
        document: Any,
    ) -> dict[str, dict[str, str]]:
        """Find the heading and text preceding each picture.

        Args:
            document (Any): Loaded DoclingDocument.

        Returns:
            dict[str, dict[str, str]]: Picture context.
        """
        context: dict[str, dict[str, str]] = {}
        current_section = ""
        recent_text: list[str] = []

        for item, _level in document.iterate_items():
            label = label_value(item)
            text = normalize_text(
                getattr(item, "text", "")
            )

            if label in {
                "section_header",
                "title",
            } and text:
                current_section = text
                recent_text = []
                continue

            if label in {
                "text",
                "list_item",
                "footnote",
            } and text:
                recent_text.append(text)
                recent_text = recent_text[-2:]
                continue

            if label in {"picture", "chart"}:
                reference = source_reference(item)

                if reference:
                    context[reference] = {
                        "section": current_section,
                        "nearby_text": " ".join(
                            recent_text
                        ),
                    }

        return context

    @staticmethod
    def _caption_text(
        picture: Any,
        document: Any,
    ) -> str:
        """Read a picture caption.

        Args:
            picture (Any): Docling PictureItem.
            document (Any): Parent DoclingDocument.

        Returns:
            str: Caption text.
        """
        caption_method = getattr(
            picture,
            "caption_text",
            None,
        )

        if callable(caption_method):
            return normalize_text(
                caption_method(document)
            )

        return ""

    @staticmethod
    def _printed_number(
        caption: str,
        fallback: int,
    ) -> int:
        """Read a printed figure number.

        Args:
            caption (str): Figure caption.
            fallback (int): Picture order.

        Returns:
            int: Figure number.
        """
        match = re.search(
            r"\bfigure\s+(\d+)\b",
            caption,
            flags=re.IGNORECASE,
        )

        return (
            int(match.group(1))
            if match
            else fallback
        )

    @staticmethod
    def _search_text(
        document_title: str,
        section: str,
        caption: str,
        nearby_text: str,
        ocr_text: str,
    ) -> str:
        """Construct searchable figure evidence.

        Args:
            document_title (str): Source title.
            section (str): Nearby heading.
            caption (str): Figure caption.
            nearby_text (str): Nearby prose.
            ocr_text (str): Text inside the image.

        Returns:
            str: Search-ready figure text.
        """
        parts = [
            f"Document: {document_title}"
        ]

        if section:
            parts.append(
                f"Section: {section}"
            )

        if caption:
            parts.append(
                f"Figure caption: {caption}"
            )

        if nearby_text:
            parts.append(
                f"Nearby text: {nearby_text}"
            )

        if ocr_text:
            parts.append(
                f"Text inside figure: {ocr_text}"
            )

        return "\n".join(parts)