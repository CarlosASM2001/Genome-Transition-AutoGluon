#!/usr/bin/env python3
"""Build assembled extraction CSV files from Ensembl text exports.

The script reads gene records from ``data_ensembl/*.txt`` and generates:

- data_ei.csv (Exon -> Intron)
- data_ie.csv (Intron -> Exon)
- data_ze.csv (Intergenic Zone -> First Exon)
- data_ez.csv (Last Exon -> Intergenic Zone)

Each sequence character is written in an independent ``B<n>`` column, as
described in ``data_extraction/README.md``.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


GENE_LINE_RE = re.compile(
    r"^\(\[([^\]]+)\],\[(\d+)\],\[(\d+)\],\[([A-Za-z]+)\],\[([^\]]+)\],\[(\d+)\],\[(\d+)\],(true|false)\)$"
)
EXON_PAIR_RE = re.compile(r"\[(\d+),(\d+)\]")


@dataclass
class GeneRecord:
    gene_id: str
    local_start: int
    local_end: int
    sequence: str
    chromosome: str
    global_start: int
    global_end: int
    reverse_strand: bool
    transcripts: list[list[tuple[int, int]]] = field(default_factory=list)


@dataclass
class Stats:
    genes_seen: int = 0
    genes_length_mismatch: int = 0
    transcripts_seen: int = 0
    transitions_ei_written: int = 0
    transitions_ie_written: int = 0
    transitions_ze_written: int = 0
    transitions_ez_written: int = 0
    transitions_ei_skipped: int = 0
    transitions_ie_skipped: int = 0
    transitions_ze_skipped: int = 0
    transitions_ez_skipped: int = 0


@dataclass
class CsvOutput:
    file_handle: object
    writer: csv.writer
    expected_window_len: int


def parse_gene_line(line: str) -> GeneRecord | None:
    match = GENE_LINE_RE.match(line)
    if not match:
        return None
    return GeneRecord(
        gene_id=match.group(1),
        local_start=int(match.group(2)),
        local_end=int(match.group(3)),
        sequence=match.group(4).lower(),
        chromosome=match.group(5),
        global_start=int(match.group(6)),
        global_end=int(match.group(7)),
        reverse_strand=match.group(8) == "true",
    )


def parse_transcript_line(line: str) -> list[tuple[int, int]]:
    return [(int(start), int(end)) for start, end in EXON_PAIR_RE.findall(line)]


def iter_gene_records(file_path: Path) -> Iterator[GeneRecord]:
    current_gene: GeneRecord | None = None
    with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            parsed_gene = parse_gene_line(line)
            if parsed_gene:
                if current_gene is not None:
                    yield current_gene
                current_gene = parsed_gene
                continue

            if current_gene is None:
                continue

            exons = parse_transcript_line(line)
            if exons:
                current_gene.transcripts.append(exons)

    if current_gene is not None:
        yield current_gene


def local_to_global(gene: GeneRecord, local_position: int) -> int:
    return gene.global_start + (local_position - gene.local_start)


def extract_window(gene: GeneRecord, start_local: int, end_local: int) -> str | None:
    if start_local > end_local:
        return None
    start_idx = start_local - gene.local_start
    end_idx = end_local - gene.local_start
    if start_idx < 0 or end_idx >= len(gene.sequence):
        return None
    return gene.sequence[start_idx : end_idx + 1]


def open_csv_output(
    out_path: Path,
    local_position_column: str,
    window_len: int,
) -> CsvOutput:
    handle = out_path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(handle)
    header = (
        ["GEN_ID", "Chromosome", "Global_Position", local_position_column]
        + [f"B{i}" for i in range(1, window_len + 1)]
    )
    writer.writerow(header)
    return CsvOutput(file_handle=handle, writer=writer, expected_window_len=window_len)


def write_row(csv_output: CsvOutput, gene: GeneRecord, local_position: int, sequence: str) -> bool:
    if len(sequence) != csv_output.expected_window_len:
        return False
    csv_output.writer.writerow(
        [gene.gene_id, gene.chromosome, local_to_global(gene, local_position), local_position, *sequence]
    )
    return True


def process_gene(gene: GeneRecord, outputs: dict[str, CsvOutput], stats: Stats) -> None:
    stats.genes_seen += 1
    expected_len = gene.local_end - gene.local_start + 1
    if expected_len != len(gene.sequence):
        stats.genes_length_mismatch += 1

    for transcript_exons in gene.transcripts:
        if not transcript_exons:
            continue
        stats.transcripts_seen += 1

        # Some records can include exons in reverse genomic order. We normalize
        # to left-to-right coordinates to follow the README extraction rules.
        ordered_exons = sorted(transcript_exons, key=lambda exon: exon[0])

        first_exon_start = ordered_exons[0][0]
        ze_sequence = extract_window(gene, first_exon_start - 500, first_exon_start + 49)
        if ze_sequence and write_row(outputs["ze"], gene, first_exon_start, ze_sequence):
            stats.transitions_ze_written += 1
        else:
            stats.transitions_ze_skipped += 1

        last_exon_end = ordered_exons[-1][1]
        ez_sequence = extract_window(gene, last_exon_end - 49, last_exon_end + 500)
        if ez_sequence and write_row(outputs["ez"], gene, last_exon_end, ez_sequence):
            stats.transitions_ez_written += 1
        else:
            stats.transitions_ez_skipped += 1

        for exon_left, exon_right in zip(ordered_exons, ordered_exons[1:]):
            exon_end = exon_left[1]
            intron_start = exon_end + 1
            ei_sequence = extract_window(gene, intron_start - 5, intron_start + 6)
            if ei_sequence and write_row(outputs["ei"], gene, intron_start, ei_sequence):
                stats.transitions_ei_written += 1
            else:
                stats.transitions_ei_skipped += 1

            exon_start = exon_right[0]
            ie_sequence = extract_window(gene, exon_start - 100, exon_start + 4)
            if ie_sequence and write_row(outputs["ie"], gene, exon_start, ie_sequence):
                stats.transitions_ie_written += 1
            else:
                stats.transitions_ie_skipped += 1


def resolve_input_files(input_dir: Path, explicit_files: list[str]) -> list[Path]:
    if explicit_files:
        files: list[Path] = []
        for input_file in explicit_files:
            resolved = Path(input_file)
            if not resolved.is_absolute():
                resolved = input_dir / input_file
            files.append(resolved)
        return sorted(files)

    return sorted(input_dir.glob("*.txt"))


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Generate EI/IE/ZE/EZ extraction CSV files from Ensembl text inputs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=project_root / "data_ensembl",
        help="Directory with source .txt files (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data_assembled",
        help="Directory where output CSV files will be written (default: %(default)s)",
    )
    parser.add_argument(
        "--input-file",
        action="append",
        default=[],
        help=(
            "Optional specific file to process. Use multiple times for multiple files. "
            "If relative, it is resolved against --input-dir."
        ),
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    input_files = resolve_input_files(input_dir, args.input_file)

    if not input_files:
        print(f"No input files found in: {input_dir}", file=sys.stderr)
        return 1

    missing_files = [str(path) for path in input_files if not path.exists()]
    if missing_files:
        print("These input files do not exist:", file=sys.stderr)
        for item in missing_files:
            print(f"  - {item}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "ei": open_csv_output(output_dir / "data_ei.csv", "Intron_Start", 12),
        "ie": open_csv_output(output_dir / "data_ie.csv", "Exon_Start", 105),
        "ze": open_csv_output(output_dir / "data_ze.csv", "First_Exon_Start", 550),
        "ez": open_csv_output(output_dir / "data_ez.csv", "Last_Exon_End", 550),
    }

    stats = Stats()
    try:
        for input_file in input_files:
            for gene in iter_gene_records(input_file):
                process_gene(gene, outputs, stats)
    finally:
        for output in outputs.values():
            output.file_handle.close()

    print(f"Processed files: {len(input_files)}")
    print(f"Genes processed: {stats.genes_seen}")
    print(f"Genes with sequence length mismatch: {stats.genes_length_mismatch}")
    print(f"Transcripts processed: {stats.transcripts_seen}")
    print(
        "EI transitions: "
        f"{stats.transitions_ei_written} written, {stats.transitions_ei_skipped} skipped"
    )
    print(
        "IE transitions: "
        f"{stats.transitions_ie_written} written, {stats.transitions_ie_skipped} skipped"
    )
    print(
        "ZE transitions: "
        f"{stats.transitions_ze_written} written, {stats.transitions_ze_skipped} skipped"
    )
    print(
        "EZ transitions: "
        f"{stats.transitions_ez_written} written, {stats.transitions_ez_skipped} skipped"
    )

    return 0


def main() -> int:
    args = parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
