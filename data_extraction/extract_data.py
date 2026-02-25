import re
import csv
import sys
import argparse
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

try:
    from .classes import TransitionExtractor, build_transition_extractors
    from .classes.base import TransitionSpec
except ImportError:
    from classes import TransitionExtractor, build_transition_extractors
    from classes.base import TransitionSpec

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
class CsvOutput:
    file_handle: object
    writer: csv.writer
    expected_window_len: int


GENE_LINE_REGEX = re.compile(r"^\(\[([^\]]+)\],\[(\d+)\],\[(\d+)\],\[([A-Za-z]+)\],\[([^\]]+)\],\[(\d+)\],\[(\d+)\],(true|false)\)$")
EXON_LINE_REGEX = re.compile(r"\[(\d+),(\d+)\]")

DEFAULT_FLANK_SIZE = 1000


def parse_gene_line(line) -> GeneRecord | None:
    match = GENE_LINE_REGEX.match(line)
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
    matches = EXON_LINE_REGEX.findall(line)
    return [(int(start), int(end)) for start, end in matches]


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


def local_to_sequence_index(gene: GeneRecord, local_position: int, flank_size: int) -> int:
    return local_position - gene.local_start + flank_size


def extract_window(gene: GeneRecord, start_local: int, end_local: int, flank_size:int) -> str | None:
    if start_local > end_local:
        return None

    start_index = local_to_sequence_index(gene, start_local, flank_size)
    end_index = local_to_sequence_index(gene, end_local, flank_size)

    if start_index < 0 or end_index >= len(gene.sequence):
        return None
    
    return gene.sequence[start_index:end_index + 1]


def open_csv_output(output_path: Path, local_position_column: str, window_len: int) -> CsvOutput:
    handle=output_path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(handle)
    header =(["gene_id", "chromosome", "global_position", local_position_column]+
                     [f"B{i}" for i in range(1,window_len+1)]+["label"])

    writer.writerow(header)
    return CsvOutput(file_handle=handle, writer=writer, expected_window_len=window_len) 


def write_csv_row(output: CsvOutput, gene: GeneRecord, local_position: int, sequence: str, label: bool) -> bool:

    if len(sequence) != output.expected_window_len:
        return False

    output.writer.writerow([
        gene.gene_id,
        gene.chromosome,
        local_to_global(gene, local_position),
        local_position,
        *sequence,
        label
    ])
    return True


def is_far_from_position(position: int, positions: list[int], min_distance: int) -> bool:
    return all(abs(position - existing_position) >= min_distance for existing_position in positions)


def sample_negative_example(
    gene: GeneRecord, 
    spec: TransitionSpec, 
    anchor_position: int, 
    true_positions: list[int],
    used_positions: set[int],
    flank_size: int,
    rng: random.Random,) -> tuple[int,str] | None:
    
    shifted_candidates = list(spec.shift_candidates)
    rng.shuffle(shifted_candidates)

    for shift in shifted_candidates:
        candidate_position = anchor_position + shift

        if candidate_position in used_positions:
            continue

        if not is_far_from_position(candidate_position, true_positions, spec.min_distance_from_positive):
            continue

        candidate_sequence = extract_window(gene, candidate_position + spec.window_start_offset, candidate_position + spec.window_end_offset, flank_size)

        if candidate_sequence and len(candidate_sequence) == spec.window_len:
            return candidate_position, candidate_sequence
        

    min_candidate = gene.local_start - flank_size - spec.window_start_offset
    max_candidate = gene.local_end + flank_size - spec.window_end_offset
    if min_candidate > max_candidate:
        return None

    for _ in range(200):
        candidate_position = rng.randint(min_candidate, max_candidate)
        if candidate_position in used_positions:
            continue
        if not is_far_from_position(candidate_position, true_positions, spec.min_distance_from_positive):
            continue

        candidate_sequence = extract_window(
            gene,
            candidate_position + spec.window_start_offset,
            candidate_position + spec.window_end_offset,
            flank_size,
        )
        if candidate_sequence and len(candidate_sequence) == spec.window_len:
            return candidate_position, candidate_sequence

    return None


def process_gene(gene: GeneRecord, output: CsvOutput,transition_extractors: dict[str, TransitionExtractor], flank_size: int, negatives_per_positive: int, rng: random.Random) -> None:
    
    positive_positions: dict[str, list[int]] = {key: [] for key in transition_extractors}

    for transcript_exons in gene.transcripts:

        ordered_exons = sorted(transcript_exons, key=lambda exon: exon[0])

        if not ordered_exons:
            continue

        for transition_key, extractor in transition_extractors.items():
            spec = extractor.spec
            for anchor_position in extractor.iter_anchor_positions(ordered_exons):
                positive_sequence = extract_window(
                    gene,
                    anchor_position + spec.window_start_offset,
                    anchor_position + spec.window_end_offset,
                    flank_size,
                )
                if positive_sequence and write_csv_row(output[transition_key], gene, anchor_position, positive_sequence, True):
                    positive_positions[transition_key].append(anchor_position)
    
    if negatives_per_positive <= 0:
        return

    for transition_key, anchor_positions in positive_positions.items():
        if not anchor_positions:
            continue

        spec = transition_extractors[transition_key].spec
        true_positions = sorted(set(anchor_positions))
        used_positions = set(true_positions)

        for anchor_position in anchor_positions:
            for _ in range(negatives_per_positive):
                negative_example = sample_negative_example(
                    gene=gene,
                    spec=spec,
                    anchor_position=anchor_position,
                    true_positions=true_positions,
                    used_positions=used_positions,
                    flank_size=flank_size,
                    rng=rng,
                )
                if not negative_example:
                    continue

                negative_position, negative_sequence = negative_example
                if write_csv_row(output[transition_key], gene, negative_position, negative_sequence, False):
                    used_positions.add(negative_position)

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
        default=project_root / "data",
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
    parser.add_argument(
        "--flank-size",
        type=int,
        default=DEFAULT_FLANK_SIZE,
        help=(
            "Number of context bases available at each side of a gene sequence "
            f"(default: {DEFAULT_FLANK_SIZE})."
        ),
    )
    parser.add_argument(
        "--negatives-per-positive",
        type=int,
        default=1,
        help=(
            "Number of negative examples generated per positive transition "
            "(default: %(default)s). Set to 0 to disable negatives."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed for reproducible negative sampling (default: %(default)s).",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    input_files = resolve_input_files(input_dir, args.input_file)
    flank_size: int = args.flank_size
    negatives_per_positive: int = getattr(args, "negatives_per_positive", 1)
    rng = random.Random(getattr(args, "random_seed", 42))

    if not input_files:
        print(f"No input files found in: {input_dir}", file=sys.stderr)
        return 1

    missing_files = [str(path) for path in input_files if not path.exists()]
    if missing_files:
        print("These input files do not exist:", file=sys.stderr)
        for item in missing_files:
            print(f"  - {item}", file=sys.stderr)
        return 1
    
    if negatives_per_positive < 0:
        print("--negatives-per-positive must be >= 0", file=sys.stderr)
        return 1
    
    transition_extractors= build_transition_extractors()

    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        key: open_csv_output(
            output_dir / f"data_{key}.csv",
            extractor.spec.local_position_column,
            extractor.spec.window_len,
        )
        for key,extractor in transition_extractors.items()
    }

    try:
        for input_file in input_files:
            for gene in iter_gene_records(input_file):
                process_gene(gene, outputs, transition_extractors, flank_size, negatives_per_positive, rng)
    finally:
        for output in outputs.values():
            output.file_handle.close()

    print(f"Processed files: {len(input_files)}")

    return 0


def main() -> int:
    args = parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())