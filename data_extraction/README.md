# Data Extraction

## Extraction Overview

For each gene in the Ensembl data, the extraction process identifies four types of transitions:

1. Exon → Intron (EI)
2. Intron → Exon (IE)
3. Intergenic Zone → First Exon (ZE)
4. Last Exon → Intergenic Zone (EZ)

### Data Estructure

* Gen Information line
```
([GEN_ID],[CORD_INI_GEN],[CORD_FINAL_GEN],[NUCLEOTIDES],[CHROMOSOME_NUMBER],[CORD_INI_GLOBAL_GEN],[CORD_FINAL_GLOBAL_GEN])
```
_**Example**_
```
([ENSG00000278678.1],[600],[895],[....],[21],[10575693],[10577188],false)
```

* Transcription Information Line
```
([EXON_1_START,EXON_1_END][EXON_2_START,EXON_2_END]....[EXON_N_START,EXON_N_END])
```
_**Example**_
```
([1000,1240],[2117,2482],[1])
```
_**Note**: Some genes might not have transcript lines. In such cases, only the gene information line is present._

## Extraction Process
Detailed information about data extraction
## Exon -> Intron (EI)

* **Location**: End position of the first exon (e.g., 1240)
* **Intron Start**: Starts at the position `exon_end + 1` (e.g., 1241)
    * Intron starts with the nucleotides `"gt"`, should be located at positions 1241-1242
* **Data Extraction**
    * Extract 5 characters to the left of the intron starts.
    * Extrtact 5 characters to the right of the intron starts.
    * The result is a substring consisting of a sequence of 12 characters.
* **Save**: Save the result into ``data_ei.csv``. Each character of the sequence is stored in its own column (B1, B2, …, B12).

## Intron -> Exon (IE)

* **Location**: Start position of the second exon (e.g., 2117)
* **Intron End**: Intron end at the position `exon_start - 1` (e.g., 2116)
    * Intron end with the nucleotides `"ag"`, should be located at positions 2115-2116
* **Data Extraction**
    * Extract 100 characters to the left of the exon starts.
    * Extrtact 5 characters to the right of the exon starts.
    * The result is a substring consisting of a sequence of 105 characters.
* **Save**: Save the result into ``data_ie.csv``. Each character of the sequence is stored in its own column (B1, B2, …, B105).

## Intergenic Zone -> First Exon (ZE)

* **Location**: Start position of the first exon (e.g., 1000)
* **Data Extraction**
    * Extract 500 characters to the left of the first exon starts.
    * Extrtact 50 characters to the right of the firts exon starts.
    * The result is a substring consisting of a sequence of 550 characters.
* **Save**: Save the result into ``data_ze.csv``. Each character of the sequence is stored in its own column (B1, B2, …, B550).

## Last Exon -> Intergenic Xone (EZ)

* **Location**: End position of the Last exon (e.g., 2482)
* **Data Extraction**
    * Extract 50 characters to the left of the last exon ends.
    * Extrtact 500 characters to the right of the last exon ends.
    * The result is a substring consisting of a sequence of 550 characters.
* **Save**: Save the result into ``data_ez.csv``. Each character of the sequence is stored in its own column (B1, B2, …, B550).

## Files and Outputs

* data_ei.csv: Contains the EI transition sequences.
* data_ie.csv: Contains the IE transition sequences.
* data_ze.csv: Contains the ZE transition sequences.
* data_ez.csv: Contains the EZ transition sequences.

## Script

You can automate the extraction process with:

```bash
python3 data_extraction/extract_data.py
```

By default, the script reads all `*.txt` files in `data_ensembl` and creates
the four output CSV files in `data_assembled`.

Optional arguments:

```bash
python3 data_extraction/extract_data.py \
  --input-dir data_ensembl \
  --output-dir data_assembled \
  --input-file 3-187668812-187670494.txt
```

_*Example*_

| GEN_ID  | Chromosome|Global_Start|Exon_End|B1|B2|.....|Bn|
| --------|:----------|------------|:-------|:-|:-|:----|:-|
|ENSG00000154646.9|21|18268517|135269| a| g|.....| t|
|ENSG00000154646.9|21|182517|135469| g| g|.....| c|
|ENSG00000154646.9|21|1826851|5269| c| t|.....| t|