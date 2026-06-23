# TB sample corpus — sources & provenance

A tuberculosis-domain document set for exercising the RAG system (vector / pageindex /
hybrid). 14 documents, ~660 pages, all machine-readable text (no scanned/OCR-only files).
Gathered 2026-06-20.

**What's in this repo:** only the US-Government **public-domain** documents (the FDA drug
labels and the CDC Core Curriculum). The **NWT (Government of Northwest Territories)**
files below are Crown-copyright and marked *not for redistribution*, so they are
gitignored — download them from the URLs in this file into `data/kb/tb/` to get the full set.

## Government of Northwest Territories — HSS (Canada)  *(not included in this repo)*
Crown-copyright clinical resources, freely published for health professionals (use here for
demo/research, not redistribution). These files are gitignored — fetch them from the URLs
below into `data/kb/tb/`.

| File | Pages | Document | Source |
|---|---|---|---|
| `nwt-tb-cdm-manual.pdf` | 34 | Communicable Disease Manual — Tuberculosis chapter (2023); numbered sections 1.x/2.x | https://www.hss.gov.nt.ca/professionals/sites/professionals/files/resources/tuberculosis-manual.pdf |
| `nwt-tb-section8-adverse.pdf` | 5 | TB Section 8 — Active TB: Adverse Reactions / Treatment-Induced Side Effects (2014) *(the original example)* | https://www.hss.gov.nt.ca/professionals/sites/professionals/files/resources/tb-section-8-active-tb-adverse-reactions-treatment-induced-side-effects.pdf |
| `nwt-tb-program-standards.pdf` | 14 | NWT TB Program Standards | https://www.hss.gov.nt.ca/professionals/sites/professionals/files/resources/nwt-tb-program-standards.pdf |
| `nwt-cpi-114-tb-protocol.pdf` | 236 | CPI-114 — Revised NWT TB Protocol (large) | https://www.hss.gov.nt.ca/professionals/sites/professionals/files/resources/cpi-114-revised-nwt-tb-protocol.pdf |
| `nwt-cpi-179-tb-program.pdf` | 2 | Clinical Practice Information Notice CPI-179 — NWT TB Program | https://www.hss.gov.nt.ca/professionals/sites/professionals/files/resources/cpi-179-nwt-tb-program.pdf |
| `nwt-physician-std-practice.pdf` | 22 | NWT Standards of Practice for Physicians | https://www.hss.gov.nt.ca/sites/hss/files/resources/nwt-physician-standard-practice.pdf |
| `nwt-tb-assessment-form.pdf` | 5 | TB Assessment Form | https://www.hss.gov.nt.ca/professionals/sites/professionals/files/resources/tb-assessment-form.pdf |

## FDA drug labels (accessdata.fda.gov) — US Government work, public domain
Structured "Highlights of Prescribing Information" (Indications, Dosage, Warnings, **Adverse
Reactions**, Drug Interactions) — the closest analog to the original Section 8 example.

| File | Pages | Drug | Source |
|---|---|---|---|
| `fda-isoniazid-label.pdf` | 13 | Isoniazid (first-line) | https://www.accessdata.fda.gov/drugsatfda_docs/label/2025/008678s030lbl.pdf |
| `fda-rifadin-rifampin-label.pdf` | 21 | RIFADIN (rifampin) | https://www.accessdata.fda.gov/drugsatfda_docs/label/2023/050420s089,050627s034lbl.pdf |
| `fda-rifater-label.pdf` | 28 | RIFATER (rifampin + isoniazid + pyrazinamide) | https://www.accessdata.fda.gov/drugsatfda_docs/label/2023/050705s022lbl.pdf |
| `fda-ethambutol-myambutol-label.pdf` | 5 | MYAMBUTOL (ethambutol) | https://www.accessdata.fda.gov/drugsatfda_docs/label/2008/016320s063lbl.pdf |
| `fda-rifapentine-priftin-label.pdf` | 30 | PRIFTIN (rifapentine) | https://www.accessdata.fda.gov/drugsatfda_docs/label/2020/021024s017s018lbl.pdf |
| `fda-bedaquiline-sirturo-label.pdf` | 34 | SIRTURO (bedaquiline; drug-resistant TB) | https://www.accessdata.fda.gov/drugsatfda_docs/label/2024/204384s019lbl.pdf |

## CDC — US Government work, public domain
| File | Pages | Document | Source |
|---|---|---|---|
| `cdc-core-curriculum-tb.pdf` | 211 | Core Curriculum on Tuberculosis: What the Clinician Should Know (7th ed., 2021) | https://www.cdc.gov/tb/media/Core_Curriculum_TB_eBook.pdf |

## Not included (sites block scripted download — fetch via browser if wanted)
- WHO Consolidated Guidelines on TB, Module 4: Treatment — https://www.who.int/publications/i/item/9789240063129
- Canadian TB Standards, 8th ed. (the standard the NWT now endorses)

## Example questions this corpus answers
- "What are the adverse reactions / treatment-induced side effects of TB drugs?"
- "What is the case definition of active TB vs latent TB infection?"
- "What is DOT (directly observed therapy) and when is it required?"
- "What is the dosing for isoniazid? For the RIFATER combination?"
- "How is bedaquiline used for drug-resistant TB, and what are its warnings?"
- "What are the TB reporting requirements in the NWT?"
