# Example queries and results

Twelve devices from the test split (cleared 2025 to 2026), with the top 5 results of four methods. Dataset: predx build of 2026-11-22, 52,123 corpus devices.

The examples are a random draw (seed 0) inside four groups, so they illustrate behaviour and are not a representative sample. Share of all 5,308 test queries in each group:

- Fine-tuned search finds a cited predicate in the top 10; TF-IDF does not: 6.6%
- Both find a cited predicate in the top 10: 81.9%
- TF-IDF finds a cited predicate in the top 10; fine-tuned search does not: 1.2%
- Neither finds a cited predicate in the top 10: 10.3%

A result marked **CITED** is a predicate named in the device's own 510(k) summary. Query text is shown as the models see it: K-numbers are masked as `[K-NUMBER]` and names of cited predicates as `[DEVICE]`. Long text is cut here for readability.

## Fine-tuned search finds a cited predicate in the top 10; TF-IDF does not

### Example 1: [K241333](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID=K241333) Geniant Cranial (Navigated Neurosurgical Positioning Robot)

Cleared 2025-01-17 · product code HAW · panel NE · 46,567 earlier candidates

> Indications for Use: The device is intended to aid in locating anatomical structures and for the spatial positioning and orientation of instrument holders or tool guides to be used by neurosurgeons for navigating and guiding compatible surgical instruments in open or percutaneous procedures. The device is indicated for any neurosurgical procedure in which stereotactic neurosurgery may be appropriate and where reference to a rigid anatomical structure, such as the skull, can be identified relative to anatomy images.
>
> Device Description: Geniant Cranial is a hardware platform that supports real-time surgical navigation using medical patient images. The application software reprocesses the CT or MR images of the patient acquired before surgery. It displays the contents on the software screen in various fluoroscopy directions (axial, sagittal and coronal). Before surgery, the surgeon can ...

**Cited predicates in the corpus:** [K200511](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID=K200511) ROSA ONE Brain application (primary); [K162309](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID=K162309) StealthStation S8 System Platforms and StealthStation Cranial Software

| Method | Rank of best cited predicate | Top 5 |
| --- | --- | --- |
| Fine-tuned + TF-IDF | 6 | 1. K133444 STEALTHSTATION S7 / I7 / FUSION NAVIGATION ...<br>2. K191597 Stealth Autoguide System, Midas Rex Legend ...<br>3. K200047 ExcelsiusGPS Cranial 1.0 Module<br>4. K201175 StealthStation Cranial Software v1.3.0<br>5. K220072 Sinobot X1 |
| Fine-tuned EmbeddingGemma | 8 | 1. K192703 Cranial Image Guided Surgery System, ...<br>2. K132755 NEUROMATE FRAMELESS GEN II<br>3. K212397 StealthStation S8 Cranial v2.0<br>4. K201175 StealthStation Cranial Software v1.3.0<br>5. K203639 StealthStation Cranial Software v1.3.2 |
| TF-IDF | 18 | 1. K133444 STEALTHSTATION S7 / I7 / FUSION NAVIGATION ...<br>2. K153247 Fusion Compact Navigation System<br>3. K200047 ExcelsiusGPS Cranial 1.0 Module<br>4. K211616 ExcelsiusHub<br>5. K200723 StealthStation FlexENT  (9736242), ... |
| EmbeddingGemma, untuned | 15 | 1. K213989 Cranial EM System<br>2. K192703 Cranial Image Guided Surgery System, ...<br>3. K082060 CRANIAL IMAGE GUIDED SURGERY SYSTEM<br>4. K200047 ExcelsiusGPS Cranial 1.0 Module<br>5. K092467 VECTORVISION CRANIAL, VECTORVISION ENT, ... |

### Example 2: [K243570](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID=K243570) Dimension LOCI Thyroid Stimulating Hormone Flex reagent cartridge (TSHL); Dimension LOCI Free Thyroxine Flex reagent cartridge (FT4L)

Cleared 2025-04-25 · product code JLW · panel CH · 47,338 earlier candidates

> Indications for Use: The TSHL method is an in vitro diagnostic test for the quantitative measurement of Thyroid Stimulating Hormone (TSH, thyrotropin) in human serum and plasma on the Dimension® EXLTM integrated chemistry system with LOCI® Module. Measurements of TSH are used in the diagnosis and monitoring of thyroid disease. The FT4L method is an in vitro diagnostic test for the quantitative measurement of Free Thyroxine in human serum and plasma on the Dimension® EXLTM integrated chemistry system with LOCI® module. Measurements of free thyroxine are used in the diagnosis and monitoring of thyroid disease.
>
> Device Description: The Dimension® LOCI® Thyroid Stimulating Hormone Flex® reagent cartridge (TSHL) and Dimension® LOCI® Free Thyroxine Flex® reagent cartridge (FT4L) assays were cleared under [K-NUMBER] and [K-NUMBER], respectively. The components of the cleared assays were ...

**Cited predicates in the corpus:** [K081074](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID=K081074) DIMENSION TSHL FLEX REAGENT CARTRIDGE AND TSH SAMPLE DILUENT WITH MODELS RF 612 AND KD691 (primary); [K073604](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID=K073604) DIMENSION EXL WITH LM CLINICAL CHEMISTRY SYSTEM, DIMENSION FT4L FLEX REAGENT CARTRIDGE, AND LOCI THYROID CALIBRATOR, MOD

| Method | Rank of best cited predicate | Top 5 |
| --- | --- | --- |
| Fine-tuned + TF-IDF | 4 | 1. K140842 DIMENSION LOCI FREE THYROXINE FLEX REAGENT ...<br>2. K140859 DIMENSION VISTA FREE THYROXINE FLEX REAGENT ...<br>3. K100778 DIMENSION EXL LOCI THYROID CALIBRATOR, MODEL ...<br>4. **K073604 DIMENSION EXL WITH LM CLINICAL CHEMISTRY ... CITED**<br>5. **K081074 DIMENSION TSHL FLEX REAGENT CARTRIDGE AND ... CITED** |
| Fine-tuned EmbeddingGemma | 2 | 1. K140842 DIMENSION LOCI FREE THYROXINE FLEX REAGENT ...<br>2. **K073604 DIMENSION EXL WITH LM CLINICAL CHEMISTRY ... CITED**<br>3. K140859 DIMENSION VISTA FREE THYROXINE FLEX REAGENT ...<br>4. K100778 DIMENSION EXL LOCI THYROID CALIBRATOR, MODEL ...<br>5. **K081074 DIMENSION TSHL FLEX REAGENT CARTRIDGE AND ... CITED** |
| TF-IDF | 14 | 1. K093441 DIMENSION VISTA LOCI DIGOXIN FLEX REAGENT ...<br>2. K100344 DIMENSION VISTA LOCI CA 15-3 FLEX REAGENT ...<br>3. K100375 DIMENSION VISTA LOCI CA 19-9 FLEX REAGENT ...<br>4. K190675 Dimension EXL High-Sensitivity Troponin I ...<br>5. K182225 Dimension Vista High-Sensitivity Troponin I ... |
| EmbeddingGemma, untuned | 3 | 1. K140842 DIMENSION LOCI FREE THYROXINE FLEX REAGENT ...<br>2. K140859 DIMENSION VISTA FREE THYROXINE FLEX REAGENT ...<br>3. **K073604 DIMENSION EXL WITH LM CLINICAL CHEMISTRY ... CITED**<br>4. **K081074 DIMENSION TSHL FLEX REAGENT CARTRIDGE AND ... CITED**<br>5. K100778 DIMENSION EXL LOCI THYROID CALIBRATOR, MODEL ... |

### Example 3: [K251377](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID=K251377) Magic InBra

Cleared 2025-09-18 · product code HGX · panel OB · 48,582 earlier candidates

> Indications for Use: The Magic InBraTM breast pump is a powered breast pump to be used by lactating women to express and collect milk from their breasts. The Magic InBraTM powered breast pump is intended for a single user. The breast pump is intended to be used in a home environment."
>
> Device Description: The Magic InBra wearable breast pump is a single user pump to be used by lactating women to express and collect milk from their breasts in the home environment. Magic InBra is designed to fit most breast shapes and nursing bras allowing handsfree pumping. The pump is intended to be used by a single user for pumping on one [K-NUMBER] breast with one Magic InBra breast pump or on both breasts at the same time with two Magic InBra breast pumps. An optional smartphone application (app) for iPhone and Android, called “Medela Family Pump Control”, connects to the breast pump via Bluetooth ...

**Cited predicates in the corpus:** [K200508](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID=K200508) Pump In Style (primary)

| Method | Rank of best cited predicate | Top 5 |
| --- | --- | --- |
| Fine-tuned + TF-IDF | 4 | 1. K213311 Lucy Breast Pump<br>2. K220926 Spectra Wearable<br>3. K191577 Willow Wearable Breast Pump 2.0<br>4. **K200508 Pump In Style CITED**<br>5. K191653 Freestyle Flex |
| Fine-tuned EmbeddingGemma | 1 | 1. **K200508 Pump In Style CITED**<br>2. K150499 Freestyle Deluxe, Freestyle Solution Set, ...<br>3. K031614 MEDELA PUMP IN STYLE ADVANCED BREAST PUMP<br>4. K181937 Pump In Style Advanced, Advanced Personal ...<br>5. K191653 Freestyle Flex |
| TF-IDF | 24 | 1. K081932 ENDEARE BREAST PUMP<br>2. K213311 Lucy Breast Pump<br>3. K102516 LUCINA MODEL MELODI<br>4. K220926 Spectra Wearable<br>5. K230102 Smart double sides wearable breast pump, ... |
| EmbeddingGemma, untuned | 4 | 1. K232608 Duo Hands-free Breast Pump<br>2. K221598 Wearable breast pump<br>3. K241004 Wearable Breast Pump (ABP-1508PRO, ABP-1506, ...<br>4. **K200508 Pump In Style CITED**<br>5. K031614 MEDELA PUMP IN STYLE ADVANCED BREAST PUMP |

### Example 4: [K242056](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID=K242056) miacare (DELiGHT daily wear/ DELiGHT 1-day/ CONFiDENCE Color daily wear/CONFiDENCE Color 1-day) Contact Lens with EautraSil Plus

Cleared 2025-04-17 · product code LPL · panel OP · 47,275 earlier candidates

> Indications for Use: miacare (miafilcon A) DELiGHT Contact Lens with EautraSil Plus is indicated as daily wear soft contact lens for correcting refractive ametropia (myopia and hyperopia) in phakic or aphakic persons with non-diseased eyes who have refractive astigmatism of 1.00D or less, as long as the astigmatism does not impact visual acuity. Eye care professional may prescribe the lens for single-use disposable wear, or for daily wear with frequent replacement. When prescribed for frequent replacement, the lens may be disinfected using a chemical disinfection system only. miacare (miafilcon A) CONFiDENCE Color Contact Lens with EautraSil Plus is indicated as daily wear soft contact lens for correcting refractive ametropia (myopia and hyperopia) in phakic or aphakic persons with non-diseased eyes who have refractive astigmatism of 1.00D or less, as long as the astigmatism does not ...

**Cited predicates in the corpus:** [K171447](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID=K171447) OxyPure Color Silicone Hydrogel Soft Contact Lenses (primary); [K180322](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID=K180322) Si-Hy (olifilcon B) color silicone hydrogel soft contact lens

| Method | Rank of best cited predicate | Top 5 |
| --- | --- | --- |
| Fine-tuned + TF-IDF | 2 | 1. K141348 VISCO SOFT CONTACT LENS<br>2. **K180322 Si-Hy (olifilcon B) color silicone hydrogel ... CITED**<br>3. K160344 Si-Hy (olifilcon B) sperical silicone ...<br>4. K160245 Visco (olifilcon A) soft contact lens<br>5. **K171447 OxyPure Color Silicone Hydrogel Soft Contact ... CITED** |
| Fine-tuned EmbeddingGemma | 3 | 1. K242413 VizionFocus (mififilcon A) Silicone Hydrogel ...<br>2. K052560 BIOFINITY (COMFILCON A) SOFT CONTACT LENS<br>3. **K180322 Si-Hy (olifilcon B) color silicone hydrogel ... CITED**<br>4. K180398 Focus DAILIES / Focus DAILIES Toric / Focus ...<br>5. K180669 Focus DAILIES / Focus DAILIES Toric / Focus ... |
| TF-IDF | 12 | 1. K141348 VISCO SOFT CONTACT LENS<br>2. K173958 OxyAqua (olifilcon D) spherical silicone ...<br>3. K160344 Si-Hy (olifilcon B) sperical silicone ...<br>4. K141280 ILENS (OCUFILCON D) DAILY SOFT CONTACT LENS<br>5. K191397 OxySoft (olifilcon C) silicone hydrogel soft ... |
| EmbeddingGemma, untuned | 50 | 1. K111267 CONCISE, SILICONE HYDROGEL DAILY WEAR SOFT ...<br>2. K061948 PROCLEAR (OMAFILCON A) DAILY ...<br>3. K202036 Focus DAILIES / Focus DAILIES Toric / Focus ...<br>4. K111966 PROCLEAR, PROCLEAR TORIC, PROCLEAR ...<br>5. K100443 DEFINITIVE, FLEXLENS - SILICONE HYDROGEL |

## Both find a cited predicate in the top 10

### Example 5: [K253528](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID=K253528) Atellica IM CA 19-9 II

Cleared 2026-08-10 · product code NIG · panel IM · 51,375 earlier candidates

> Indications for Use: The Atellica IM CA 19-9 II (CA 19-9 II) assay is an in vitro diagnostic immunoassay for the quantitative measurement of the CA 19-9 tumor-associated antigen in human serum and plasma (EDTA and lithium heparin) using the Atellica IM Analyzer. This assay is indicated for the serial measurement of CA 19-9 to aid in the management of patients diagnosed with cancers of the exocrine pancreas. The test is useful as an aid in monitoring of disease status in those patients having confirmed pancreatic cancer who have levels of serum or plasma CA 19-9 at some point in their disease process exceeding the upper limit of normal determined for the apparently healthy cohort. CA 19-9 values must be interpreted in conjunction with all other clinical and laboratory data before a medical decision is determined.
>
> Device Description: 510(k) Summary Siemens Healthcare Diagnostics Inc., ...

**Cited predicates in the corpus:** [K031393](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID=K031393) BAYER ADVIA CENTAUR CA 19-9 ASSAY (primary)

| Method | Rank of best cited predicate | Top 5 |
| --- | --- | --- |
| Fine-tuned + TF-IDF | 1 | 1. **K031393 BAYER ADVIA CENTAUR CA 19-9 ASSAY CITED**<br>2. K231525 VITROS Immunodiagnostic Products CA 19-9TM ...<br>3. K052889 VITROS CA 19-9<br>4. K200997 Lumipulse G CA19-9-N<br>5. K251630 Atellica IM Total PSA II (tPSAII) |
| Fine-tuned EmbeddingGemma | 1 | 1. **K031393 BAYER ADVIA CENTAUR CA 19-9 ASSAY CITED**<br>2. K231525 VITROS Immunodiagnostic Products CA 19-9TM ...<br>3. K191973 Lumpipulse G CA19-9-N<br>4. K200997 Lumipulse G CA19-9-N<br>5. K052889 VITROS CA 19-9 |
| TF-IDF | 1 | 1. **K031393 BAYER ADVIA CENTAUR CA 19-9 ASSAY CITED**<br>2. K261364 Atellica CH Acetaminophen 2 (ACTM)<br>3. K242981 Atellica IM Thyroglobulin (Tg)<br>4. K171566 Atellica IM High-Sensitivity Troponin I ...<br>5. K172322 Atellica IM Total hCG (ThCG) |
| EmbeddingGemma, untuned | 2 | 1. K231525 VITROS Immunodiagnostic Products CA 19-9TM ...<br>2. **K031393 BAYER ADVIA CENTAUR CA 19-9 ASSAY CITED**<br>3. K052889 VITROS CA 19-9<br>4. K191973 Lumpipulse G CA19-9-N<br>5. K052000 ARCHITECT CA 19-9XR ASSAY |

### Example 6: [K252018](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID=K252018) HERA Z20 Diagnostic Ultrasound System; HERA Z20e Diagnostic Ultrasound System; HERA Z20s Diagnostic Ultrasound System; R20 Diagnostic Ultrasound System; HERA Z30 Diagnostic Ultrasound System; R30 Diagnostic Ultrasound System

Cleared 2026-01-05 · product code IYN · panel RA · 49,458 earlier candidates

> Indications for Use: Please type in the marketing application/submission number, if it is known. This [K-NUMBER] ? textbox will be left blank for original applications/submissions. Please provide the device trade name(s). ? HERA Z20 Diagnostic Ultrasound System; HERA Z20e Diagnostic Ultrasound System; HERA Z20s Diagnostic Ultrasound System; R20 Diagnostic Ultrasound System; HERA Z30 Diagnostic Ultrasound System· R30 Diagnostic Ultrasound System Please provide your Indications for Use below. ? HERA Z20 Diagnostic Ultrasound System; HERA Z20e Diagnostic Ultrasound System; HERA Z20s Diagnostic Ultrasound System; R20 Diagnostic Ultrasound System; HERA Z30 Diagnostic Ultrasound System; R30 Diagnostic Ultrasound System and probes are designed to obtain ultrasound images and analyze body fluids. The clinical applications include: Fetal/Obstetrics, Abdominal, Gynecology, Intra-operative, ...

**Cited predicates in the corpus:** [K241971](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID=K241971) HERA Z20, R20, HERA Z30, R30 Diagnostic Ultrasound System (primary); [K243702](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID=K243702) V8 Diagnostic Ultrasound System; cV8 Diagnostic Ultrasound System; V7 Diagnostic Ultrasound System; cV7 Diagnostic Ultrasound System; V6 Diagnostic Ultrasound System; cV6 Diagnostic Ultrasound System

| Method | Rank of best cited predicate | Top 5 |
| --- | --- | --- |
| Fine-tuned + TF-IDF | 1 | 1. **K241971 HERA Z20, R20, HERA Z30, R30 Diagnostic ... CITED**<br>2. K223387 V8 Diagnostic Ultrasound System, V7 ...<br>3. K220975 V8 Diagnostic Ultrasound System, V7 ...<br>4. K211945 V8 Diagnostic Ultrasound System<br>5. K231772 V8/H8 Diagnostic Ultrasound System, V7/H7 ... |
| Fine-tuned EmbeddingGemma | 1 | 1. **K241971 HERA Z20, R20, HERA Z30, R30 Diagnostic ... CITED**<br>2. K230084 HERA W10 Diagnostic Ultrasound System; HERA ...<br>3. K242444 HERA W10 Diagnostic Ultrasound System; HERA ...<br>4. K211824 HERA W9, HERA W10 Diagnostic Ultrasound ...<br>5. K220043 HERA W10 Diagnostic Ultrasound System, HERA ... |
| TF-IDF | 1 | 1. **K241971 HERA Z20, R20, HERA Z30, R30 Diagnostic ... CITED**<br>2. K211945 V8 Diagnostic Ultrasound System<br>3. K223387 V8 Diagnostic Ultrasound System, V7 ...<br>4. K220975 V8 Diagnostic Ultrasound System, V7 ...<br>5. K231772 V8/H8 Diagnostic Ultrasound System, V7/H7 ... |
| EmbeddingGemma, untuned | 1 | 1. **K241971 HERA Z20, R20, HERA Z30, R30 Diagnostic ... CITED**<br>2. K230084 HERA W10 Diagnostic Ultrasound System; HERA ...<br>3. K242444 HERA W10 Diagnostic Ultrasound System; HERA ...<br>4. K192319 HERA W9 Diagnostic Ultrasound System, HERA ...<br>5. K220043 HERA W10 Diagnostic Ultrasound System, HERA ... |

### Example 7: [K260676](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID=K260676) Radius VSM and Accessories

Cleared 2026-10-14 · product code MHX · panel CV · 51,933 earlier candidates

> Indications for Use: [K-NUMBER] ? textbox will be left blank for original applications/submissions. Please provide the device trade name(s). ? Radius VSM and Accessories Please provide your Indications for Use below. ? The indications for use for the PPG AFib feature is provided below: to replace traditional methods of diagnosis or treatment. It is indicated for use on patients 18 years or older The Radius VSM and accessories are intended to be used as both a wearable multi-parameter patient physiological patient monitoring in hospital and healthcare facilities. with the orientation and activity of adults. during both no motion and motion conditions. The Radius VSM and accessories are indicated for continuous monitoring of skin temperature of adults including those susceptible to pressure ulcers. measure of relative variability of the photoplethysmograph (pleth) of adults during no ...

**Cited predicates in the corpus:** [K213409](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID=K213409) ZEUS System (Zio Watch) (primary); [K250757](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID=K250757) Radius VSM and Accessories

| Method | Rank of best cited predicate | Top 5 |
| --- | --- | --- |
| Fine-tuned + TF-IDF | 1 | 1. **K250757 Radius VSM and Accessories CITED**<br>2. K223498 Radius VSM and Accessories<br>3. K253417 Masimo Root Monitoring System<br>4. K201770 Masimo Rad-G Pulse Oximeter, Masimo Rad-G YI ...<br>5. K193626 Masimo Rad-97 Pulse CO-Oximeter and ... |
| Fine-tuned EmbeddingGemma | 1 | 1. **K250757 Radius VSM and Accessories CITED**<br>2. K223498 Radius VSM and Accessories<br>3. K191882 Masimo Centroid System<br>4. K180046 Masimo Rad-97 and Accessories<br>5. K153225 Masimo Root Monitoring System and Accessories |
| TF-IDF | 1 | 1. **K250757 Radius VSM and Accessories CITED**<br>2. K223498 Radius VSM and Accessories<br>3. K262089 Disposable Warming Blanket<br>4. K110723 M-LNCS OXIMETRY SENSORS, REPROCESSED M-LNCS ...<br>5. K261741 Hemostatic Clip |
| EmbeddingGemma, untuned | 1 | 1. **K250757 Radius VSM and Accessories CITED**<br>2. K223498 Radius VSM and Accessories<br>3. K232389 Carescape SpO2 - Masimo; Masimo rainbow SET ...<br>4. K253417 Masimo Root Monitoring System<br>5. K183697 Masimo Rad-97 and Accessories |

## TF-IDF finds a cited predicate in the top 10; fine-tuned search does not

### Example 8: [K253590](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID=K253590) Millipede System

Cleared 2026-03-16 · product code NRY · panel NE · 50,023 earlier candidates

> Indications for Use: Millipede88 Aspiration Catheter As part of the Millipede System, the Millipede88 Aspiration Catheter with the Perfuze Aspiration Tube Set and a compatible aspiration pump, is indicated for use in the revascularization of patients with acute ischemic stroke secondary to intracranial large vessel occlusive disease within the internal carotid artery and middle cerebral artery M1 segment within 8 hours of symptom onset. Patients who are ineligible for intravenous thrombolytic drug therapy or who have not responded to thrombolytic drug therapy are candidates for treatment. Millipede70 Aspiration Catheter As part of the Millipede System, the Millipede70 Aspiration Catheter with the Perfuze Aspiration Tube Set and a compatible aspiration pump, is indicated for use in the revascularization of patients with acute ischemic stroke secondary to intracranial large vessel ...

**Cited predicates in the corpus:** [K243601](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID=K243601) Route 92 Medical HiPoint Reperfusion System and Aspiration Tubing Set (primary); [K233648](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID=K233648) Millipede 088 Access Catheter

| Method | Rank of best cited predicate | Top 5 |
| --- | --- | --- |
| Fine-tuned + TF-IDF | 14 | 1. K252392 Millipede70 Aspiration Catheter; Perfuze ...<br>2. K250012 Millipede 070 Aspiration Catheter; Perfuze ...<br>3. K232524 Millipede 070 Aspiration Catheter; Perfuze ...<br>4. K242504 Millipede 088 Access Catheter; Millipede 070 ...<br>5. K173761 Penumbra System Reperfusion Catheter JET 7 |
| Fine-tuned EmbeddingGemma | 8 | 1. K252392 Millipede70 Aspiration Catheter; Perfuze ...<br>2. K250012 Millipede 070 Aspiration Catheter; Perfuze ...<br>3. K232524 Millipede 070 Aspiration Catheter; Perfuze ...<br>4. K242504 Millipede 088 Access Catheter; Millipede 070 ...<br>5. K151623 Penumbra System 110 Aspiration Tubing |
| TF-IDF | 7 | 1. K252392 Millipede70 Aspiration Catheter; Perfuze ...<br>2. K250012 Millipede 070 Aspiration Catheter; Perfuze ...<br>3. K232524 Millipede 070 Aspiration Catheter; Perfuze ...<br>4. K242504 Millipede 088 Access Catheter; Millipede 070 ...<br>5. K251006 NeVasc Aspiration System |
| EmbeddingGemma, untuned | 12 | 1. K242504 Millipede 088 Access Catheter; Millipede 070 ...<br>2. K252392 Millipede70 Aspiration Catheter; Perfuze ...<br>3. K250012 Millipede 070 Aspiration Catheter; Perfuze ...<br>4. K232524 Millipede 070 Aspiration Catheter; Perfuze ...<br>5. K251006 NeVasc Aspiration System |

### Example 9: [K252075](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID=K252075) Program insite® Powder-Free, Disposable Nitrile Exam Gloves

Cleared 2025-10-17 · product code LZA · panel HO · 48,829 earlier candidates

> Indications for Use: A powder-free patient examination glove is a disposable device intended for medical purposes that is worn on the examiner's hand or finger to prevent contamination between patient and examiner.
>
> Device Description: The powder-free nitrile exam gloves are non-sterile, single-use, ambidextrous medical examination gloves made from nitrile butadiene rubber (NBR). They are blue, feature a beaded cuff, and are available in sizes S–XL. The gloves are not made with natural rubber latex. Their physical and performance characteristics meet all requirements of ASTM D6319-19. The gloves are powder-free and meet the requirements of ASTM D6124-06. Powder is not used during any stage of manufacturing. Rather, the gloves are manufactured using a non-chlorinated polyurethane (PU) inner coating to reduce surface friction and facilitate donning. More specifically, following the dipping ...

**Cited predicates in the corpus:** [K190159](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID=K190159) Showa(R) Medical Exam Glove (Green) powder-free, disposable nitrile glove (primary)

| Method | Rank of best cited predicate | Top 5 |
| --- | --- | --- |
| Fine-tuned + TF-IDF | 26 | 1. K211864 Powder Free Nitrile Patient Examination ...<br>2. K222498 Medical Examination Gloves (Nitrile) (XS, S, ...<br>3. K203191 LYDUS Nitrile Examination Gloves, Powder Free<br>4. K213993 Powder Free Chloroprene Examination Glove ...<br>5. K133193 RS ORANGE NITRILE MEDICAL EXAMINATION GLOVES ... |
| Fine-tuned EmbeddingGemma | 164 | 1. K172642 Orange Non Sterile Powder Free Nitrile ...<br>2. K150340 POWDER FREE Nitrile GLOVES (White), POWDER ...<br>3. K162113 Powder Free Nitrile Examination Gloves (Blue)<br>4. K123469 POWDER FREE NITRILE PATIENT EXAMINATION ...<br>5. K212085 Siam NTD+ Nitrile Powder-Free Examination ... |
| TF-IDF | 6 | 1. K220229 Powder Free Nitrile Examination Gloves, ...<br>2. K210755 Powder Free Nitrile Patient Examination ...<br>3. K211864 Powder Free Nitrile Patient Examination ...<br>4. K143131 Powder Free Nitrile Patient Examination ...<br>5. K140418 POWDER FREE BLUE NITRILE PATIENT EXAMINATION ... |
| EmbeddingGemma, untuned | 414 | 1. K162113 Powder Free Nitrile Examination Gloves (Blue)<br>2. K210755 Powder Free Nitrile Patient Examination ...<br>3. K212804 Powder free nitrile examination glove- blue, ...<br>4. K220697 Nitrile Patient Examination Gloves, Powder ...<br>5. K140418 POWDER FREE BLUE NITRILE PATIENT EXAMINATION ... |

## Neither finds a cited predicate in the top 10

### Example 10: [K252167](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID=K252167) Enhanced Silver Gelling Fiber Dressing (OTC) Extra Enhanced Silver Gelling Fiber Dressing (Prescription)

Cleared 2025-12-19 · product code FRO · panel SU · 49,335 earlier candidates

> Indications for Use: Enhanced silver gelling fiber dressing (OTC) may be used for the management of: Abrasions Lacerations; Minor cuts; Minor scalds and burns. Extra enhanced silver gelling fiber dressing (Rx only) may be used for the management of: Partial thickness (superficial second degree) burns; Diabetic foot ulcers, leg ulcers, (venous stasis ulcers, arterial ulcers and leg ulcers of mixed etiology) and pressure ulcers/sores (partial & full thickness); Surgical wounds left to heal by secondary intention such as dehisced surgical incisions; Surgical wounds that heal by primary intent such as dermatological and surgical incisions(e.g., orthopedic and vascular); Traumatic wounds; Wounds that are prone to bleeding, such as wounds that have been mechanically or surgically debrided and donor sites; Model: ...

**Cited predicates in the corpus:** [K173675](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID=K173675) AQUACEL Ag+ EXTRA Enhanced Hydrofiber Dressing with Silver and Strengthening Fiber (primary)

| Method | Rank of best cited predicate | Top 5 |
| --- | --- | --- |
| Fine-tuned + TF-IDF | 11 | 1. K240485 LUOFUCON® Extra Silver Gelling Fiber ...<br>2. K210718 LUOFUCON Extra Silver Gelling Fiber ...<br>3. K210974 High Performance Antimicrobial Gelling Fiber ...<br>4. K080383 AQUACEL HYDROFIBER WOUND DRESSING AND AG ...<br>5. K210466 Silver Gelling Fiber Dressing |
| Fine-tuned EmbeddingGemma | 2 | 1. K240485 LUOFUCON® Extra Silver Gelling Fiber ...<br>2. **K173675 AQUACEL Ag+ EXTRA Enhanced Hydrofiber ... CITED**<br>3. K080383 AQUACEL HYDROFIBER WOUND DRESSING AND AG ...<br>4. K210718 LUOFUCON Extra Silver Gelling Fiber ...<br>5. K210974 High Performance Antimicrobial Gelling Fiber ... |
| TF-IDF | 23 | 1. K240485 LUOFUCON® Extra Silver Gelling Fiber ...<br>2. K210718 LUOFUCON Extra Silver Gelling Fiber ...<br>3. K221720 Extra Silver Gelling Fiber Dressing<br>4. K210466 Silver Gelling Fiber Dressing<br>5. K160379 Exufiber Ag+ |
| EmbeddingGemma, untuned | 11 | 1. K221720 Extra Silver Gelling Fiber Dressing<br>2. K220673 Sterile Silver Alginate Wound Dressing<br>3. K210466 Silver Gelling Fiber Dressing<br>4. K210974 High Performance Antimicrobial Gelling Fiber ...<br>5. K202982 Silver Alginate Dressing (Prescription use), ... |

### Example 11: [K250312](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID=K250312) Hudson RCI® Comfort Flo Nasal Cannula Premature, Infant; Hudson RCI® Comfort Flo Plus and Soft Plus Cannula Extra Small

Cleared 2025-07-02 · product code BTT · panel AN · 47,965 earlier candidates

> Indications for Use: Hudson RCI Comfort Flo Nasal Cannulas Premature, Infant: Hudson RCl Comfort Flo Nasal Cannulas Premature, Infant are intended to be used in conjunction with the Comfort Flo Humidification System to provide a continuous flow of heated and humidified healthcare environments. Hudson RCl Comfort Flo Plus Cannulas and Comfort Flo Soft Plus Cannulas Extra Small: intended to be used in conjunction with the Humidification System to provide a continuous flow of heated and humidified air/oxygen mixtures to spontaneously breathing patients It is indicated for single use by adult and pediatric (12 years and above) patients in professional healthcare environments.
>
> Device Description: Hudson RCI® Comfort Flo Nasal Cannulas Premature, Infant Comfort Flo nasal cannulas are intended to be used in conjunction with the Comfort Flo Humidification System to provide a continuous flow of ...

**Cited predicates in the corpus:** [K222197](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID=K222197) F&P Optiflow Junior 2/2+ Nasal Cannula Interface Range (primary)

| Method | Rank of best cited predicate | Top 5 |
| --- | --- | --- |
| Fine-tuned + TF-IDF | 13 | 1. K240163 Hudson RCI® Comfort Flo Nasal Cannula; ...<br>2. K162242 Comfort Flo Humidification System<br>3. K241778 Hudson RCI Comfort Flo® CubCannula<br>4. K131912 CONCHATHERM NEPTUNE HEATED HUMIDIFIER, ...<br>5. K961150 DUAL LUMEN OXYGEN DELIVERY GAS SAMPLING ... |
| Fine-tuned EmbeddingGemma | 21 | 1. K240163 Hudson RCI® Comfort Flo Nasal Cannula; ...<br>2. K241778 Hudson RCI Comfort Flo® CubCannula<br>3. K162242 Comfort Flo Humidification System<br>4. K031745 INFANT FLOW PLUS INFANT CPAP SYSTEM WITH ...<br>5. K093716 BABI PLUS INFANT NASAL CANNULA SYSTEM |
| TF-IDF | 17 | 1. K240163 Hudson RCI® Comfort Flo Nasal Cannula; ...<br>2. K162242 Comfort Flo Humidification System<br>3. K241778 Hudson RCI Comfort Flo® CubCannula<br>4. K131912 CONCHATHERM NEPTUNE HEATED HUMIDIFIER, ...<br>5. K223100 Hudson RCI AquaTherm III Plus External ... |
| EmbeddingGemma, untuned | 25 | 1. K240163 Hudson RCI® Comfort Flo Nasal Cannula; ...<br>2. K241778 Hudson RCI Comfort Flo® CubCannula<br>3. K162242 Comfort Flo Humidification System<br>4. K234032 Hudson RCI Dri-Tech Breathing Circuits; ...<br>5. K961150 DUAL LUMEN OXYGEN DELIVERY GAS SAMPLING ... |

### Example 12: [K253549](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID=K253549) DeVilbiss 5 Liter Oxygen Concentrator (555)

Cleared 2026-03-26 · product code CAW · panel AN · 50,124 earlier candidates

> Indications for Use: The DeVilbiss 5 Liter (555) Oxygen Concentrator is intended to provide supplemental oxygen to adult patients requiring oxygen therapy. The device may be used in the home or an institutional setting. The device is not intended to be life sustaining or life supporting.
>
> Device Description: The DeVilbiss 5-Liter (555) Oxygen Concentrator (hereafter referred to as “555”) is a portable device that produces an oxygen enriched gas mixture by drawing in room air and extracting nitrogen through a pressure swing adsorption process, allowing oxygen to be delivered at a range of prescribed flows to patients in need of supplemental oxygen. DeVilbiss Healthcare LLC: DeVilbiss 5 Liter (555) Oxygen Concentrator

**Cited predicates in the corpus:** [K230969](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID=K230969) Oxygen Concentrator (primary)

| Method | Rank of best cited predicate | Top 5 |
| --- | --- | --- |
| Fine-tuned + TF-IDF | 73 | 1. K250805 DeVilbiss PulmO2 10-Liter Oxygen ...<br>2. K081468 DEVILBISS PORTABLE OXYGEN CONCENTRATOR, 306 ...<br>3. K061261 L4 OXYGEN CONCENTRATOR<br>4. K132489 INOGEN AT HOME OXYGEN CONCENTRATOR (AKA ...<br>5. K113317 LIFECHOICE OXYGEN CONCENTRATOR |
| Fine-tuned EmbeddingGemma | 154 | 1. K250805 DeVilbiss PulmO2 10-Liter Oxygen ...<br>2. K081468 DEVILBISS PORTABLE OXYGEN CONCENTRATOR, 306 ...<br>3. K061261 L4 OXYGEN CONCENTRATOR<br>4. K251607 Oxygen Concentrator (J5A)<br>5. K240840 Oxygen Concentrator (ZY-10AB/115) |
| TF-IDF | 15 | 1. K250805 DeVilbiss PulmO2 10-Liter Oxygen ...<br>2. K081468 DEVILBISS PORTABLE OXYGEN CONCENTRATOR, 306 ...<br>3. K061261 L4 OXYGEN CONCENTRATOR<br>4. K113317 LIFECHOICE OXYGEN CONCENTRATOR<br>5. K191875 Owgels Oxygen Concentrator |
| EmbeddingGemma, untuned | 67 | 1. K250805 DeVilbiss PulmO2 10-Liter Oxygen ...<br>2. K081468 DEVILBISS PORTABLE OXYGEN CONCENTRATOR, 306 ...<br>3. K240840 Oxygen Concentrator (ZY-10AB/115)<br>4. K250625 Portable oxygen concentrator  (W-R1(MAX)); ...<br>5. K203210 Invacare Platinum 5NXG Oxygen Concentrator |
