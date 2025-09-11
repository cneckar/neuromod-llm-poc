# Archive: Full Pack Collection

This directory contains the complete collection of neuromodulation packs that were available in the original system.

## Contents

- **`config_full.json`**: Complete pack collection (82 packs) - original full set
- **`full_pack_collection.json`**: Backup of the original config.json
- **`demo_export.json`**: Demo export file (redundant with config_full.json)
- **`image_focused_packs.json`**: Image generation focused packs (separate from core research)

## Pack Categories in Full Collection

### Real-World Substances (~50 packs)
- **Stimulants**: caffeine, cocaine, amphetamine, methamphetamine, methylphenidate, modafinil, etc.
- **Psychedelics**: LSD, psilocybin, DMT, mescaline, 2C-B, etc.
- **Depressants**: alcohol, benzodiazepines, heroin, morphine, fentanyl, etc.
- **Dissociatives**: ketamine, PCP, DXM, nitrous oxide, etc.
- **Empathogens**: MDMA, MDA, 6-APB, etc.
- **Cannabis**: cannabis_thc
- **Other**: Various other psychoactive substances

### Fictional Substances (~20 packs)
- **Sci-Fi**: Melange (Dune), NZT-48 (Limitless), Soma (Brave New World)
- **Gaming**: Skooma (Elder Scrolls), Jet (Fallout), ADAM (BioShock)
- **Cyberpunk**: Black Lace, Novacoke (Shadowrun)
- **Anime/Manga**: Red Eye (Cowboy Bebop)

### Specialized Research Packs (~12 packs)
- **Research**: Mentor, Speciation, Archivist, Goldfish
- **Creative**: Tightrope, Firekeeper, Librarian's Bloom
- **Technical**: Timepiece, Echonull, Chorus, Quanta
- **Specialized**: Anchorite, Parliament

## Usage

These packs are archived but can be restored if needed for extended research or if specific fictional/specialized packs are required for particular studies.

To restore the full collection:
```bash
cp packs/archive/config_full.json packs/config.json
```

## Note

The main research focuses on the essential 28 packs in the parent directory, which cover all the core categories mentioned in the paper outline and analysis plan.
