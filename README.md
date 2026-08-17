# Pokémon Card Identifier

This project is a simple Python app that reads Pokémon card images, detects the card name and number, and looks up the card details from the TCGdex API.

It is useful for scanning a folder of card images and quickly identifying cards with OCR and online metadata.

## What it does

- Reads card images from a folder
- Detects the card name using OCR
- Detects the card number/series number
- Searches the card in the TCGdex database
- Shows card details such as:
  - HP
  - Types
  - Rarity
  - Set
  - Artist
  - Attacks
- Saves results to a JSON file

## Project files

- `card_identify.py` - main script
- `requirements.txt` - Python dependencies
- `images1/` - folder for card images
- `scan_results.json` - output file generated after processing

## Requirements

Install the needed Python packages:

```bash
pip install -r requirements.txt
```

## How to run

1. Put your Pokémon card images in the `images1` folder.
2. Open a terminal in the project folder.
3. Run:

```bash
python card_identify.py
```

You can also point to a different image folder if needed:

```bash
python card_identify.py --folder my_cards
```

## Controls

- `Space` - next card
- `B` - previous card
- `Q` - quit

## Output

The app displays each card in a window with the detected details on the right side and saves the results to `images1/scan_results.json`.

## Notes

This is a lightweight project focused on OCR and card lookup, so it works best with clear card images and good lighting.
