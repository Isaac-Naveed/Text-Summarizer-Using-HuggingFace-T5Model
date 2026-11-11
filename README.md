# Text Summarizer Using Hugging Face T5

A web app that turns chat-style conversations into short summaries. It uses a **T5-small** model fine-tuned on the **SAMSum** dataset, trained on Google Colab and served through **FastAPI** with a simple HTML/CSS frontend.

## Features

- Summarizes dialogues into one or two sentences
- Fine-tuned model hosted on the Hugging Face Hub: [IsaacNaveed/t5-small-samsum-summarizer](https://huggingface.co/IsaacNaveed/t5-small-samsum-summarizer)
- FastAPI backend with a REST endpoint (`/summarize`)
- Clean, responsive web interface
- Training notebook included

## Tech Stack

- **Model:** T5-small (Hugging Face Transformers)
- **Training:** PyTorch, Hugging Face `Trainer`, Google Colab (GPU)
- **Backend:** FastAPI, Uvicorn
- **Frontend:** HTML, CSS, JavaScript
- **Dataset:** [SAMSum](https://huggingface.co/datasets/Samsung/samsum)

## Project Structure

```
├── app.py                  # FastAPI application
├── index.html              # Frontend page
├── static/
│   └── style.css           # Styles
├── text_summarizer.ipynb   # Training notebook
├── requirements.txt        # Dependencies
└── README.md
```

## How It Works

1. Dialogues are cleaned (extra whitespace and HTML tags removed, lowercased).
2. Text is tokenized with the T5 tokenizer (max 512 input tokens, 150 summary tokens).
3. T5-small is fine-tuned on a sample of SAMSum (4,000 training and 500 validation examples, 6 epochs).
4. The app loads the model from the Hugging Face Hub and generates summaries with beam search (4 beams).

## Getting Started

### Prerequisites

- Python 3.10 or newer
- An internet connection on first run (the model, about 244 MB, downloads automatically)

### Installation

```bash
git clone https://github.com/Isaac-Naveed/Text-Summarizer-Using-HuggingFace-T5Model.git
cd Text-Summarizer-Using-HuggingFace-T5Model
pip install -r requirements.txt
```

### Run the app

```bash
uvicorn app:app --reload
```

Open http://127.0.0.1:8000 in your browser.

## API Usage

**POST** `/summarize`

Request:

```json
{
  "dialogue": "amanda: i baked cookies. do you want some? jerry: sure! amanda: i'll bring you some tomorrow :)"
}
```

Response:

```json
{
  "summary": "Amanda baked cookies and will bring Jerry some tomorrow."
}
```

Interactive API docs are available at `/docs`.

## Example Input

```
tom: did you finish the project report? lisa: almost, i still need to add the budget section. tom: the manager wants it by friday noon. lisa: i know, i'll send it by thursday evening.
```

## Training Your Own Model

Open `text_summarizer.ipynb` in Google Colab, switch the runtime to a GPU (**Runtime > Change runtime type > T4 GPU**), upload the SAMSum CSV files, and run all cells. The notebook saves the trained model, which you can download or push to the Hugging Face Hub.

## Limitations

- Works best on chat-style dialogues like those in SAMSum.
- Articles and long paragraphs without speaker names give weaker results.
- Input is truncated at 512 tokens.
- Trained on a subset of the data with a small model, so summaries can be inaccurate.
- The SAMSum dataset has its own license (CC BY-NC-ND 4.0), which applies to the training data.

## Future Improvements

- Train on the full SAMSum dataset
- Evaluate with ROUGE scores
- Try `t5-base` or `flan-t5` for better quality
- Support longer documents
- Deploy to Hugging Face Spaces

## Author

**Isaac Naveed** - [GitHub](https://github.com/Isaac-Naveed)
