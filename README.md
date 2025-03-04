# AI Football Betting Assistant ⚽

A Streamlit-powered web application that combines real-time betting odds scraping with AI-powered match predictions.

![Demo](https://via.placeholder.com/800x400.png?text=AI+Betting+Interface+Preview)

## Features
- Real-time odds scraping from Betpawa
- Perplexity AI-powered match predictions
- Risk assessment analysis
- Historical performance tracking
- Interactive Streamlit web interface

## Installation
```bash
pip install -r requirements.txt
```

## Configuration
1. Create `.env` file:
```env
PPLX_API_KEY="your_perplexity_api_key_here"
```

## Usage
```bash
streamlit run app.py
```

## Project Structure
```
├── app.py                 - Main application logic
├── requirements.txt       - Python dependencies
├── .env                   - Environment configuration
└── betpawa_cache.json     - Cached betting data
```

## Deployment

1. Install Vercel CLI:
```bash
npm install -g vercel
```

2. Link and deploy:
```bash
vercel login
vercel link
vercel deploy --prod
```

## License
MIT License - See [LICENSE](LICENSE) for details
