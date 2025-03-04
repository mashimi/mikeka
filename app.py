import streamlit as st
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import hashlib
import json
import os
from langchain.chains import SequentialChain, LLMChain
from dotenv import load_dotenv
load_dotenv()
from langchain.prompts import PromptTemplate
from langchain_community.chat_models.perplexity import ChatPerplexity
import plotly.graph_objects as go

# -------------------- Configuration --------------------
PPLX_API_KEY = os.getenv("PPLX_API_KEY")
CACHE_FILE = "betpawa_cache.json"
CACHE_DURATION = 900  # 15 minutes

# -------------------- Helper Functions --------------------
def initialize_session_state():
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []

# -------------------- Web Scraping Module --------------------
class BetpawaScraper:
    def __init__(self):
        self.base_url = "https://www.betpawa.co.tz/events"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def generate_cache_key(self, params):
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()

    def load_cache(self):
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def save_cache(self, cache):
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)

    def is_cache_valid(self, cache_entry):
        if not cache_entry:
            return False
        cache_time = datetime.fromisoformat(cache_entry['timestamp'])
        return (datetime.now() - cache_time) < timedelta(seconds=CACHE_DURATION)

    def scrape_events(self, params):
        cache = self.load_cache()
        cache_key = self.generate_cache_key(params)

        if cache_key in cache and self.is_cache_valid(cache[cache_key]):
            return cache[cache_key]['data']

        try:
            response = requests.get(self.base_url, params=params, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            events = []
            
            for event in soup.select('.event-list-group-container'):
                try:
                    teams = [t.text.strip() for t in event.select('.event-name-team')]
                    odds = [float(o.text.strip()) for o in event.select('.button-odds')[:3]]
                    
                    events.append({
                        'teams': teams,
                        'odds': {
                            'home': odds[0],
                            'draw': odds[1],
                            'away': odds[2]
                        },
                        'time': event.select_one('.event-time').text.strip(),
                        'competition': event.select_one('.event-competition').text.strip()
                    })
                except Exception as e:
                    continue

            cache[cache_key] = {
                'data': events,
                'timestamp': datetime.now().isoformat()
            }
            self.save_cache(cache)
            
            return events

        except requests.RequestException as e:
            st.error(f"Failed to fetch data: {str(e)}")
            return None

# -------------------- AI Prediction System --------------------
class PerplexityConfig:
    MODELS = {
        'reasoning': {
            'name': 'sonar-reasoning',
            'max_tokens': 4096,
            'temperature_range': (0.1, 1.0)
        }
    }

class FootballPredictor:
    def __init__(self, model_type='reasoning', temperature=0.5):
        if model_type not in PerplexityConfig.MODELS:
            raise ValueError(f"Invalid model type. Choose from: {list(PerplexityConfig.MODELS.keys())}")
            
        model_config = PerplexityConfig.MODELS[model_type]
        
        self.llm = ChatPerplexity(
            model=model_config['name'],
            temperature=temperature,
            max_tokens=model_config['max_tokens'],
            pplx_api_key=PPLX_API_KEY
        )
        self.chains = self._build_chains()

    def _build_chains(self):
        team_stats_template = """Analyze these football teams: {home_team} vs {away_team}.
        Consider their last 5 matches, head-to-head history, and key player statistics.
        Provide detailed analysis in bullet points."""
        
        prediction_template = """Based on the following analysis:
        {team_stats}
        {sentiment_analysis}
        {match_context}
        Provide a predicted outcome with confidence percentage and recommended bet."""
        
        chains = [
            LLMChain(
                llm=self.llm,
                prompt=PromptTemplate(
                    input_variables=["home_team", "away_team"],
                    template=team_stats_template
                ),
                output_key="team_stats"
            ),
            LLMChain(
                llm=self.llm,
                prompt=PromptTemplate(
                    input_variables=["home_team", "away_team"],
                    template="Analyze social media sentiment around {home_team} vs {away_team} match."
                ),
                output_key="sentiment_analysis"
            ),
            LLMChain(
                llm=self.llm,
                prompt=PromptTemplate(
                    input_variables=["home_team", "away_team"],
                    template="Identify key contextual factors for {home_team} vs {away_team} match."
                ),
                output_key="match_context"
            ),
            LLMChain(
                llm=self.llm,
                prompt=PromptTemplate(
                    input_variables=["team_stats", "sentiment_analysis", "match_context"],
                    template=prediction_template
                ),
                output_key="prediction"
            )
        ]
        return SequentialChain(
            chains=chains,
            input_variables=["home_team", "away_team"],
            output_variables=["team_stats", "sentiment_analysis", "match_context", "prediction"],
            verbose=False
        )

    def analyze_match(self, home_team, away_team):
        return self.chains({
            "home_team": home_team,
            "away_team": away_team
        })

    def _create_odds_chart(self, odds):
        fig = go.Figure(go.Bar(
            x=list(odds.keys()),
            y=list(odds.values()),
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c']
        ))
        fig.update_layout(
            title="Live Odds Comparison",
            xaxis_title="Outcome",
            yaxis_title="Odds",
            template="plotly_white"
        )
        return fig

# -------------------- Streamlit UI --------------------
def main():
    st.set_page_config(
        page_title="AI Football Betting Assistant",
        page_icon="⚽",
        layout="wide"
    )
    
    initialize_session_state()
    
    st.title("⚽ AI Football Betting Assistant")
    st.markdown("---")
    
    # Sidebar Controls
    with st.sidebar:
        st.header("Match Parameters")
        home_team = st.text_input("Home Team", "Manchester City")
        away_team = st.text_input("Away Team", "Liverpool")
        analyze_button = st.button("Analyze Match")
    
    # Main Content Area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Match Analysis")
        
        if analyze_button:
            with st.spinner("Analyzing match..."):
                # Get live odds
                scraper = BetpawaScraper()
                params = {
                    'marketId': '1X2',
                    'competitions': '11965,12541,12546,12545,12097,12110,12039,12127,12355',
                    'categoryId': '2'
                }
                events = scraper.scrape_events(params)
                
                # Get AI prediction
                predictor = FootballPredictor()
                result = predictor.analyze_match(home_team, away_team)
                
                # Store results
                analysis_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "teams": f"{home_team} vs {away_team}",
                    "prediction": result,
                    "odds": next((e for e in events if 
                                home_team.lower() in [t.lower() for t in e['teams']] and
                                away_team.lower() in [t.lower() for t in e['teams']]), None)
                }
                st.session_state.analysis_history.append(analysis_entry)
                
        if st.session_state.analysis_history:
            latest = st.session_state.analysis_history[-1]
            
            # Display Results
            with st.expander("Team Statistics", expanded=True):
                st.markdown(latest['prediction']['team_stats'])
            
            with st.expander("Sentiment Analysis"):
                st.markdown(latest['prediction']['sentiment_analysis'])
            
            with st.expander("Match Context"):
                st.markdown(latest['prediction']['match_context'])
            
            with st.expander("AI Prediction"):
                st.markdown(latest['prediction']['prediction'])
    
    with col2:
        st.subheader("Live Odds Comparison")
        
        if st.session_state.analysis_history and latest['odds']:
            odds = latest['odds']['odds']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"{home_team} Win", value=f"{odds['home']:.2f}")
            with col2:
                st.metric("Draw", value=f"{odds['draw']:.2f}")
            with col3:
                st.metric(f"{away_team} Win", value=f"{odds['away']:.2f}")
            
            st.plotly_chart(predictor._create_odds_chart(odds))
        else:
            st.info("No odds data available for this match")

if __name__ == "__main__":
    main()
