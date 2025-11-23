"""
Real-Time Ad Serving System.

This module implements the complete ad serving pipeline:
1. Receive user request
2. Retrieve candidates using generative retrieval
3. Rank candidates using DLRM
4. Run VCG auction
5. Return winning ads
6. Record feedback

FastAPI-based REST API for production serving.
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import torch
import uvicorn
from datetime import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semantic_indexer import RQVAE
from generative_retrieval import GenerativeRetrievalModel, SemanticIDTrie
from ranking import DLRM
from auction import VCGAuction, RealTimeAuctionEngine, BudgetTracker, BudgetPacer, Bid
from feedback import FeedbackLoop, FeedbackEvent
from data.schemas import InteractionType


# Request/Response models
class AdRequest(BaseModel):
    """Ad serving request."""
    user_id: str
    user_history: List[List[int]]  # List of semantic IDs
    demographics: Dict[str, int]
    context: Dict[str, float]
    num_ads: int = 3


class AdResponse(BaseModel):
    """Ad serving response."""
    request_id: str
    ads: List[Dict]
    timestamp: str


class FeedbackRequest(BaseModel):
    """Feedback event request."""
    request_id: str
    user_id: str
    ad_id: str
    event_type: str  # 'click', 'conversion', 'skip'
    value: float = 0.0


class AdServer:
    """
    Complete ad serving system.

    Coordinates all components for real-time ad serving.
    """

    def __init__(
        self,
        rqvae_path: str,
        generative_model_path: str,
        ranking_model_path: str,
        trie_path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device

        # Load models
        print("Loading models...")

        # RQ-VAE
        self.rqvae = RQVAE(
            text_dim=768,
            image_dim=512,
            category_vocab_size=100,
            embedding_dim=256,
            num_quantizers=4,
            codebook_size=256
        ).to(device)

        if os.path.exists(rqvae_path):
            self.rqvae.load_state_dict(torch.load(rqvae_path, map_location=device))
        self.rqvae.eval()

        # Generative Retrieval
        self.generative_model = GenerativeRetrievalModel(
            num_quantizers=4,
            codebook_size=256,
            embed_dim=512,
            num_encoder_layers=6,
            num_decoder_layers=6,
            num_heads=8
        ).to(device)

        if os.path.exists(generative_model_path):
            self.generative_model.load_state_dict(
                torch.load(generative_model_path, map_location=device)
            )
        self.generative_model.eval()

        # Ranking Model
        self.ranking_model = DLRM(
            num_user_categorical=10,
            num_user_continuous=20,
            semantic_embed_dim=256,
            num_ad_categorical=5,
            num_ad_continuous=10
        ).to(device)

        if os.path.exists(ranking_model_path):
            self.ranking_model.load_state_dict(
                torch.load(ranking_model_path, map_location=device)
            )
        self.ranking_model.eval()

        # Trie for constrained generation
        if os.path.exists(trie_path):
            self.trie = SemanticIDTrie.load(trie_path)
        else:
            self.trie = SemanticIDTrie(num_quantizers=4, codebook_size=256)

        # Auction system
        self.vcg_auction = VCGAuction(reserve_price=0.01)
        self.budget_tracker = BudgetTracker()
        self.budget_pacer = BudgetPacer(self.budget_tracker)
        self.auction_engine = RealTimeAuctionEngine(
            self.vcg_auction,
            self.budget_tracker
        )

        # Feedback loop
        self.feedback_loop = FeedbackLoop(
            event_buffer_size=10000,
            feature_store=None,  # Could integrate with online training
            online_trainer=None
        )
        self.feedback_loop.start()

        # Ad database (mock - would be real DB in production)
        self.ad_database: Dict[str, Dict] = {}

        print("Ad server initialized successfully!")

    @torch.no_grad()
    def serve_ads(self, request: AdRequest) -> AdResponse:
        """
        Main ad serving pipeline.

        Args:
            request: Ad serving request

        Returns:
            Ad response with winning ads
        """
        request_id = f"req_{datetime.now().timestamp()}"

        # 1. Prepare input
        user_history = torch.LongTensor(request.user_history).unsqueeze(0).to(self.device)
        demographics = torch.LongTensor(list(request.demographics.values())).unsqueeze(0).to(self.device)
        context = torch.FloatTensor(list(request.context.values())).unsqueeze(0).to(self.device)

        # Create attention mask
        history_len = len(request.user_history)
        attention_mask = torch.ones(1, history_len, dtype=torch.bool).to(self.device)

        # 2. Generative Retrieval
        retrieved_ids, retrieval_scores = self.generative_model.generate(
            semantic_ids=user_history,
            demographic_ids=demographics,
            context_features=context,
            trie=self.trie,
            num_beams=10,
            num_return=50,
            attention_mask=attention_mask
        )

        # 3. Get ad candidates
        candidates = []
        for i in range(retrieved_ids.shape[1]):
            semantic_id = tuple(retrieved_ids[0, i].cpu().tolist())

            # Look up ad from trie
            ad_id = self.trie.search(semantic_id)
            if ad_id and ad_id in self.ad_database:
                ad_data = self.ad_database[ad_id]

                # Get semantic embedding
                semantic_embedding = self.rqvae.lookup_by_semantic_id(
                    retrieved_ids[0:1, i:i+1].to(self.device)
                )

                candidates.append({
                    'ad_id': ad_id,
                    'advertiser_id': ad_data['advertiser_id'],
                    'semantic_embedding': semantic_embedding,
                    'retrieval_score': retrieval_scores[0, i].item(),
                    **ad_data
                })

        if not candidates:
            return AdResponse(
                request_id=request_id,
                ads=[],
                timestamp=datetime.now().isoformat()
            )

        # 4. Ranking
        ranking_results = []

        for candidate in candidates:
            # Prepare features (mock - would extract real features)
            user_cat = torch.zeros(1, 10, dtype=torch.long).to(self.device)
            user_cont = torch.zeros(1, 20, dtype=torch.float).to(self.device)
            ad_sem = candidate['semantic_embedding']
            ad_cat = torch.zeros(1, 5, dtype=torch.long).to(self.device)
            ad_cont = torch.zeros(1, 10, dtype=torch.float).to(self.device)

            # Get predictions
            predictions = self.ranking_model(user_cat, user_cont, ad_sem, ad_cat, ad_cont)

            ranking_results.append({
                'ad_id': candidate['ad_id'],
                'advertiser_id': candidate['advertiser_id'],
                'click_prob': predictions['click'].item(),
                'conversion_prob': predictions['conversion'].item(),
                'skip_prob': predictions['skip'].item(),
                'quality_score': 1.0 - predictions['skip'].item()
            })

        # 5. Auction
        # Mock advertiser bids
        advertiser_bids = {
            candidate['advertiser_id']: {
                'bid_price': 5.0,  # $5 CPM
                'bid_type': 'cpm'
            }
            for candidate in candidates
        }

        auction_outcomes = self.auction_engine.run_real_time_auction(
            candidates=ranking_results,
            advertiser_bids=advertiser_bids,
            num_slots=request.num_ads
        )

        # 6. Prepare response
        winning_ads = []
        for outcome in auction_outcomes:
            ad_data = self.ad_database.get(outcome.ad_id, {})
            winning_ads.append({
                'ad_id': outcome.ad_id,
                'advertiser_id': outcome.advertiser_id,
                'title': ad_data.get('title', ''),
                'description': ad_data.get('description', ''),
                'image_url': ad_data.get('image_url', ''),
                'rank': outcome.rank,
                'price': outcome.vcg_price
            })

        return AdResponse(
            request_id=request_id,
            ads=winning_ads,
            timestamp=datetime.now().isoformat()
        )

    def record_feedback(self, feedback: FeedbackRequest):
        """
        Record user feedback event.

        Args:
            feedback: Feedback event
        """
        event = FeedbackEvent(
            event_id=f"event_{datetime.now().timestamp()}",
            user_id=feedback.user_id,
            ad_id=feedback.ad_id,
            event_type=feedback.event_type,
            timestamp=datetime.now(),
            value=feedback.value
        )

        self.feedback_loop.record_event(event)

    def register_ad(self, ad_data: Dict):
        """Register a new ad in the system."""
        self.ad_database[ad_data['ad_id']] = ad_data

    def get_metrics(self) -> Dict:
        """Get system metrics."""
        return {
            'feedback_metrics': self.feedback_loop.get_metrics(),
            'online_metrics': self.feedback_loop.compute_online_metrics(),
            'budget_states': {
                adv_id: {
                    'remaining': state.remaining_budget(),
                    'utilization': state.utilization()
                }
                for adv_id, state in self.budget_tracker.get_all_states().items()
            }
        }


# FastAPI Application
app = FastAPI(title="Generative Ads System API")

# Global ad server instance
ad_server: Optional[AdServer] = None


@app.on_event("startup")
async def startup_event():
    """Initialize ad server on startup."""
    global ad_server

    # Paths to model checkpoints
    rqvae_path = os.getenv('RQVAE_PATH', './checkpoints/rqvae/best_model.pt')
    generative_path = os.getenv('GENERATIVE_PATH', './checkpoints/generative/best_model.pt')
    ranking_path = os.getenv('RANKING_PATH', './checkpoints/ranking/best_model.pt')
    trie_path = os.getenv('TRIE_PATH', './data/semantic_trie.pkl')

    ad_server = AdServer(
        rqvae_path=rqvae_path,
        generative_model_path=generative_path,
        ranking_model_path=ranking_path,
        trie_path=trie_path
    )


@app.post("/serve_ads")
async def serve_ads_endpoint(request: AdRequest) -> AdResponse:
    """Serve ads endpoint."""
    if ad_server is None:
        raise HTTPException(status_code=500, detail="Ad server not initialized")

    try:
        response = ad_server.serve_ads(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
async def feedback_endpoint(feedback: FeedbackRequest):
    """Record feedback endpoint."""
    if ad_server is None:
        raise HTTPException(status_code=500, detail="Ad server not initialized")

    try:
        ad_server.record_feedback(feedback)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics_endpoint():
    """Get system metrics."""
    if ad_server is None:
        raise HTTPException(status_code=500, detail="Ad server not initialized")

    try:
        metrics = ad_server.get_metrics()
        return JSONResponse(content=metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


def main():
    """Run the ad server."""
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )


if __name__ == "__main__":
    main()
