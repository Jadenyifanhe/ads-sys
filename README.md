# Generative Ads System

A Meta-style production-grade ads system built with generative retrieval, deep learning ranking, and real-time auction mechanisms.

## Overview

This system implements a modern, state-of-the-art advertising platform inspired by Meta's foundational models approach. Instead of thousands of small models, it uses a few massive "Foundational Models" that work together to deliver highly relevant ads while maximizing advertiser value.

### Key Features

- **Generative Retrieval**: Autoregressive transformer that generates semantic IDs of relevant ads (TIGER/DSI paradigm)
- **Semantic Indexing**: RQ-VAE encoder that converts ads into discrete semantic IDs
- **Unified Ranking**: DLRM-based multi-task learning model predicting clicks, conversions, and engagement
- **Real-Time Auction**: VCG auction mechanism with budget pacing
- **Online Learning**: Real-time model updates based on user feedback

## Architecture

### The 5 Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                     Ad Serving Pipeline                          │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────┐
│  1. Semantic Indexer │  ─── Converts ads to Semantic IDs
│      (RQ-VAE)        │      using vector quantization
└──────────────────────┘
           │
           ▼
┌──────────────────────┐
│  2. Generative       │  ─── Generates candidate Semantic IDs
│     Retrieval Engine │      using constrained beam search
│   (Transformer)      │
└──────────────────────┘
           │
           ▼
┌──────────────────────┐
│  3. Unified Ranking  │  ─── Scores candidates with multi-task
│      (DLRM)          │      learning (CTR, CVR, etc.)
└──────────────────────┘
           │
           ▼
┌──────────────────────┐
│  4. VCG Auction &    │  ─── Runs auction and manages budgets
│     Budget Pacing    │
└──────────────────────┘
           │
           ▼
┌──────────────────────┐
│  5. Feedback Loop &  │  ─── Collects feedback and updates
│  Online Learning     │      models in real-time
└──────────────────────┘
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/ads-sys.git
cd ads-sys

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
ads-sys/
├── src/
│   ├── semantic_indexer/      # RQ-VAE implementation
│   │   ├── rq_vae.py         # RQ-VAE model
│   │   └── __init__.py
│   ├── generative_retrieval/  # Generative retrieval engine
│   │   ├── generative_model.py  # Transformer model
│   │   ├── prefix_trie.py       # Constrained beam search
│   │   └── __init__.py
│   ├── ranking/               # DLRM ranking model
│   │   ├── dlrm.py           # DLRM with multi-task learning
│   │   └── __init__.py
│   ├── auction/               # Auction and budget pacing
│   │   ├── vcg_auction.py    # VCG auction mechanism
│   │   ├── budget_pacing.py  # Budget management
│   │   └── __init__.py
│   ├── feedback/              # Online learning
│   │   ├── online_learning.py
│   │   └── __init__.py
│   ├── data/                  # Data models and schemas
│   │   ├── schemas.py
│   │   └── __init__.py
│   ├── training/              # Training pipelines
│   │   ├── train_rqvae.py
│   │   ├── train_generative.py
│   │   ├── train_ranking.py
│   │   └── __init__.py
│   ├── serving/               # Inference serving
│   │   ├── ad_server.py      # FastAPI server
│   │   └── __init__.py
│   └── utils/                 # Utilities
│       ├── metrics.py
│       ├── config.py
│       └── __init__.py
├── configs/                   # Configuration files
│   └── default_config.yaml
├── scripts/                   # Deployment scripts
├── tests/                     # Unit tests
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Train the Semantic Indexer (RQ-VAE)

First, train the RQ-VAE model to convert ads into discrete semantic IDs:

```bash
python src/training/train_rqvae.py
```

This will:
- Encode ads into 256-dimensional continuous embeddings
- Quantize embeddings into 4-level semantic IDs (e.g., `(12, 56, 99, 3)`)
- Save the trained model to `./checkpoints/rqvae/`

### 2. Build the Semantic ID Trie

After training RQ-VAE, build the prefix trie for constrained generation:

```python
from src.generative_retrieval import SemanticIDTrie, build_trie_from_ads
from src.semantic_indexer import RQVAE

# Load trained RQ-VAE
model = RQVAE(...)
model.load_state_dict(torch.load('./checkpoints/rqvae/best_model.pt'))

# Encode all ads to semantic IDs
semantic_ids = []
ad_ids = []

for ad in ads:
    semantic_id = model.get_semantic_id(ad.text_features, ad.image_features, ad.category_id)
    semantic_ids.append(tuple(semantic_id.cpu().tolist()))
    ad_ids.append(ad.ad_id)

# Build trie
trie = build_trie_from_ads(semantic_ids, ad_ids, num_quantizers=4, codebook_size=256)
trie.save('./data/semantic_trie.pkl')
```

### 3. Train the Generative Retrieval Model

Train the autoregressive model to generate semantic IDs:

```bash
python src/training/train_generative.py
```

This model learns to predict which ads a user will interact with next, given their history.

### 4. Train the Ranking Model

Train the DLRM ranking model:

```bash
python src/training/train_ranking.py
```

This model predicts:
- `P(Click)` - Probability of click
- `P(Conversion)` - Probability of conversion
- `P(Skip)` - Probability of skip
- `Expected View Time`

### 5. Start the Ad Server

Launch the production ad serving API:

```bash
python src/serving/ad_server.py
```

The server will start on `http://localhost:8000`.

### 6. Serve Ads

Make a request to serve ads:

```bash
curl -X POST http://localhost:8000/serve_ads \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "user_history": [[12, 56, 99, 3], [45, 78, 23, 1]],
    "demographics": {"age": 25, "gender": 1, "location": 10},
    "context": {"hour": 14.5, "device_type": 1.0},
    "num_ads": 3
  }'
```

Response:

```json
{
  "request_id": "req_1234567890",
  "ads": [
    {
      "ad_id": "ad_5678",
      "advertiser_id": "adv_123",
      "title": "Amazing Product",
      "description": "Buy now!",
      "image_url": "https://...",
      "rank": 1,
      "price": 4.52
    },
    ...
  ],
  "timestamp": "2025-11-23T10:30:00"
}
```

### 7. Record Feedback

Send user feedback (clicks, conversions) to improve the models:

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "req_1234567890",
    "user_id": "user123",
    "ad_id": "ad_5678",
    "event_type": "click",
    "value": 1.0
  }'
```

## Detailed Component Documentation

### 1. Semantic Indexer (RQ-VAE)

**Purpose**: Convert ads into discrete semantic IDs

**Architecture**:
- **Encoder**: Multimodal encoder for text, images, and metadata
- **Quantizer**: Residual Vector Quantization with 4 levels
- **Decoder**: Reconstructs original features (for training)

**Key Files**:
- `src/semantic_indexer/rq_vae.py`

**Usage**:
```python
from src.semantic_indexer import RQVAE

model = RQVAE(
    text_dim=768,
    image_dim=512,
    embedding_dim=256,
    num_quantizers=4,
    codebook_size=256
)

# Get semantic ID for an ad
semantic_id = model.get_semantic_id(text_features, image_features, category_id)
# Returns: tensor([12, 56, 99, 3])
```

### 2. Generative Retrieval Engine

**Purpose**: Generate candidate ad semantic IDs given user history

**Architecture**:
- **Encoder**: Processes user history and context
- **Decoder**: Autoregressively generates semantic ID tokens
- **Constrained Beam Search**: Uses prefix trie to prevent hallucination

**Key Files**:
- `src/generative_retrieval/generative_model.py`
- `src/generative_retrieval/prefix_trie.py`

**Usage**:
```python
from src.generative_retrieval import GenerativeRetrievalModel, SemanticIDTrie

model = GenerativeRetrievalModel(num_quantizers=4, codebook_size=256)
trie = SemanticIDTrie.load('./data/semantic_trie.pkl')

# Generate candidate ads
generated_ids, scores = model.generate(
    semantic_ids=user_history,
    demographic_ids=demographics,
    context_features=context,
    trie=trie,
    num_beams=10,
    num_return=50
)
```

### 3. Unified Ranking Layer (DLRM)

**Purpose**: Score retrieved candidates for multiple objectives

**Architecture**:
- **User Tower**: Processes user features
- **Ad Tower**: Processes ad features
- **Feature Interaction**: Cross-features between user and ad
- **Multi-Task Heads**: Separate predictions for each objective

**Key Files**:
- `src/ranking/dlrm.py`

**Usage**:
```python
from src.ranking import DLRM

model = DLRM(
    num_user_categorical=10,
    num_user_continuous=20,
    semantic_embed_dim=256,
    num_ad_categorical=5,
    num_ad_continuous=10,
    num_tasks=4
)

predictions = model(
    user_categorical,
    user_continuous,
    ad_semantic_embeddings,
    ad_categorical,
    ad_continuous
)
# Returns: {
#   'click': tensor([0.05, 0.12, ...]),
#   'conversion': tensor([0.002, 0.008, ...]),
#   'skip': tensor([0.1, 0.05, ...]),
#   'view_time': tensor([5.2, 8.1, ...])
# }
```

### 4. VCG Auction & Budget Pacing

**Purpose**: Economic engine for ad allocation and pricing

**VCG Auction**:
- Strategy-proof (truthful bidding is optimal)
- Efficient (maximizes social welfare)
- Charges "second price" generalized to multiple slots

**Budget Pacing**:
- Even pacing throughout the day
- Dynamic adjustment based on spend rate
- Forecast-based optimization

**Key Files**:
- `src/auction/vcg_auction.py`
- `src/auction/budget_pacing.py`

**Usage**:
```python
from src.auction import VCGAuction, BudgetTracker, BudgetPacer

# Run auction
auction = VCGAuction(reserve_price=0.01)
outcomes = auction.run_auction(bids, num_slots=3)

# Manage budgets
tracker = BudgetTracker()
pacer = BudgetPacer(tracker)

# Apply pacing to bids
adjusted_bid = pacer.apply_pacing_to_bid(advertiser_id, original_bid)
```

### 5. Feedback Loop & Online Learning

**Purpose**: Real-time model updates based on user interactions

**Components**:
- **Event Buffer**: Stores recent events
- **Feature Store**: Real-time feature aggregations
- **Online Trainer**: Incremental model updates
- **Model Registry**: Version management

**Key Files**:
- `src/feedback/online_learning.py`

**Usage**:
```python
from src.feedback import FeedbackLoop, FeedbackEvent

loop = FeedbackLoop()
loop.start()

# Record event
event = FeedbackEvent(
    event_id="evt_123",
    user_id="user123",
    ad_id="ad_456",
    event_type="click",
    timestamp=datetime.now()
)
loop.record_event(event)

# Get metrics
metrics = loop.compute_online_metrics()
```

## Configuration

All system parameters are configurable via `configs/default_config.yaml`:

```yaml
models:
  rqvae:
    num_quantizers: 4
    codebook_size: 256
    embedding_dim: 256

  generative_retrieval:
    num_encoder_layers: 6
    num_decoder_layers: 6
    embed_dim: 512

  ranking:
    num_tasks: 4
    num_experts: 4

training:
  rqvae:
    batch_size: 128
    learning_rate: 0.0001

serving:
  num_retrieval_candidates: 50
  num_final_ads: 3

auction:
  reserve_price: 0.01

budget_pacing:
  default_strategy: even
```

## API Reference

### Serve Ads

**Endpoint**: `POST /serve_ads`

**Request**:
```json
{
  "user_id": "string",
  "user_history": [[int, int, int, int], ...],
  "demographics": {"key": int, ...},
  "context": {"key": float, ...},
  "num_ads": int
}
```

**Response**:
```json
{
  "request_id": "string",
  "ads": [
    {
      "ad_id": "string",
      "advertiser_id": "string",
      "title": "string",
      "description": "string",
      "image_url": "string",
      "rank": int,
      "price": float
    }
  ],
  "timestamp": "string"
}
```

### Record Feedback

**Endpoint**: `POST /feedback`

**Request**:
```json
{
  "request_id": "string",
  "user_id": "string",
  "ad_id": "string",
  "event_type": "click|conversion|skip",
  "value": float
}
```

### Get Metrics

**Endpoint**: `GET /metrics`

**Response**:
```json
{
  "feedback_metrics": {...},
  "online_metrics": {
    "ctr": float,
    "cvr": float,
    "skip_rate": float
  },
  "budget_states": {...}
}
```

## Performance Optimization

### Model Optimization

1. **Quantization**: Use INT8 quantization for faster inference
2. **TorchScript**: Compile models with TorchScript
3. **ONNX**: Export to ONNX for deployment
4. **Batching**: Batch requests for throughput

### System Optimization

1. **Caching**: Cache semantic embeddings and trie lookups
2. **Async Processing**: Use async I/O for non-blocking operations
3. **Load Balancing**: Distribute requests across multiple servers
4. **GPU Acceleration**: Use CUDA for model inference

## Monitoring & Logging

### Metrics to Track

- **Model Metrics**: AUC, NDCG, Calibration Error
- **Business Metrics**: CTR, CVR, Revenue, ROI
- **System Metrics**: Latency, Throughput, Error Rate

### Logging

Logs are written to `./logs/` with the following structure:

```
logs/
├── server.log        # Server logs
├── training.log      # Training logs
├── metrics.log       # Metrics logs
└── errors.log        # Error logs
```

## Testing

Run unit tests:

```bash
pytest tests/
```

Run integration tests:

```bash
pytest tests/ -m integration
```

## Deployment

### Docker

Build Docker image:

```bash
docker build -t ads-system:latest .
```

Run container:

```bash
docker run -p 8000:8000 ads-system:latest
```

### Kubernetes

Deploy to Kubernetes:

```bash
kubectl apply -f k8s/deployment.yaml
```

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Citations

This system is inspired by and builds upon research from:

1. **Generative Retrieval**:
   - "Autoregressive Entity Retrieval" (TIGER)
   - "Transformer Memory as a Differentiable Search Index" (DSI)

2. **Semantic Indexing**:
   - "Residual Vector Quantization" (Lee et al.)
   - "RQ-VAE" papers

3. **Ranking**:
   - "Deep Learning Recommendation Model for Personalization" (DLRM)
   - "Modeling Task Relationships in Multi-task Learning" (MMoE)

4. **Auction Theory**:
   - Vickrey-Clarke-Groves mechanism
   - "Internet Advertising and the Generalized Second-Price Auction"

5. **Budget Pacing**:
   - "Budget Pacing for Targeted Online Advertisements at LinkedIn"

## Contact

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ for modern advertising systems**
