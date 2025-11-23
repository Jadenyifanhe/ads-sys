# Quick Start Guide

Get up and running with the Generative Ads System in 5 minutes.

## 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ads-sys.git
cd ads-sys

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## 2. Run Example Usage

See all components in action:

```bash
python scripts/example_usage.py
```

This will demonstrate:
- ✅ Semantic indexing (converting ads to IDs)
- ✅ Generative retrieval (finding relevant ads)
- ✅ Ranking (scoring candidates)
- ✅ VCG auction (ad allocation)
- ✅ Budget pacing (spend management)
- ✅ Feedback loop (online learning)

## 3. Start the Ad Server (Development)

```bash
# Start the FastAPI server
python src/serving/ad_server.py
```

The server will be available at `http://localhost:8000`

## 4. Make Your First Request

```bash
# Request ads
curl -X POST http://localhost:8000/serve_ads \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "user_history": [[1, 2, 3, 4], [5, 6, 7, 8]],
    "demographics": {"age": 1, "gender": 0, "location": 5},
    "context": {"hour": 14.0, "device": 1.0},
    "num_ads": 3
  }'
```

## 5. Using Docker (Production)

```bash
# Build and run with Docker Compose
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f ad-server

# Stop
docker-compose down
```

## Next Steps

### Train Your Own Models

1. **Prepare your data**: See `src/data/schemas.py` for data formats

2. **Train RQ-VAE**:
```bash
python src/training/train_rqvae.py
```

3. **Build Semantic Trie**:
```python
from src.generative_retrieval import build_trie_from_ads

# Build trie from your ads
trie = build_trie_from_ads(semantic_ids, ad_ids, num_quantizers=4, codebook_size=256)
trie.save('./data/semantic_trie.pkl')
```

4. **Train Generative Model**:
```bash
python src/training/train_generative.py
```

5. **Train Ranking Model**:
```bash
python src/training/train_ranking.py
```

### Configure the System

Edit `configs/default_config.yaml` to customize:

- Model architectures
- Training hyperparameters
- Serving parameters
- Auction settings
- Budget pacing strategies

### Monitor Performance

Access metrics:

```bash
curl http://localhost:8000/metrics
```

With Docker Compose, access monitoring tools:

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

### API Documentation

Interactive API docs available at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Common Issues

### Issue: Models not found

**Solution**: Train models first or set paths to pre-trained models:

```bash
export RQVAE_PATH=/path/to/model.pt
export GENERATIVE_PATH=/path/to/model.pt
export RANKING_PATH=/path/to/model.pt
```

### Issue: Out of memory

**Solution**: Reduce batch sizes in config or use CPU:

```yaml
# configs/default_config.yaml
serving:
  device: cpu
  batch_size: 1
```

### Issue: Slow inference

**Solution**: Use GPU and enable optimizations:

```yaml
serving:
  device: cuda
  use_amp: true  # Automatic Mixed Precision
```

## Getting Help

- 📖 [Full Documentation](README.md)
- 🐛 [Report Issues](https://github.com/yourusername/ads-sys/issues)
- 💬 [Discussions](https://github.com/yourusername/ads-sys/discussions)

## What's Next?

- Read the [full README](README.md) for detailed documentation
- Explore the [source code](src/) to understand internals
- Check out [example usage](scripts/example_usage.py) for code samples
- Review [configuration options](configs/default_config.yaml)

---

**Happy ad serving! 🚀**
