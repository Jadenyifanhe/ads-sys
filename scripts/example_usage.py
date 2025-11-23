"""
Example usage of the Generative Ads System.

This script demonstrates how to use all components of the system.
"""
import torch
import numpy as np
from datetime import datetime
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from semantic_indexer import RQVAE
from generative_retrieval import GenerativeRetrievalModel, SemanticIDTrie, build_trie_from_ads
from ranking import DLRM
from auction import VCGAuction, Bid, BudgetTracker, BudgetPacer
from feedback import FeedbackLoop, FeedbackEvent


def example_semantic_indexing():
    """Example: Convert ads to semantic IDs."""
    print("=" * 80)
    print("Example 1: Semantic Indexing with RQ-VAE")
    print("=" * 80)

    # Create model
    model = RQVAE(
        text_dim=768,
        image_dim=512,
        category_vocab_size=100,
        embedding_dim=256,
        num_quantizers=4,
        codebook_size=256
    )

    # Mock ad features
    text_features = torch.randn(5, 768)  # 5 ads
    image_features = torch.randn(5, 512)
    category_ids = torch.randint(0, 100, (5,))

    # Get semantic IDs
    semantic_ids = model.get_semantic_id(text_features, image_features, category_ids)

    print(f"\nGenerated Semantic IDs for 5 ads:")
    for i, sid in enumerate(semantic_ids):
        print(f"  Ad {i+1}: {tuple(sid.tolist())}")

    return semantic_ids


def example_generative_retrieval():
    """Example: Generate candidate ads using generative retrieval."""
    print("\n" + "=" * 80)
    print("Example 2: Generative Retrieval")
    print("=" * 80)

    # Create model
    model = GenerativeRetrievalModel(
        num_quantizers=4,
        codebook_size=256,
        embed_dim=512
    )

    # Build trie (mock)
    print("\nBuilding semantic ID trie...")
    semantic_ids = [(i, i+1, i+2, i+3) for i in range(100)]
    ad_ids = [f"ad_{i}" for i in range(100)]
    trie = build_trie_from_ads(semantic_ids, ad_ids, num_quantizers=4, codebook_size=256)

    print(f"Trie contains {len(trie)} semantic IDs")

    # Mock user history
    user_history = torch.LongTensor([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ]).unsqueeze(0)  # [1, 3, 4]

    demographics = torch.randint(0, 50, (1, 5))
    context = torch.randn(1, 32)

    # Generate candidates
    print("\nGenerating ad candidates...")
    generated_ids, scores = model.generate(
        semantic_ids=user_history,
        demographic_ids=demographics,
        context_features=context,
        trie=trie,
        num_beams=5,
        num_return=10
    )

    print(f"\nTop 10 generated candidates:")
    for i in range(min(10, generated_ids.shape[1])):
        semantic_id = tuple(generated_ids[0, i].tolist())
        score = scores[0, i].item()
        ad_id = trie.search(semantic_id)
        print(f"  {i+1}. {semantic_id} -> {ad_id} (score: {score:.4f})")


def example_ranking():
    """Example: Rank candidates with DLRM."""
    print("\n" + "=" * 80)
    print("Example 3: Unified Ranking with DLRM")
    print("=" * 80)

    # Create model
    model = DLRM(
        num_user_categorical=10,
        num_user_continuous=20,
        semantic_embed_dim=256,
        num_ad_categorical=5,
        num_ad_continuous=10,
        num_tasks=4
    )

    # Mock features for 10 candidates
    batch_size = 10
    user_cat = torch.randint(0, 100, (batch_size, 10))
    user_cont = torch.randn(batch_size, 20)
    ad_sem = torch.randn(batch_size, 256)
    ad_cat = torch.randint(0, 100, (batch_size, 5))
    ad_cont = torch.randn(batch_size, 10)

    # Get predictions
    print("\nRanking candidates...")
    predictions = model(user_cat, user_cont, ad_sem, ad_cat, ad_cont)

    print(f"\nRanking results for {batch_size} candidates:")
    print(f"{'Rank':<6} {'P(Click)':<12} {'P(Convert)':<12} {'P(Skip)':<12} {'View Time':<12}")
    print("-" * 60)

    # Sort by click probability
    click_probs = predictions['click'].detach().numpy()
    sorted_indices = np.argsort(-click_probs)

    for rank, idx in enumerate(sorted_indices[:10], 1):
        print(f"{rank:<6} "
              f"{predictions['click'][idx].item():<12.4f} "
              f"{predictions['conversion'][idx].item():<12.4f} "
              f"{predictions['skip'][idx].item():<12.4f} "
              f"{predictions['view_time'][idx].item():<12.2f}")


def example_auction():
    """Example: Run VCG auction."""
    print("\n" + "=" * 80)
    print("Example 4: VCG Auction")
    print("=" * 80)

    # Create auction
    auction = VCGAuction(reserve_price=0.10)

    # Create bids
    bids = [
        Bid("ad_1", "adv_1", 5.00, "cpm", 0.9, 0.05, 0.01),
        Bid("ad_2", "adv_2", 4.50, "cpm", 0.95, 0.06, 0.012),
        Bid("ad_3", "adv_3", 6.00, "cpm", 0.85, 0.04, 0.008),
        Bid("ad_4", "adv_4", 3.00, "cpc", 1.0, 0.08, 0.015),
        Bid("ad_5", "adv_5", 5.50, "cpm", 0.88, 0.055, 0.011),
    ]

    # Run auction for 3 slots
    print("\nRunning VCG auction for 3 ad slots...")
    outcomes = auction.run_auction(bids, num_slots=3)

    print(f"\nAuction results:")
    print(f"{'Rank':<6} {'Ad ID':<10} {'Bid Price':<12} {'VCG Price':<12} {'Expected Value':<15}")
    print("-" * 60)

    for outcome in outcomes:
        print(f"{outcome.rank:<6} "
              f"{outcome.ad_id:<10} "
              f"${outcome.bid_price:<11.2f} "
              f"${outcome.vcg_price:<11.2f} "
              f"${outcome.expected_value:<14.2f}")


def example_budget_pacing():
    """Example: Budget pacing."""
    print("\n" + "=" * 80)
    print("Example 5: Budget Pacing")
    print("=" * 80)

    # Create budget tracker
    tracker = BudgetTracker()
    pacer = BudgetPacer(tracker)

    # Register advertisers
    from auction import BudgetConfig

    config = BudgetConfig(
        advertiser_id="adv_1",
        daily_budget=1000.0,
        total_budget=30000.0,
        start_date=datetime.now(),
        end_date=datetime.now(),
        pacing_strategy="even"
    )
    tracker.register_budget(config)

    # Simulate spending
    print("\nSimulating budget pacing throughout the day...")
    print(f"{'Hour':<6} {'Spent':<12} {'Remaining':<12} {'Pacing Mult':<15} {'Adjusted Bid':<15}")
    print("-" * 70)

    original_bid = 5.00
    state = tracker.get_budget_state("adv_1")

    for hour in range(0, 24, 2):
        # Simulate some spending
        amount = np.random.uniform(30, 60)
        state.spent_today += amount

        # Compute pacing multiplier
        multiplier = pacer.compute_pacing_multiplier("adv_1")
        adjusted_bid = original_bid * multiplier

        print(f"{hour:<6} "
              f"${state.spent_today:<11.2f} "
              f"${state.remaining_budget():<11.2f} "
              f"{multiplier:<14.3f} "
              f"${adjusted_bid:<14.2f}")


def example_feedback_loop():
    """Example: Feedback loop and online learning."""
    print("\n" + "=" * 80)
    print("Example 6: Feedback Loop")
    print("=" * 80)

    # Create feedback loop
    loop = FeedbackLoop()
    loop.start()

    # Simulate events
    print("\nRecording user interactions...")
    events = [
        ("user_1", "ad_1", "impression"),
        ("user_1", "ad_1", "click"),
        ("user_2", "ad_2", "impression"),
        ("user_2", "ad_2", "skip"),
        ("user_3", "ad_3", "impression"),
        ("user_3", "ad_3", "click"),
        ("user_3", "ad_3", "conversion"),
    ]

    for i, (user_id, ad_id, event_type) in enumerate(events):
        event = FeedbackEvent(
            event_id=f"evt_{i}",
            user_id=user_id,
            ad_id=ad_id,
            event_type=event_type,
            timestamp=datetime.now(),
            value=1.0 if event_type == "conversion" else 0.0
        )
        loop.record_event(event)
        print(f"  Recorded: {user_id} -> {ad_id} ({event_type})")

    # Compute metrics
    import time
    time.sleep(0.5)  # Let events process

    metrics = loop.compute_online_metrics()
    print(f"\nOnline Metrics:")
    print(f"  Impressions: {metrics.get('impressions', 0)}")
    print(f"  Clicks: {metrics.get('clicks', 0)}")
    print(f"  Conversions: {metrics.get('conversions', 0)}")
    print(f"  CTR: {metrics.get('ctr', 0):.2%}")
    print(f"  CVR: {metrics.get('cvr', 0):.2%}")

    loop.stop()


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("GENERATIVE ADS SYSTEM - EXAMPLE USAGE")
    print("=" * 80)

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Run examples
    example_semantic_indexing()
    example_generative_retrieval()
    example_ranking()
    example_auction()
    example_budget_pacing()
    example_feedback_loop()

    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
