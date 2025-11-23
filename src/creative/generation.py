"""
Creative Generation Module (Google Performance Max-style).

This module provides AI-powered creative generation for ads:
- Text generation (headlines, descriptions) using LLMs
- Image generation using diffusion models
- Multi-modal optimization

Based on:
- Google Performance Max with Gemini and Imagen 3
- https://blog.google/products/ads-commerce/get-creative-with-generative-ai-in-performance-max/
"""
from typing import List, Dict, Optional
from dataclasses import dataclass
import hashlib


@dataclass
class CreativeAssets:
    """
    Generated creative assets for an ad.

    Attributes:
        headlines: List of generated headlines
        long_headlines: List of long headlines
        descriptions: List of descriptions
        image_prompts: Generated prompts for image generation
        image_urls: URLs of generated images (if generated)
        sitelinks: Generated sitelinks
    """
    headlines: List[str]
    long_headlines: List[str]
    descriptions: List[str]
    image_prompts: List[str]
    image_urls: Optional[List[str]] = None
    sitelinks: Optional[List[Dict[str, str]]] = None

    def to_dict(self) -> Dict:
        return {
            'headlines': self.headlines,
            'long_headlines': self.long_headlines,
            'descriptions': self.descriptions,
            'image_prompts': self.image_prompts,
            'image_urls': self.image_urls or [],
            'sitelinks': self.sitelinks or []
        }


class CreativeGenerator:
    """
    AI-powered creative generation system.

    In production, this would integrate with:
    - OpenAI GPT-4 / Anthropic Claude / Google Gemini for text
    - DALL-E 3 / Stable Diffusion / Imagen for images

    For this implementation, we provide a mock/template system that
    can be easily extended with real LLM/image model APIs.
    """

    def __init__(
        self,
        llm_provider: str = 'mock',  # 'openai', 'anthropic', 'google', or 'mock'
        image_provider: str = 'mock',  # 'dalle3', 'stable-diffusion', 'imagen', or 'mock'
        max_headlines: int = 5,
        max_descriptions: int = 3
    ):
        self.llm_provider = llm_provider
        self.image_provider = image_provider
        self.max_headlines = max_headlines
        self.max_descriptions = max_descriptions

    def generate_text_assets(
        self,
        product_name: str,
        product_description: str,
        target_audience: Optional[str] = None,
        tone: str = 'professional',
        brand_voice: Optional[str] = None
    ) -> CreativeAssets:
        """
        Generate text assets (headlines, descriptions).

        Args:
            product_name: Name of product/service
            product_description: Description of product/service
            target_audience: Target audience description
            tone: Tone of voice ('professional', 'casual', 'urgent', 'friendly')
            brand_voice: Optional brand voice guidelines

        Returns:
            CreativeAssets with generated text
        """
        if self.llm_provider == 'mock':
            return self._generate_mock_text_assets(
                product_name,
                product_description,
                target_audience,
                tone
            )

        # In production, call real LLM API here
        # Example for OpenAI:
        # response = openai.ChatCompletion.create(...)

        raise NotImplementedError(f"LLM provider {self.llm_provider} not implemented")

    def generate_image_prompts(
        self,
        product_name: str,
        product_description: str,
        style: str = 'modern',
        reference_images: Optional[List[str]] = None
    ) -> List[str]:
        """
        Generate image prompts for image generation.

        Args:
            product_name: Product name
            product_description: Product description
            style: Visual style ('modern', 'minimalist', 'vibrant', 'professional')
            reference_images: Optional reference image URLs for style matching

        Returns:
            List of image generation prompts
        """
        prompts = [
            f"{product_name} product photography, {style} style, high quality, professional lighting",
            f"{product_description}, {style} aesthetic, clean background, centered composition",
            f"Advertisement for {product_name}, {style} design, eye-catching, professional"
        ]

        if reference_images:
            prompts.append(
                f"{product_name} in the style of reference images, {style}, consistent branding"
            )

        return prompts[:3]  # Return top 3 prompts

    def generate_sitelinks(
        self,
        product_category: str,
        website_url: str
    ) -> List[Dict[str, str]]:
        """
        Generate sitelink suggestions.

        Args:
            product_category: Category of product
            website_url: Base website URL

        Returns:
            List of sitelink dicts with 'text', 'description', 'url'
        """
        # Common sitelink patterns based on category
        sitelink_templates = {
            'ecommerce': [
                {'text': 'Shop Now', 'description': 'Browse our full collection', 'url': '/shop'},
                {'text': 'Sale', 'description': 'Limited time offers', 'url': '/sale'},
                {'text': 'New Arrivals', 'description': 'See what\'s new', 'url': '/new'},
                {'text': 'Reviews', 'description': 'See customer reviews', 'url': '/reviews'}
            ],
            'service': [
                {'text': 'Get Quote', 'description': 'Request a free quote', 'url': '/quote'},
                {'text': 'Services', 'description': 'View all services', 'url': '/services'},
                {'text': 'Contact Us', 'description': 'Get in touch', 'url': '/contact'},
                {'text': 'About', 'description': 'Learn more about us', 'url': '/about'}
            ],
            'saas': [
                {'text': 'Free Trial', 'description': 'Start your free trial', 'url': '/trial'},
                {'text': 'Pricing', 'description': 'View pricing plans', 'url': '/pricing'},
                {'text': 'Features', 'description': 'Explore features', 'url': '/features'},
                {'text': 'Demo', 'description': 'Schedule a demo', 'url': '/demo'}
            ]
        }

        category = product_category.lower()
        if 'shop' in category or 'store' in category or 'ecommerce' in category:
            templates = sitelink_templates['ecommerce']
        elif 'software' in category or 'saas' in category or 'app' in category:
            templates = sitelink_templates['saas']
        else:
            templates = sitelink_templates['service']

        # Add full URLs
        base_url = website_url.rstrip('/')
        for link in templates:
            link['url'] = base_url + link['url']

        return templates

    def _generate_mock_text_assets(
        self,
        product_name: str,
        product_description: str,
        target_audience: Optional[str] = None,
        tone: str = 'professional'
    ) -> CreativeAssets:
        """
        Mock text generation for testing.

        In production, this would be replaced with real LLM calls.
        """
        # Generate headlines
        headlines = [
            f"{product_name} - {self._get_tone_prefix(tone)} Solution",
            f"Discover {product_name} Today",
            f"{product_name}: {self._extract_key_benefit(product_description)}",
            f"Get Started with {product_name}",
            f"{product_name} | {self._get_tone_suffix(tone)}"
        ][:self.max_headlines]

        # Generate long headlines
        long_headlines = [
            f"{product_name} - The {self._get_tone_prefix(tone)} Way to {self._extract_action(product_description)}",
            f"Transform Your Business with {product_name} - {self._extract_key_benefit(product_description)}"
        ]

        # Generate descriptions
        descriptions = [
            f"{product_description[:90]}... Learn more today!",
            f"Experience the power of {product_name}. {self._extract_key_benefit(product_description)}.",
            f"{self._get_audience_prefix(target_audience)}{product_name} delivers results."
        ][:self.max_descriptions]

        # Generate image prompts
        image_prompts = self.generate_image_prompts(product_name, product_description)

        return CreativeAssets(
            headlines=headlines,
            long_headlines=long_headlines,
            descriptions=descriptions,
            image_prompts=image_prompts
        )

    @staticmethod
    def _get_tone_prefix(tone: str) -> str:
        """Get prefix based on tone."""
        tone_map = {
            'professional': 'Professional',
            'casual': 'Easy',
            'urgent': 'Limited Time',
            'friendly': 'Your Friendly'
        }
        return tone_map.get(tone.lower(), 'Best')

    @staticmethod
    def _get_tone_suffix(tone: str) -> str:
        """Get suffix based on tone."""
        tone_map = {
            'professional': 'Trusted by Professionals',
            'casual': 'Simple & Effective',
            'urgent': 'Act Now',
            'friendly': 'We\'re Here to Help'
        }
        return tone_map.get(tone.lower(), 'Try It Today')

    @staticmethod
    def _extract_key_benefit(description: str) -> str:
        """Extract key benefit from description (simple heuristic)."""
        words = description.split()
        return ' '.join(words[:5]) if len(words) >= 5 else description

    @staticmethod
    def _extract_action(description: str) -> str:
        """Extract action verbs from description."""
        # Simple extraction - in production, use NLP
        action_words = ['grow', 'improve', 'increase', 'enhance', 'optimize', 'transform']
        for word in action_words:
            if word in description.lower():
                return word.capitalize() + ' Your Business'
        return 'Achieve More'

    @staticmethod
    def _get_audience_prefix(audience: Optional[str]) -> str:
        """Get audience-specific prefix."""
        if audience:
            return f"For {audience}: "
        return ""


class CreativeOptimizer:
    """
    Optimizes creative assets based on performance data.

    Learns which creatives perform best and suggests improvements.
    """

    def __init__(self):
        self.performance_data: Dict[str, Dict] = {}

    def record_performance(
        self,
        creative_id: str,
        metrics: Dict[str, float]
    ):
        """
        Record performance for a creative.

        Args:
            creative_id: Unique creative identifier
            metrics: Performance metrics (CTR, CVR, etc.)
        """
        if creative_id not in self.performance_data:
            self.performance_data[creative_id] = {
                'impressions': 0,
                'clicks': 0,
                'conversions': 0
            }

        data = self.performance_data[creative_id]
        data['impressions'] += metrics.get('impressions', 0)
        data['clicks'] += metrics.get('clicks', 0)
        data['conversions'] += metrics.get('conversions', 0)

    def get_top_performers(
        self,
        metric: str = 'ctr',
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get top performing creatives.

        Args:
            metric: Metric to rank by ('ctr', 'cvr')
            top_k: Number of top creatives to return

        Returns:
            List of (creative_id, score) tuples
        """
        scores = []

        for creative_id, data in self.performance_data.items():
            if metric == 'ctr':
                score = data['clicks'] / max(data['impressions'], 1)
            elif metric == 'cvr':
                score = data['conversions'] / max(data['impressions'], 1)
            else:
                score = 0.0

            scores.append((creative_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def suggest_improvements(
        self,
        creative_id: str
    ) -> List[str]:
        """
        Suggest improvements for a creative based on top performers.

        Args:
            creative_id: Creative to improve

        Returns:
            List of improvement suggestions
        """
        suggestions = []

        if creative_id not in self.performance_data:
            return ["Not enough data to suggest improvements"]

        data = self.performance_data[creative_id]
        ctr = data['clicks'] / max(data['impressions'], 1)

        # Get average CTR of top performers
        top_performers = self.get_top_performers('ctr', top_k=10)
        if top_performers:
            avg_top_ctr = sum(score for _, score in top_performers) / len(top_performers)

            if ctr < avg_top_ctr * 0.5:
                suggestions.append("Consider rewriting headline to be more compelling")
                suggestions.append("Try a different tone or call-to-action")
                suggestions.append("Test new image variations")

        return suggestions if suggestions else ["Creative is performing well!"]


def create_hash_for_creative(assets: CreativeAssets) -> str:
    """
    Create a unique hash for creative assets.

    Args:
        assets: Creative assets

    Returns:
        Hash string
    """
    content = (
        ''.join(assets.headlines) +
        ''.join(assets.descriptions) +
        ''.join(assets.image_prompts)
    )

    return hashlib.md5(content.encode()).hexdigest()
