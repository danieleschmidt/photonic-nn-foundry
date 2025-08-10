"""
Internationalization (I18n) support for Photonic Foundry.
Provides multi-language support with regional compliance integration.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from functools import lru_cache

logger = logging.getLogger(__name__)

# Supported languages and their regional associations
SUPPORTED_LANGUAGES = {
    'en': {
        'name': 'English',
        'regions': ['us-east-1', 'us-west-2', 'global'],
        'compliance': ['ccpa'],
        'rtl': False
    },
    'es': {
        'name': 'Español',
        'regions': ['us-east-1', 'us-west-2'],
        'compliance': ['ccpa'],
        'rtl': False
    },
    'fr': {
        'name': 'Français',
        'regions': ['eu-west-1', 'eu-central-1'],
        'compliance': ['gdpr'],
        'rtl': False
    },
    'de': {
        'name': 'Deutsch',
        'regions': ['eu-central-1', 'eu-west-1'],
        'compliance': ['gdpr'],
        'rtl': False
    },
    'ja': {
        'name': '日本語',
        'regions': ['ap-northeast-1', 'ap-southeast-1'],
        'compliance': ['pdpa'],
        'rtl': False
    },
    'zh': {
        'name': '中文',
        'regions': ['ap-southeast-1', 'ap-northeast-1'],
        'compliance': ['pdpa'],
        'rtl': False
    }
}

# Default language fallback chain
DEFAULT_FALLBACK_CHAIN = ['en']

class I18nManager:
    """
    Manages internationalization for the Photonic Foundry application.
    Handles language detection, translation loading, and regional compliance.
    """
    
    def __init__(self, default_language: str = 'en', region: Optional[str] = None):
        self.default_language = default_language
        self.region = region
        self.translations_cache: Dict[str, Dict[str, Any]] = {}
        self.translations_dir = Path(__file__).parent / 'translations'
        
        # Ensure translations directory exists
        self.translations_dir.mkdir(exist_ok=True)
        
        # Load all available translations
        self._load_all_translations()
    
    def _load_all_translations(self) -> None:
        """Load all translation files into cache."""
        for lang_code in SUPPORTED_LANGUAGES.keys():
            try:
                self._load_translation(lang_code)
            except Exception as e:
                logger.warning(f"Failed to load translation for {lang_code}: {e}")
    
    def _load_translation(self, lang_code: str) -> Dict[str, Any]:
        """Load translation file for a specific language."""
        if lang_code in self.translations_cache:
            return self.translations_cache[lang_code]
        
        translation_file = self.translations_dir / f"{lang_code}.json"
        
        if not translation_file.exists():
            logger.warning(f"Translation file not found for {lang_code}: {translation_file}")
            return {}
        
        try:
            with open(translation_file, 'r', encoding='utf-8') as f:
                translations = json.load(f)
                self.translations_cache[lang_code] = translations
                logger.info(f"Loaded {len(translations)} translations for {lang_code}")
                return translations
        except Exception as e:
            logger.error(f"Error loading translation file {translation_file}: {e}")
            return {}
    
    @lru_cache(maxsize=1000)
    def get_text(self, key: str, lang_code: Optional[str] = None, **kwargs) -> str:
        """
        Get translated text for a given key.
        
        Args:
            key: Translation key (e.g., 'quantum.optimization.started')
            lang_code: Language code (defaults to instance default)
            **kwargs: Variables for string formatting
        
        Returns:
            Translated and formatted text
        """
        if lang_code is None:
            lang_code = self.default_language
        
        # Get the translation
        text = self._get_translation(key, lang_code)
        
        # Format with provided variables
        if kwargs:
            try:
                text = text.format(**kwargs)
            except (KeyError, ValueError) as e:
                logger.warning(f"Error formatting translation '{key}' for {lang_code}: {e}")
        
        return text
    
    def _get_translation(self, key: str, lang_code: str) -> str:
        """Get translation with fallback logic."""
        # Try primary language
        if lang_code in self.translations_cache:
            text = self._extract_nested_key(self.translations_cache[lang_code], key)
            if text:
                return text
        
        # Try fallback languages
        fallback_chain = self._get_fallback_chain(lang_code)
        for fallback_lang in fallback_chain:
            if fallback_lang in self.translations_cache:
                text = self._extract_nested_key(self.translations_cache[fallback_lang], key)
                if text:
                    logger.debug(f"Using fallback {fallback_lang} for key '{key}'")
                    return text
        
        # Return key if no translation found
        logger.warning(f"No translation found for key '{key}' in any language")
        return f"[{key}]"
    
    def _extract_nested_key(self, translations: Dict[str, Any], key: str) -> Optional[str]:
        """Extract value from nested dictionary using dot notation."""
        keys = key.split('.')
        current = translations
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return None
        
        return current if isinstance(current, str) else None
    
    def _get_fallback_chain(self, lang_code: str) -> List[str]:
        """Get fallback language chain for a given language."""
        fallback_chain = []
        
        # Regional fallback
        if self.region and lang_code in SUPPORTED_LANGUAGES:
            lang_info = SUPPORTED_LANGUAGES[lang_code]
            if self.region in lang_info['regions']:
                # Add related languages from the same region
                for other_lang, other_info in SUPPORTED_LANGUAGES.items():
                    if (other_lang != lang_code and 
                        self.region in other_info['regions'] and 
                        other_lang not in fallback_chain):
                        fallback_chain.append(other_lang)
        
        # Add default fallback chain
        fallback_chain.extend(DEFAULT_FALLBACK_CHAIN)
        
        # Remove duplicates while preserving order
        seen = set()
        return [x for x in fallback_chain if not (x in seen or seen.add(x))]
    
    def get_supported_languages_for_region(self, region: str) -> List[Dict[str, Any]]:
        """Get supported languages for a specific region."""
        languages = []
        
        for lang_code, lang_info in SUPPORTED_LANGUAGES.items():
            if region in lang_info['regions'] or region == 'global':
                languages.append({
                    'code': lang_code,
                    'name': lang_info['name'],
                    'rtl': lang_info['rtl'],
                    'compliance': lang_info['compliance']
                })
        
        return languages
    
    def detect_language_from_region(self, region: str) -> str:
        """Detect primary language for a region."""
        region_language_map = {
            'us-east-1': 'en',
            'us-west-2': 'en',
            'eu-west-1': 'en',  # English as primary, but supports others
            'eu-central-1': 'de',
            'ap-southeast-1': 'en',  # English as lingua franca
            'ap-northeast-1': 'ja',
            'global': 'en'
        }
        
        return region_language_map.get(region, 'en')
    
    def get_compliance_messages(self, compliance_framework: str, lang_code: str) -> Dict[str, str]:
        """Get compliance-specific messages for a language."""
        compliance_key = f"compliance.{compliance_framework}"
        
        if lang_code not in self.translations_cache:
            lang_code = self.default_language
        
        translations = self.translations_cache.get(lang_code, {})
        compliance_messages = self._extract_nested_key(translations, compliance_key)
        
        if isinstance(compliance_messages, dict):
            return compliance_messages
        
        return {}
    
    def reload_translations(self) -> None:
        """Reload all translation files from disk."""
        self.translations_cache.clear()
        self.get_text.cache_clear()
        self._load_all_translations()
        logger.info("All translations reloaded")

# Global instance
_global_i18n: Optional[I18nManager] = None

def get_i18n_manager(default_language: str = 'en', region: Optional[str] = None) -> I18nManager:
    """Get or create global I18n manager instance."""
    global _global_i18n
    
    if _global_i18n is None:
        # Detect region from environment if not provided
        if region is None:
            region = os.getenv('REGION', 'global')
        
        _global_i18n = I18nManager(default_language, region)
    
    return _global_i18n

def _(key: str, lang_code: Optional[str] = None, **kwargs) -> str:
    """Shorthand function for getting translated text."""
    return get_i18n_manager().get_text(key, lang_code, **kwargs)

def get_supported_languages() -> Dict[str, Dict[str, Any]]:
    """Get all supported languages."""
    return SUPPORTED_LANGUAGES.copy()

def is_rtl_language(lang_code: str) -> bool:
    """Check if a language uses right-to-left writing."""
    return SUPPORTED_LANGUAGES.get(lang_code, {}).get('rtl', False)