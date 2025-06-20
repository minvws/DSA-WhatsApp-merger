try:
    from .Mapping import Mapping
    from .Parser import Parser
    from .WhatsAppMerger import WhatsAppMerger
except ImportError:
    from Mapping import Mapping
    from Parser import Parser
    from WhatsAppMerger import WhatsAppMerger