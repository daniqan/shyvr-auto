# Token Architecture Decision Record

**Date**: August 2024  
**Status**: Implemented  
**Decision**: Maintain separate TokenConfig and DiscoveredToken with converter utility

## Context

The system has evolved with two parallel token representations:
- `DiscoveredToken` for runtime token discovery and ML/RL operations
- `TokenConfig` for corpus collection and static configuration

This created potential technical debt and compatibility issues between corpus collection and ML systems.

## Decision

Maintain both token types with a converter utility for compatibility.

### Rationale

1. **Semantic Correctness**: Each token type serves a distinct purpose
   - `TokenConfig`: Static configuration for known tokens in corpus collection
   - `DiscoveredToken`: Runtime representation with discovery metadata

2. **No Breaking Changes**: Existing systems continue to work without modification

3. **Clear Migration Path**: The converter provides a bridge between systems

## Implementation

Created `src/data_pipeline/token_converter.py` with bidirectional conversion:
- `token_config_to_discovered()`: For using corpus tokens in ML/RL systems
- `discovered_to_token_config()`: For adding discovered tokens to corpus

## Consequences

### Positive
- Clean separation of concerns
- No immediate refactoring required
- Flexibility for future evolution

### Negative
- Two token representations to maintain
- Converter adds slight complexity

## Future Considerations

Long-term migration to unified token hierarchy:
- Base `Token` class with core attributes
- `ConfiguredToken` for static configuration
- `DiscoveredToken` for runtime discovery
- `EnrichedToken` with full market/social data

This would eliminate duplication while maintaining semantic clarity.