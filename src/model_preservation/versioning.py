"""
Semantic Versioning Implementation for Model Preservation System

This module provides semantic versioning functionality following the SemVer 2.0.0 specification.
It supports version parsing, validation, comparison, and increment operations for model versions.

Based on https://semver.org/spec/v2.0.0.html
"""

import re
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass

from .base import VersionError


# Regex pattern for semantic version parsing (SemVer 2.0.0 compliant)
SEMVER_PATTERN = re.compile(
    r'^[vV]?'  # Optional 'v' or 'V' prefix
    r'(?P<major>0|[1-9]\d*)'  # Major version
    r'\.'
    r'(?P<minor>0|[1-9]\d*)'  # Minor version
    r'\.'
    r'(?P<patch>0|[1-9]\d*)'  # Patch version
    r'(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?'  # Optional prerelease
    r'(?:\+(?P<build>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$'  # Optional build metadata
)


@dataclass(frozen=True)
class SemanticVersion:
    """
    Semantic version implementation following SemVer 2.0.0 specification.
    
    A semantic version consists of three numeric parts (major.minor.patch)
    with optional prerelease and build metadata identifiers.
    
    Examples:
        - 1.0.0
        - 2.1.3-alpha
        - 1.0.0-beta.1+build.123
        - v3.2.1 (v prefix is normalized away)
    
    Attributes:
        major: Major version number (breaking changes)
        minor: Minor version number (backward compatible features)
        patch: Patch version number (backward compatible bug fixes)
        prerelease: Optional prerelease identifier (e.g., 'alpha', 'beta.1')
        build: Optional build metadata (ignored in version comparison)
    """
    major: int
    minor: int
    patch: int
    prerelease: Optional[str] = None
    build: Optional[str] = None
    
    def __init__(self, version_string: str):
        """
        Parse a semantic version string.
        
        Args:
            version_string: Version string to parse (e.g., "1.0.0", "v2.1.3-alpha")
            
        Raises:
            VersionError: If version string is invalid
        """
        if not isinstance(version_string, str):
            raise VersionError(f"Version must be a string, got {type(version_string)}")
        
        match = SEMVER_PATTERN.match(version_string.strip())
        if not match:
            raise VersionError(f"Invalid semantic version format: {version_string}")
        
        groups = match.groupdict()
        
        # Parse version components
        try:
            major = int(groups['major'])
            minor = int(groups['minor'])
            patch = int(groups['patch'])
        except ValueError:
            raise VersionError(f"Invalid semantic version format: {version_string}")
        
        # Validate non-negative version numbers
        if major < 0 or minor < 0 or patch < 0:
            raise VersionError(f"Invalid semantic version format: {version_string}")
        
        # Set attributes (bypassing frozen dataclass)
        object.__setattr__(self, 'major', major)
        object.__setattr__(self, 'minor', minor)
        object.__setattr__(self, 'patch', patch)
        object.__setattr__(self, 'prerelease', groups['prerelease'])
        object.__setattr__(self, 'build', groups['build'])
    
    def __str__(self) -> str:
        """Return string representation of the version."""
        version = f"{self.major}.{self.minor}.{self.patch}"
        
        if self.prerelease:
            version += f"-{self.prerelease}"
        
        if self.build:
            version += f"+{self.build}"
        
        return version
    
    def __repr__(self) -> str:
        """Return detailed string representation."""
        return f"SemanticVersion('{self}')"
    
    def __eq__(self, other) -> bool:
        """Test version equality (build metadata is ignored)."""
        if not isinstance(other, SemanticVersion):
            return False
        
        return (
            self.major == other.major and
            self.minor == other.minor and
            self.patch == other.patch and
            self.prerelease == other.prerelease
            # Build metadata is ignored per SemVer spec
        )
    
    def __hash__(self) -> int:
        """Return hash value (build metadata is ignored)."""
        return hash((self.major, self.minor, self.patch, self.prerelease))
    
    def __lt__(self, other) -> bool:
        """Test if this version is less than another."""
        if not isinstance(other, SemanticVersion):
            raise TypeError(f"Cannot compare SemanticVersion with {type(other)}")
        
        # Compare major.minor.patch first
        if (self.major, self.minor, self.patch) != (other.major, other.minor, other.patch):
            return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)
        
        # If major.minor.patch are equal, compare prerelease
        # Normal version has higher precedence than prerelease version
        if self.prerelease is None and other.prerelease is not None:
            return False  # 1.0.0 > 1.0.0-alpha
        
        if self.prerelease is not None and other.prerelease is None:
            return True   # 1.0.0-alpha < 1.0.0
        
        if self.prerelease is None and other.prerelease is None:
            return False  # Equal versions
        
        # Both have prerelease, compare them
        return self._compare_prerelease(self.prerelease, other.prerelease) < 0
    
    def __le__(self, other) -> bool:
        """Test if this version is less than or equal to another."""
        return self < other or self == other
    
    def __gt__(self, other) -> bool:
        """Test if this version is greater than another."""
        return not self <= other
    
    def __ge__(self, other) -> bool:
        """Test if this version is greater than or equal to another."""
        return not self < other
    
    def _compare_prerelease(self, pre1: str, pre2: str) -> int:
        """
        Compare prerelease versions according to SemVer specification.
        
        Returns:
            -1 if pre1 < pre2
             0 if pre1 == pre2
             1 if pre1 > pre2
        """
        if pre1 == pre2:
            return 0
        
        # Split prerelease identifiers by dots
        parts1 = pre1.split('.')
        parts2 = pre2.split('.')
        
        # Compare each part
        for i in range(max(len(parts1), len(parts2))):
            # Handle missing parts (shorter version has lower precedence)
            if i >= len(parts1):
                return -1  # pre1 has fewer parts, so it's smaller
            if i >= len(parts2):
                return 1   # pre2 has fewer parts, so it's smaller
            
            part1, part2 = parts1[i], parts2[i]
            
            # Numeric identifiers are compared numerically
            # Alphanumeric identifiers are compared lexically in ASCII sort order
            # Numeric identifiers always have lower precedence than non-numeric
            is_num1 = part1.isdigit()
            is_num2 = part2.isdigit()
            
            if is_num1 and is_num2:
                # Both numeric - compare as integers
                diff = int(part1) - int(part2)
                if diff != 0:
                    return -1 if diff < 0 else 1
            elif is_num1 and not is_num2:
                # Numeric has lower precedence than alphanumeric
                return -1
            elif not is_num1 and is_num2:
                # Alphanumeric has higher precedence than numeric
                return 1
            else:
                # Both alphanumeric - compare lexically
                if part1 < part2:
                    return -1
                elif part1 > part2:
                    return 1
        
        return 0  # All parts are equal
    
    def bump_major(self) -> 'SemanticVersion':
        """
        Return a new version with major version incremented.
        Minor and patch are reset to 0, prerelease and build are removed.
        """
        return SemanticVersion(f"{self.major + 1}.0.0")
    
    def bump_minor(self) -> 'SemanticVersion':
        """
        Return a new version with minor version incremented.
        Patch is reset to 0, prerelease and build are removed.
        """
        return SemanticVersion(f"{self.major}.{self.minor + 1}.0")
    
    def bump_patch(self) -> 'SemanticVersion':
        """
        Return a new version with patch version incremented.
        Prerelease and build are removed.
        """
        return SemanticVersion(f"{self.major}.{self.minor}.{self.patch + 1}")
    
    def is_prerelease(self) -> bool:
        """Check if this is a prerelease version."""
        return self.prerelease is not None
    
    def is_stable(self) -> bool:
        """
        Check if this is a stable version.
        A version is considered stable if it's not a prerelease and major >= 1.
        """
        return not self.is_prerelease() and self.major >= 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert version to dictionary representation."""
        return {
            'major': self.major,
            'minor': self.minor,
            'patch': self.patch,
            'prerelease': self.prerelease,
            'build': self.build
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SemanticVersion':
        """Create version from dictionary representation."""
        major = data['major']
        minor = data['minor']
        patch = data['patch']
        prerelease = data.get('prerelease')
        build = data.get('build')
        
        version_str = f"{major}.{minor}.{patch}"
        
        if prerelease:
            version_str += f"-{prerelease}"
        
        if build:
            version_str += f"+{build}"
        
        return cls(version_str)


# Utility functions
def is_valid_semantic_version(version_string: str) -> bool:
    """
    Check if a string is a valid semantic version.
    
    Args:
        version_string: String to validate
        
    Returns:
        True if valid semantic version, False otherwise
    """
    try:
        SemanticVersion(version_string)
        return True
    except VersionError:
        return False


def validate_version_components(major: int, minor: int, patch: int) -> bool:
    """
    Validate that version components are valid.
    
    Args:
        major: Major version number
        minor: Minor version number
        patch: Patch version number
        
    Returns:
        True if all components are valid (non-negative integers)
    """
    return (
        isinstance(major, int) and major >= 0 and
        isinstance(minor, int) and minor >= 0 and
        isinstance(patch, int) and patch >= 0
    )


def normalize_version_string(version_string: str) -> str:
    """
    Normalize a version string by removing 'v' prefix and ensuring proper format.
    
    Args:
        version_string: Version string to normalize
        
    Returns:
        Normalized version string
        
    Raises:
        VersionError: If version string is invalid
    """
    version = SemanticVersion(version_string)
    return str(version)


def parse_version_range(range_string: str) -> Dict[str, SemanticVersion]:
    """
    Parse a version range string (future enhancement).
    
    Examples:
        - ">=1.0.0" 
        - "~1.2.0" (patch-level changes)
        - "^1.2.0" (compatible within major version)
        
    Args:
        range_string: Version range to parse
        
    Returns:
        Dictionary describing the range constraints
        
    Note:
        This is a placeholder for future range parsing functionality.
    """
    # TODO: Implement version range parsing in future phases
    raise NotImplementedError("Version range parsing will be implemented in Phase 2.2")


def find_latest_version(versions: list[Union[str, SemanticVersion]], 
                       include_prerelease: bool = False) -> Optional[SemanticVersion]:
    """
    Find the latest version from a list of versions.
    
    Args:
        versions: List of version strings or SemanticVersion objects
        include_prerelease: Whether to include prerelease versions
        
    Returns:
        Latest version or None if no valid versions found
    """
    parsed_versions = []
    
    for version in versions:
        try:
            if isinstance(version, str):
                parsed_version = SemanticVersion(version)
            elif isinstance(version, SemanticVersion):
                parsed_version = version
            else:
                continue  # Skip invalid version types
            
            if include_prerelease or not parsed_version.is_prerelease():
                parsed_versions.append(parsed_version)
                
        except VersionError:
            continue  # Skip invalid versions
    
    return max(parsed_versions) if parsed_versions else None


def find_compatible_versions(target_version: Union[str, SemanticVersion],
                           available_versions: list[Union[str, SemanticVersion]],
                           compatibility_mode: str = "patch") -> list[SemanticVersion]:
    """
    Find versions compatible with a target version.
    
    Args:
        target_version: Target version for compatibility check
        available_versions: List of available versions
        compatibility_mode: Type of compatibility ("patch", "minor", "major")
        
    Returns:
        List of compatible versions sorted in descending order
        
    Raises:
        VersionError: If target version is invalid
        ValueError: If compatibility mode is invalid
    """
    if isinstance(target_version, str):
        target = SemanticVersion(target_version)
    else:
        target = target_version
    
    if compatibility_mode not in ("patch", "minor", "major"):
        raise ValueError(f"Invalid compatibility mode: {compatibility_mode}")
    
    compatible = []
    
    for version in available_versions:
        try:
            if isinstance(version, str):
                parsed_version = SemanticVersion(version)
            elif isinstance(version, SemanticVersion):
                parsed_version = version
            else:
                continue
            
            # Check compatibility based on mode
            if compatibility_mode == "patch":
                # Same major.minor, patch can be >= target
                is_compatible = (
                    parsed_version.major == target.major and
                    parsed_version.minor == target.minor and
                    parsed_version >= target
                )
            elif compatibility_mode == "minor":
                # Same major, minor can be >= target
                is_compatible = (
                    parsed_version.major == target.major and
                    parsed_version >= target
                )
            else:  # major
                # Any version >= target
                is_compatible = parsed_version >= target
            
            if is_compatible:
                compatible.append(parsed_version)
                
        except VersionError:
            continue  # Skip invalid versions
    
    return sorted(compatible, reverse=True)