"""
Tests for RequirementAnalyzer
"""
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from src.tools.analyzer import RequirementAnalyzer
from src.models.schemas import RequirementAnalysis


class TestRequirementAnalyzer:
    
    def setup_method(self):
        """Setup test fixtures"""
        self.analyzer = RequirementAnalyzer()
    
    def test_analyzer_initialization(self):
        """Test analyzer initializes correctly"""
        assert self.analyzer is not None
        assert "web" in self.analyzer.project_types
        assert "mobile" in self.analyzer.project_types
    
    def test_analyze_web_project(self):
        """Test analyzing a web project"""
        description = "Build a web application with user authentication and database"
        result = self.analyzer.analyze(description, "web")
        
        assert isinstance(result, RequirementAnalysis)
        assert len(result.requirements) > 0
        assert "must" in result.moscow_priority
        assert "should" in result.moscow_priority
        assert "could" in result.moscow_priority
        assert "wont" in result.moscow_priority
    
    def test_analyze_mobile_project(self):
        """Test analyzing a mobile project"""
        description = "Create a mobile app for iOS and Android with offline capability"
        result = self.analyzer.analyze(description, "mobile")
        
        assert isinstance(result, RequirementAnalysis)
        assert len(result.requirements) > 0
        assert len(result.gaps) >= 0
        assert len(result.suggestions) > 0
    
    def test_extract_requirements(self):
        """Test requirement extraction"""
        description = "Build a system with authentication and reporting features"
        requirements = self.analyzer._extract_requirements(description, "web")
        
        assert len(requirements) > 0
        assert any("authentication" in req.lower() for req in requirements)
    
    def test_moscow_categorization(self):
        """Test MoSCoW categorization"""
        requirements = ["User authentication", "Advanced analytics", "Database integration"]
        moscow = self.analyzer._categorize_moscow(requirements, "web")
        
        assert "must" in moscow
        assert "should" in moscow
        assert "could" in moscow
        assert "wont" in moscow
        
        # Check that all requirements are categorized
        total_categorized = sum(len(moscow[category]) for category in moscow)
        assert total_categorized == len(requirements)


if __name__ == "__main__":
    pytest.main([__file__])
