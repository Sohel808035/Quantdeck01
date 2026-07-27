"""
tests/test_android_app.py
──────────────────────────
Unit Test Suite for QuantSphereX Android Application Architecture.

Covers:
  1. Manifest & Gradle File Structure
  2. Data Layer (Models, Repositories, Local Cache, Remote Data Source)
  3. Presentation Layer (Theme, MVVM ViewModels, Composables, Charts)
  4. Repository Pattern & Offline Caching Fallback Verification
  5. Architecture Integrity & Kotlin Package Standards
"""

import os
import unittest


class TestAndroidAppArchitecture(unittest.TestCase):
    def setUp(self):
        self.root = "android_app"

    def test_gradle_build_file_exists(self):
        gradle_path = os.path.join(self.root, "build.gradle.kts")
        self.assertTrue(os.path.exists(gradle_path))
        with open(gradle_path, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("compileSdk = 34", content)
        self.assertIn("androidx.compose.material3:material3", content)
        self.assertIn("androidx.room:room-runtime", content)

    def test_android_manifest_exists(self):
        manifest_path = os.path.join(self.root, "src", "main", "AndroidManifest.xml")
        self.assertTrue(os.path.exists(manifest_path))
        with open(manifest_path, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("package=\"com.quantspherex.app\"", content)
        self.assertIn("android.permission.INTERNET", content)

    def test_kotlin_data_models_exist(self):
        models_path = os.path.join(self.root, "src", "main", "java", "com", "quantspherex", "app", "data", "model", "Models.kt")
        self.assertTrue(os.path.exists(models_path))
        with open(models_path, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("data class PortfolioSummary", content)
        self.assertIn("data class ResearchAlphaItem", content)

    def test_local_cache_data_source_exists(self):
        cache_path = os.path.join(self.root, "src", "main", "java", "com", "quantspherex", "app", "data", "local", "LocalCacheDataSource.kt")
        self.assertTrue(os.path.exists(cache_path))
        with open(cache_path, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("class LocalCacheDataSource", content)
        self.assertIn("getFallbackPortfolio", content)

    def test_repository_pattern_implementation(self):
        repo_path = os.path.join(self.root, "src", "main", "java", "com", "quantspherex", "app", "data", "repository", "Repositories.kt")
        self.assertTrue(os.path.exists(repo_path))
        with open(repo_path, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("class PortfolioRepository", content)
        self.assertIn("class AuthRepository", content)
        self.assertIn("class ResearchRepository", content)

    def test_mvvm_viewmodels_exist(self):
        base = os.path.join(self.root, "src", "main", "java", "com", "quantspherex", "app", "presentation")
        self.assertTrue(os.path.exists(os.path.join(base, "auth", "AuthViewModel.kt")))
        self.assertTrue(os.path.exists(os.path.join(base, "portfolio", "PortfolioViewModel.kt")))
        self.assertTrue(os.path.exists(os.path.join(base, "research", "ResearchViewModel.kt")))

    def test_canvas_equity_chart_component_exists(self):
        chart_path = os.path.join(self.root, "src", "main", "java", "com", "quantspherex", "app", "presentation", "components", "EquityCurveChart.kt")
        self.assertTrue(os.path.exists(chart_path))
        with open(chart_path, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("fun EquityCurveChart", content)
        self.assertIn("Canvas", content)

    def test_main_activity_exists(self):
        activity_path = os.path.join(self.root, "src", "main", "java", "com", "quantspherex", "app", "presentation", "MainActivity.kt")
        self.assertTrue(os.path.exists(activity_path))
        with open(activity_path, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("class MainActivity", content)
        self.assertIn("PortfolioDashboardScreen", content)
        self.assertIn("ResearchDashboardScreen", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
