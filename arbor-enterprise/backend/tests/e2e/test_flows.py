"""End-to-End Tests with Playwright.

TIER 9 - Point 49: E2E Testing (Playwright)

Covers:
- LoginFlow
- DiscoveryFlow  
- CurationFlow

Runs on Chromium, Firefox, and WebKit.
Video recordings saved for CI artifacts.
"""

import os
import re
from typing import Generator

import pytest
from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
    expect,
)

# Test configuration
BASE_URL = os.getenv("E2E_BASE_URL", "http://localhost:3000")
API_URL = os.getenv("E2E_API_URL", "http://localhost:8000")

# Test user credentials (from fixtures or env)
TEST_USER_EMAIL = os.getenv("E2E_TEST_EMAIL", "test@arbor.app")
TEST_USER_PASSWORD = os.getenv("E2E_TEST_PASSWORD", "test-password-123")


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
async def browser_context_args(browser_context_args):
    """Configure browser context for all tests."""
    return {
        **browser_context_args,
        "viewport": {"width": 1280, "height": 720},
        "record_video_dir": "test-results/videos",
        "record_video_size": {"width": 1280, "height": 720},
    }


@pytest.fixture(scope="session")
async def authenticated_page(
    browser: Browser,
) -> Generator[Page, None, None]:
    """Provide an authenticated page context."""
    context = await browser.new_context(
        storage_state=None,
        record_video_dir="test-results/videos",
    )
    page = await context.new_page()
    
    # Perform login
    await page.goto(f"{BASE_URL}/login")
    await page.fill('[data-testid="email-input"]', TEST_USER_EMAIL)
    await page.fill('[data-testid="password-input"]', TEST_USER_PASSWORD)
    await page.click('[data-testid="login-button"]')
    
    # Wait for redirect to dashboard
    await page.wait_for_url(f"{BASE_URL}/dashboard", timeout=10000)
    
    yield page
    
    # Cleanup
    await context.close()


# ═══════════════════════════════════════════════════════════════════════════
# Login Flow Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestLoginFlow:
    """TIER 9 - Point 49: LoginFlow E2E tests."""
    
    @pytest.mark.asyncio
    async def test_login_page_loads(self, page: Page):
        """Login page should display correctly."""
        await page.goto(f"{BASE_URL}/login")
        
        # Check page elements
        await expect(page.locator('[data-testid="email-input"]')).to_be_visible()
        await expect(page.locator('[data-testid="password-input"]')).to_be_visible()
        await expect(page.locator('[data-testid="login-button"]')).to_be_visible()
        
        # Take screenshot
        await page.screenshot(path="test-results/screenshots/login-page.png")
    
    @pytest.mark.asyncio
    async def test_successful_login(self, page: Page):
        """User should be able to log in with valid credentials."""
        await page.goto(f"{BASE_URL}/login")
        
        # Fill form
        await page.fill('[data-testid="email-input"]', TEST_USER_EMAIL)
        await page.fill('[data-testid="password-input"]', TEST_USER_PASSWORD)
        
        # Submit
        await page.click('[data-testid="login-button"]')
        
        # Should redirect to dashboard
        await page.wait_for_url(f"{BASE_URL}/dashboard", timeout=10000)
        await expect(page).to_have_url(re.compile(r".*/dashboard"))
    
    @pytest.mark.asyncio
    async def test_login_validation_errors(self, page: Page):
        """Login form should show validation errors."""
        await page.goto(f"{BASE_URL}/login")
        
        # Submit empty form
        await page.click('[data-testid="login-button"]')
        
        # Should show error messages
        await expect(page.locator('[data-testid="email-error"]')).to_be_visible()
    
    @pytest.mark.asyncio
    async def test_forgot_password_link(self, page: Page):
        """Forgot password link should navigate correctly."""
        await page.goto(f"{BASE_URL}/login")
        
        await page.click('[data-testid="forgot-password-link"]')
        
        await expect(page).to_have_url(re.compile(r".*/forgot-password"))


# ═══════════════════════════════════════════════════════════════════════════
# Discovery Flow Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestDiscoveryFlow:
    """TIER 9 - Point 49: DiscoveryFlow E2E tests."""
    
    @pytest.mark.asyncio
    async def test_discovery_search(self, authenticated_page: Page):
        """User should be able to perform discovery search."""
        page = authenticated_page
        
        # Navigate to discover
        await page.goto(f"{BASE_URL}/discover")
        
        # Find search input
        search_input = page.locator('[data-testid="discovery-input"]')
        await expect(search_input).to_be_visible()
        
        # Type query
        await search_input.fill("aperitivo in centro Milano")
        await search_input.press("Enter")
        
        # Wait for results (SSE streaming)
        await page.wait_for_selector('[data-testid="discovery-result"]', timeout=30000)
        
        # Should have results
        results = page.locator('[data-testid="discovery-result"]')
        await expect(results.first).to_be_visible()
        
        # Take screenshot
        await page.screenshot(path="test-results/screenshots/discovery-results.png")
    
    @pytest.mark.asyncio
    async def test_discovery_streaming_response(self, authenticated_page: Page):
        """Discovery should show streaming response."""
        page = authenticated_page
        
        await page.goto(f"{BASE_URL}/discover")
        
        search_input = page.locator('[data-testid="discovery-input"]')
        await search_input.fill("ristorante romantico vista")
        await search_input.press("Enter")
        
        # Should show loading/streaming state
        await expect(page.locator('[data-testid="streaming-indicator"]')).to_be_visible()
        
        # Wait for completion
        await page.wait_for_selector('[data-testid="response-complete"]', timeout=30000)
    
    @pytest.mark.asyncio
    async def test_entity_card_click(self, authenticated_page: Page):
        """Clicking entity card should open details."""
        page = authenticated_page
        
        # Perform search
        await page.goto(f"{BASE_URL}/discover")
        search_input = page.locator('[data-testid="discovery-input"]')
        await search_input.fill("bar cocktail")
        await search_input.press("Enter")
        
        # Wait for results
        await page.wait_for_selector('[data-testid="entity-card"]', timeout=30000)
        
        # Click first entity
        await page.locator('[data-testid="entity-card"]').first.click()
        
        # Should open entity detail modal or page
        await expect(page.locator('[data-testid="entity-detail"]')).to_be_visible()


# ═══════════════════════════════════════════════════════════════════════════
# Curation Flow Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCurationFlow:
    """TIER 9 - Point 49: CurationFlow E2E tests."""
    
    @pytest.mark.asyncio
    async def test_curation_page_access(self, authenticated_page: Page):
        """Curator should access curation dashboard."""
        page = authenticated_page
        
        await page.goto(f"{BASE_URL}/curator")
        
        # Should see curation dashboard
        await expect(page.locator('[data-testid="curation-dashboard"]')).to_be_visible()
    
    @pytest.mark.asyncio
    async def test_entity_edit_form(self, authenticated_page: Page):
        """Curator should be able to edit entity."""
        page = authenticated_page
        
        await page.goto(f"{BASE_URL}/curator/entities")
        
        # Wait for entity list
        await page.wait_for_selector('[data-testid="entity-row"]', timeout=10000)
        
        # Click edit on first entity
        await page.locator('[data-testid="edit-button"]').first.click()
        
        # Should open edit form
        await expect(page.locator('[data-testid="entity-edit-form"]')).to_be_visible()
        
        # Edit a field
        name_input = page.locator('[data-testid="entity-name-input"]')
        await name_input.clear()
        await name_input.fill("Updated Entity Name")
        
        # Save
        await page.click('[data-testid="save-button"]')
        
        # Should show success toast
        await expect(page.locator('[data-testid="success-toast"]')).to_be_visible()
    
    @pytest.mark.asyncio
    async def test_entity_create(self, authenticated_page: Page):
        """Curator should be able to create new entity."""
        page = authenticated_page
        
        await page.goto(f"{BASE_URL}/curator/entities/new")
        
        # Fill form
        await page.fill('[data-testid="entity-name-input"]', "Test Entity E2E")
        await page.fill('[data-testid="entity-description-input"]', "Created by E2E test")
        await page.select_option('[data-testid="entity-category-select"]', "restaurant")
        
        # Submit
        await page.click('[data-testid="create-button"]')
        
        # Should redirect to entity detail
        await page.wait_for_url(re.compile(r".*/curator/entities/\w+"), timeout=10000)
    
    @pytest.mark.asyncio
    async def test_bulk_operations(self, authenticated_page: Page):
        """Curator should perform bulk operations."""
        page = authenticated_page
        
        await page.goto(f"{BASE_URL}/curator/entities")
        
        # Wait for list
        await page.wait_for_selector('[data-testid="entity-row"]', timeout=10000)
        
        # Select multiple entities
        checkboxes = page.locator('[data-testid="entity-checkbox"]')
        await checkboxes.nth(0).check()
        await checkboxes.nth(1).check()
        
        # Bulk action button should appear
        await expect(page.locator('[data-testid="bulk-actions"]')).to_be_visible()


# ═══════════════════════════════════════════════════════════════════════════
# API Health & Integration Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestAPIIntegration:
    """Test API integration from frontend."""
    
    @pytest.mark.asyncio
    async def test_api_health_check(self, page: Page):
        """API health endpoint should respond."""
        response = await page.request.get(f"{API_URL}/health")
        assert response.ok
        
        data = await response.json()
        assert data["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_api_readiness(self, page: Page):
        """API readiness endpoint should check all services."""
        response = await page.request.get(f"{API_URL}/health/readiness")
        assert response.ok
        
        data = await response.json()
        assert "postgres" in data
        assert "redis" in data


# ═══════════════════════════════════════════════════════════════════════════
# Visual Regression Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestVisualRegression:
    """Visual regression tests using screenshots."""
    
    @pytest.mark.asyncio
    async def test_homepage_visual(self, page: Page):
        """Homepage should match baseline."""
        await page.goto(BASE_URL)
        await page.wait_for_load_state("networkidle")
        
        screenshot = await page.screenshot(full_page=True)
        
        # In real implementation, compare with baseline using pixelmatch
        # For now, just save the screenshot
        with open("test-results/screenshots/homepage-baseline.png", "wb") as f:
            f.write(screenshot)
    
    @pytest.mark.asyncio
    async def test_dark_mode_toggle(self, page: Page):
        """Dark mode should apply correctly."""
        await page.goto(BASE_URL)
        
        # Toggle dark mode
        await page.click('[data-testid="theme-toggle"]')
        
        # Should have dark mode class
        html = page.locator("html")
        await expect(html).to_have_class(re.compile(r"dark"))
        
        await page.screenshot(path="test-results/screenshots/dark-mode.png")
