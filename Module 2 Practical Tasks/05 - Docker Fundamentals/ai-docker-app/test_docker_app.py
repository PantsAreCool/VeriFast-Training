"""
test_docker_app.py - Test script to verify the AI Docker application.
Run AFTER: docker compose up -d --build
"""

import time
import httpx

BASE_URL = "http://localhost:8000"


def test_health():
    """Test the health check endpoint."""
    response = httpx.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    print(f"[PASS] Health check: {data}")
    return data


def test_generate(prompt: str, use_cache: bool = False):
    """Test the generate endpoint."""
    payload = {
        "prompt": prompt,
        "model": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 256,
    }
    response = httpx.post(f"{BASE_URL}/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    cache_status = "CACHED" if data["cached"] else "FRESH"
    print(f"[PASS] Generate ({cache_status}): id={data['id']}, latency={data['latency_ms']}ms")
    return data


def test_cache_stats():
    """Test the cache stats endpoint."""
    response = httpx.get(f"{BASE_URL}/cache/stats")
    assert response.status_code == 200
    data = response.json()
    print(f"[PASS] Cache stats: {data}")
    return data


def test_flush_cache():
    """Test the flush cache endpoint."""
    response = httpx.delete(f"{BASE_URL}/cache/flush")
    assert response.status_code == 200
    print(f"[PASS] Cache flushed: {response.json()}")


def main():
    print("=" * 60)
    print("Testing AI Docker Application")
    print("=" * 60)

    # Wait for the service to be ready
    print("\nWaiting for API to be ready...")
    for i in range(10):
        try:
            test_health()
            break
        except httpx.ConnectError:
            time.sleep(2)
    else:
        print("[FAIL] Could not connect to API after 20 seconds")
        return

    # Test generation (first call - should be fresh)
    print("\n--- Test 1: Fresh generation ---")
    test_generate("What is machine learning?")

    # Test generation (same prompt - should be cached)
    print("\n--- Test 2: Cached generation ---")
    test_generate("What is machine learning?")

    # Test different prompt
    print("\n--- Test 3: Different prompt ---")
    test_generate("Explain neural networks.")

    # Check cache stats
    print("\n--- Test 4: Cache statistics ---")
    test_cache_stats()

    # Flush and verify
    print("\n--- Test 5: Flush cache ---")
    test_flush_cache()

    # Verify cache is empty
    print("\n--- Test 6: Cache after flush ---")
    test_cache_stats()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()