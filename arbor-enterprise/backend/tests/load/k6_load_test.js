"""K6 load testing script for ARBOR Enterprise API.

TIER 9 - Point 47: Load Testing(K6)

Script for verifying API performance under load.

    Usage:
    k6 run tests / load / k6_load_test.js

Performance targets:
- p99 latency < 5s
    - Error rate < 1 %
        - Throughput: 50 RPS sustained
"""

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const discoveryLatency = new Trend('discovery_latency');
const searchLatency = new Trend('search_latency');

// Test configuration
export const options = {
    stages: [
        { duration: '1m', target: 10 },   // Ramp up to 10 VUs
        { duration: '2m', target: 50 },   // Ramp up to 50 VUs
        { duration: '5m', target: 50 },   // Stay at 50 VUs
        { duration: '1m', target: 0 },    // Ramp down
    ],
    thresholds: {
        http_req_duration: ['p(99)<5000'],  // 99% of requests under 5s
        errors: ['rate<0.01'],               // Error rate under 1%
        discovery_latency: ['p(95)<3000'],   // 95% discover requests under 3s
    },
};

const BASE_URL = __ENV.API_URL || 'http://localhost:8000';

// Sample queries for discovery
const DISCOVERY_QUERIES = [
    'Romantic Italian restaurant in Milan',
    'Trendy cocktail bar for young professionals',
    'Cozy coffee shop with good wifi',
    'High-end fashion boutique',
    'Art gallery with contemporary exhibitions',
    'Authentic Japanese restaurant',
    'Rooftop bar with city views',
    'Vintage clothing store',
];

// Sample categories for search
const CATEGORIES = ['restaurant', 'bar', 'shop', 'gallery', 'cafe'];

export function setup() {
    // Verify API is healthy
    const healthRes = http.get(`${BASE_URL}/health`);
    check(healthRes, {
        'health check passed': (r) => r.status === 200,
    });

    console.log(`Load testing ${BASE_URL}`);
    return { startTime: Date.now() };
}

export default function () {
    // Randomly choose between endpoint types
    const choice = Math.random();

    if (choice < 0.4) {
        testDiscovery();
    } else if (choice < 0.7) {
        testSearch();
    } else if (choice < 0.9) {
        testEntityList();
    } else {
        testHealthCheck();
    }

    sleep(Math.random() * 2 + 0.5);  // 0.5-2.5s between requests
}

function testDiscovery() {
    const query = DISCOVERY_QUERIES[Math.floor(Math.random() * DISCOVERY_QUERIES.length)];

    const payload = JSON.stringify({
        query: query,
        limit: 5,
    });

    const params = {
        headers: {
            'Content-Type': 'application/json',
        },
        timeout: '25s',
    };

    const start = Date.now();
    const res = http.post(`${BASE_URL}/api/v1/discover`, payload, params);
    const duration = Date.now() - start;

    discoveryLatency.add(duration);

    const success = check(res, {
        'discover status 200': (r) => r.status === 200,
        'discover has recommendations': (r) => {
            try {
                const body = JSON.parse(r.body);
                return body.recommendations && body.recommendations.length > 0;
            } catch {
                return false;
            }
        },
    });

    errorRate.add(!success);
}

function testSearch() {
    const category = CATEGORIES[Math.floor(Math.random() * CATEGORIES.length)];

    const params = {
        timeout: '10s',
    };

    const start = Date.now();
    const res = http.get(`${BASE_URL}/api/v1/search?category=${category}&limit=10`, params);
    const duration = Date.now() - start;

    searchLatency.add(duration);

    const success = check(res, {
        'search status 200': (r) => r.status === 200,
        'search has results': (r) => {
            try {
                const body = JSON.parse(r.body);
                return body.results !== undefined;
            } catch {
                return false;
            }
        },
    });

    errorRate.add(!success);
}

function testEntityList() {
    const res = http.get(`${BASE_URL}/api/v1/entities?limit=20`, {
        timeout: '5s',
    });

    const success = check(res, {
        'entities status 200': (r) => r.status === 200,
        'entities has items': (r) => {
            try {
                const body = JSON.parse(r.body);
                return body.items && Array.isArray(body.items);
            } catch {
                return false;
            }
        },
    });

    errorRate.add(!success);
}

function testHealthCheck() {
    const res = http.get(`${BASE_URL}/health/readiness`, {
        timeout: '5s',
    });

    check(res, {
        'readiness check status 200 or 503': (r) => r.status === 200 || r.status === 503,
    });
}

export function teardown(data) {
    const duration = (Date.now() - data.startTime) / 1000;
    console.log(`Load test completed in ${duration.toFixed(1)}s`);
}
