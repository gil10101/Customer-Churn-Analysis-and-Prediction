/**
 * Load Testing Script for Churn Prediction API
 * 
 * This script tests the API under various load conditions to ensure
 * performance requirements are met before deployment.
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const predictionTrend = new Trend('prediction_duration');
const batchPredictionTrend = new Trend('batch_prediction_duration');

// Test configuration
export const options = {
  stages: [
    // Ramp up
    { duration: '2m', target: 10 },   // Ramp up to 10 users over 2 minutes
    { duration: '5m', target: 10 },   // Stay at 10 users for 5 minutes
    { duration: '2m', target: 50 },   // Ramp up to 50 users over 2 minutes
    { duration: '5m', target: 50 },   // Stay at 50 users for 5 minutes
    { duration: '2m', target: 100 },  // Ramp up to 100 users over 2 minutes
    { duration: '5m', target: 100 },  // Stay at 100 users for 5 minutes
    // Ramp down
    { duration: '2m', target: 0 },    // Ramp down to 0 users over 2 minutes
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'], // 95% of requests should be below 500ms
    http_req_failed: ['rate<0.1'],    // Error rate should be less than 10%
    errors: ['rate<0.1'],             // Custom error rate should be less than 10%
  },
};

// Base URL
const BASE_URL = 'http://localhost:8000';

// Sample customer data for testing
const sampleCustomer = {
  customer_id: `test_customer_${Math.random().toString(36).substr(2, 9)}`,
  gender: 'Female',
  senior_citizen: false,
  partner: true,
  dependents: false,
  tenure: Math.floor(Math.random() * 72) + 1,
  contract: ['Month-to-month', 'One year', 'Two year'][Math.floor(Math.random() * 3)],
  paperless_billing: Math.random() > 0.5,
  payment_method: ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'][Math.floor(Math.random() * 4)],
  phone_service: true,
  multiple_lines: ['No', 'Yes'][Math.floor(Math.random() * 2)],
  internet_service: ['DSL', 'Fiber optic', 'No'][Math.floor(Math.random() * 3)],
  online_security: ['No', 'Yes'][Math.floor(Math.random() * 2)],
  online_backup: ['No', 'Yes'][Math.floor(Math.random() * 2)],
  device_protection: ['No', 'Yes'][Math.floor(Math.random() * 2)],
  tech_support: ['No', 'Yes'][Math.floor(Math.random() * 2)],
  streaming_tv: ['No', 'Yes'][Math.floor(Math.random() * 2)],
  streaming_movies: ['No', 'Yes'][Math.floor(Math.random() * 2)],
  monthly_charges: Math.random() * 100 + 20,
  total_charges: Math.random() * 8000 + 100,
  usage_minutes_monthly: Math.random() * 1000,
  data_usage_gb_monthly: Math.random() * 50,
  support_interactions_count: Math.floor(Math.random() * 10),
  complaint_count: Math.floor(Math.random() * 5),
  satisfaction_score: Math.random() * 10,
  payment_delay_frequency: Math.random() * 0.5,
  service_change_frequency: Math.random() * 2,
};

// Generate batch of customers
function generateCustomerBatch(size) {
  const customers = [];
  for (let i = 0; i < size; i++) {
    const customer = { ...sampleCustomer };
    customer.customer_id = `batch_customer_${i}_${Math.random().toString(36).substr(2, 9)}`;
    customer.tenure = Math.floor(Math.random() * 72) + 1;
    customer.monthly_charges = Math.random() * 100 + 20;
    customer.total_charges = Math.random() * 8000 + 100;
    customers.push(customer);
  }
  return customers;
}

export default function () {
  // Test scenario weights
  const scenario = Math.random();
  
  if (scenario < 0.6) {
    // 60% - Single predictions
    testSinglePrediction();
  } else if (scenario < 0.8) {
    // 20% - Health checks
    testHealthCheck();
  } else if (scenario < 0.9) {
    // 10% - Model info
    testModelInfo();
  } else {
    // 10% - Batch predictions
    testBatchPrediction();
  }
  
  // Random sleep between 1-3 seconds
  sleep(Math.random() * 2 + 1);
}

function testSinglePrediction() {
  const payload = JSON.stringify(sampleCustomer);
  const params = {
    headers: {
      'Content-Type': 'application/json',
    },
  };
  
  const response = http.post(`${BASE_URL}/predict`, payload, params);
  
  const success = check(response, {
    'prediction status is 200': (r) => r.status === 200,
    'prediction response has churn_probability': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.hasOwnProperty('churn_probability');
      } catch (e) {
        return false;
      }
    },
    'prediction response time < 1000ms': (r) => r.timings.duration < 1000,
  });
  
  predictionTrend.add(response.timings.duration);
  errorRate.add(!success);
}

function testBatchPrediction() {
  const batchSize = Math.floor(Math.random() * 10) + 5; // 5-15 customers
  const customers = generateCustomerBatch(batchSize);
  
  const payload = JSON.stringify({ customers });
  const params = {
    headers: {
      'Content-Type': 'application/json',
    },
  };
  
  const response = http.post(`${BASE_URL}/predict/batch`, payload, params);
  
  const success = check(response, {
    'batch prediction status is 200': (r) => r.status === 200,
    'batch prediction has results': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.predictions && body.predictions.length === batchSize;
      } catch (e) {
        return false;
      }
    },
    'batch prediction response time < 5000ms': (r) => r.timings.duration < 5000,
  });
  
  batchPredictionTrend.add(response.timings.duration);
  errorRate.add(!success);
}

function testHealthCheck() {
  const response = http.get(`${BASE_URL}/model/health`);
  
  const success = check(response, {
    'health check status is 200': (r) => r.status === 200,
    'health check has status': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.hasOwnProperty('status');
      } catch (e) {
        return false;
      }
    },
    'health check response time < 200ms': (r) => r.timings.duration < 200,
  });
  
  errorRate.add(!success);
}

function testModelInfo() {
  const response = http.get(`${BASE_URL}/model/info`);
  
  const success = check(response, {
    'model info status is 200': (r) => r.status === 200,
    'model info has model_name': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.hasOwnProperty('model_name');
      } catch (e) {
        return false;
      }
    },
    'model info response time < 500ms': (r) => r.timings.duration < 500,
  });
  
  errorRate.add(!success);
}

// Setup function - runs once before the test
export function setup() {
  console.log('Starting load test setup...');
  
  // Wait for API to be ready
  let retries = 0;
  const maxRetries = 30;
  
  while (retries < maxRetries) {
    const response = http.get(`${BASE_URL}/model/health`);
    if (response.status === 200) {
      console.log('API is ready for load testing');
      return;
    }
    
    console.log(`Waiting for API to be ready... (attempt ${retries + 1}/${maxRetries})`);
    sleep(2);
    retries++;
  }
  
  throw new Error('API not ready after maximum retries');
}

// Teardown function - runs once after the test
export function teardown(data) {
  console.log('Load test completed');
  console.log(`Total prediction requests: ${predictionTrend.values.length}`);
  console.log(`Total batch prediction requests: ${batchPredictionTrend.values.length}`);
  console.log(`Average prediction response time: ${predictionTrend.values.reduce((a, b) => a + b, 0) / predictionTrend.values.length}ms`);
  
  if (batchPredictionTrend.values.length > 0) {
    console.log(`Average batch prediction response time: ${batchPredictionTrend.values.reduce((a, b) => a + b, 0) / batchPredictionTrend.values.length}ms`);
  }
}