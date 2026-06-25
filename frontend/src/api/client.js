import axios from 'axios';

/**
 * API Client — Centralized HTTP client
 * 
 * Using axios for automatic JSON parsing, error handling, and timeouts.
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

const client = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: { 'Content-Type': 'application/json' },
});

export const analyzeText = async (text, model = 'lstm') => {
  const response = await client.post('/analyze', { text, model });
  return response.data;
};

export const analyzeBatch = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  const response = await client.post('/analyze/batch', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
};

export const getModelInfo = async () => {
  const response = await client.get('/models');
  return response.data;
};

export const healthCheck = async () => {
  const response = await client.get('/health');
  return response.data;
};

export default client;