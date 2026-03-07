/**
 * Auth header utilities for API requests.
 */

export function withAuth(headers) {
  var token = (window.__LIAGENT_TOKEN__ || localStorage.getItem('liagent_token') || '').trim();
  if (token) headers['x-liagent-token'] = token;
  return headers;
}

export function authHeaders() {
  return withAuth({});
}
