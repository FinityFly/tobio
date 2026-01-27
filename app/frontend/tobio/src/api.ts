const LOCAL_API = 'http://localhost:8000';
// const REMOTE_API = 'https://unexcusedly-unobvious-lula.ngrok-free.dev';
const REMOTE_API = 'https://tobio.onrender.com/';

const USERNAME = import.meta.env.TOBIO_API_USERNAME || 'tobio';
const PASSWORD = import.meta.env.TOBIO_API_PASSWORD || 'tobio';

function getAuthHeader(): string {
  const token = btoa(`${USERNAME}:${PASSWORD}`);
  return `Basic ${token}`;
}

export async function apiFetch(
  endpoint: string,
  options: RequestInit = {}
): Promise<Response> {
  const headers = new Headers(options.headers || {});
  headers.set('Authorization', getAuthHeader());

  const fetchOptions = { ...options, headers };

  // Try localhost first
  try {
    const res = await fetch(`${LOCAL_API}${endpoint}`, fetchOptions);
    if (!res.ok) throw new Error(`Local API failed: ${res.status}`);
    return res;
  } catch (err) {
    // Fallback to remote
    const res = await fetch(`${REMOTE_API}${endpoint}`, fetchOptions);
    if (!res.ok) throw new Error(`Remote API failed: ${res.status}`);
    return res;
  }
}