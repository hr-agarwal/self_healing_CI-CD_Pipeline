// Small formatting helpers used by the API routes in src/app.js.

function formatGreeting(name) {
  const clean = String(name).trim();
  return `Hello, ${clean}!`;
}

function formatSum(a, b) {
  return { a, b, sum: a + b };
}

module.exports = { formatGreeting, formatSum };
