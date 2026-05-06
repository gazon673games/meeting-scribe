export function formatRequestError(requestError) {
  return `${requestError?.name || "Error"}: ${requestError?.message || requestError}`;
}
