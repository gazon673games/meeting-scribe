export const formatRequestError = (requestError) => {
  const name = requestError?.name || 'Error';
  const message = requestError?.message || requestError;
  return `${name}: ${message}`;
};
