(function exposePerceptionState(root) {
  function frameNarration(hidden, reappeared, override) {
    if (override) return override;
    if (hidden) return "Model 1 is expanding possible hidden paths.";
    if (reappeared) return "Evidence arrived. The guess is scored, then Model 2 learns.";
    return "The ball is approaching the occluder.";
  }

  const api = {frameNarration};
  root.PerceptionUIState = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis === "undefined" ? this : globalThis);
