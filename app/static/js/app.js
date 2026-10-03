/**
 * FitBuddy - Lightweight Client Interactions
 */

document.addEventListener("DOMContentLoaded", () => {
  // Query live health endpoint to show real-time connectivity
  const liveHealthBadge = document.getElementById("live-health-badge");
  const liveHealthText = document.getElementById("live-health-text");

  if (liveHealthBadge && liveHealthText) {
    fetch("/health")
      .then((res) => {
        if (!res.ok) throw new Error("Health check failed");
        return res.json();
      })
      .then((data) => {
        liveHealthText.textContent = `Online • ${data.environment}`;
        liveHealthBadge.classList.add("healthy");
      })
      .catch((err) => {
        console.error("Health check error:", err);
        liveHealthText.textContent = "Offline / Connection Error";
        liveHealthBadge.style.borderColor = "rgba(239, 68, 68, 0.4)";
        liveHealthBadge.style.color = "#f87171";
      });
  }
});
