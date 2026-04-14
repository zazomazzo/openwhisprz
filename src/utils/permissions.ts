/** Returns true when all required OS permissions are granted. */
export function areRequiredPermissionsMet(micGranted: boolean): boolean {
  if (!micGranted) return false;

  // Accessibility is no longer required — falls back to clipboard-only mode.
  // Previously hard-blocked onboarding with stale TCC entries (#394).
  return true;
}

/**
 * localStorage flag recording that the user chose to proceed without macOS
 * Accessibility. Suppresses the nag toast and enables silent clipboard
 * fallback on paste. Cleared automatically when accessibility is later granted.
 */
export const ACCESSIBILITY_SKIPPED_KEY = "accessibilitySkipped";

export function isAccessibilitySkipped(): boolean {
  if (typeof window === "undefined") return false;
  return window.localStorage.getItem(ACCESSIBILITY_SKIPPED_KEY) === "true";
}
