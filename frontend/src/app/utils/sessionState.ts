const isPlainObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" &&
  value !== null &&
  !Array.isArray(value);

export function readSessionState<T>(
  key: string,
  fallback: T,
): T {
  if (typeof window === "undefined") {
    return fallback;
  }

  try {
    const rawValue = window.sessionStorage.getItem(key);

    if (!rawValue) {
      return fallback;
    }

    const parsedValue = JSON.parse(rawValue) as T;

    if (
      isPlainObject(fallback) &&
      isPlainObject(parsedValue)
    ) {
      return {
        ...fallback,
        ...parsedValue,
      } as T;
    }

    return parsedValue;
  } catch {
    return fallback;
  }
}

export function writeSessionState<T>(
  key: string,
  value: T,
) {
  if (typeof window === "undefined") {
    return;
  }

  try {
    window.sessionStorage.setItem(
      key,
      JSON.stringify(value),
    );
  } catch {
    // Ignore storage write failures.
  }
}

export function clearSessionState(key: string) {
  if (typeof window === "undefined") {
    return;
  }

  try {
    window.sessionStorage.removeItem(key);
  } catch {
    // Ignore storage removal failures.
  }
}
