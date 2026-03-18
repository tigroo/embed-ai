const CACHE_NAME = "yolo-detect-v5";

const PRECACHE = [
  "./",
  "./index.html",
  "./manifest.json",
  "./js/app.js",
  "./js/detector.js",
  "./js/coco-labels.js",
  "./icons/icon-192.png",
  // Models are NOT pre-cached — they are 3-10 MB each and caching them
  // in the SW install phase causes fetch failures (especially on Firefox
  // with self-signed certs).  They are fetched on demand by the app.
];

// Install: pre-cache app shell only
self.addEventListener("install", (e) => {
  e.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(PRECACHE))
  );
  self.skipWaiting();
});

// Activate: clean old caches
self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

// Fetch strategy
self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);

  // ONNX model files → always go to network directly (too large to cache
  // reliably, and caching interferes with fetch on some browsers).
  if (url.pathname.endsWith(".onnx")) {
    return;  // fall through to default browser fetch — no SW interception
  }

  // CDN resources (onnxruntime WASM/JS) — network first, cache fallback
  if (url.origin !== self.location.origin) {
    e.respondWith(
      fetch(e.request)
        .then((resp) => {
          const clone = resp.clone();
          caches.open(CACHE_NAME).then((c) => c.put(e.request, clone));
          return resp;
        })
        .catch(() => caches.match(e.request))
    );
    return;
  }

  // Local app-shell assets — cache first, then network (and update cache)
  e.respondWith(
    caches.match(e.request).then((cached) => {
      const fetched = fetch(e.request).then((resp) => {
        const clone = resp.clone();
        caches.open(CACHE_NAME).then((c) => c.put(e.request, clone));
        return resp;
      });
      return cached || fetched;
    })
  );
});
