const CACHE_NAME = "yolo-detect-v12";

const PRECACHE = [
  "./",
  "./index.html",
  "./manifest.json",
  "./js/app.js",
  "./js/detector.js",
  "./js/coco-labels.js",
  "./icons/icon-192.png",
];

// Install: pre-cache app shell.
// Do NOT call skipWaiting() — on iOS it can trigger a page reload that
// kills the camera stream and forces a new permission prompt.
self.addEventListener("install", (e) => {
  e.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(PRECACHE))
  );
});

// Activate: clean old caches.
// Do NOT call clients.claim() — same iOS reload issue.
self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
    )
  );
});

// Fetch: network-first for everything (avoids stale-cache surprises).
// ONNX models bypass the SW entirely.
self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);

  // .onnx model files → let browser fetch directly, never cache
  if (url.pathname.endsWith(".onnx")) return;

  e.respondWith(
    fetch(e.request)
      .then((resp) => {
        const clone = resp.clone();
        caches.open(CACHE_NAME).then((c) => c.put(e.request, clone));
        return resp;
      })
      .catch(() => caches.match(e.request))
  );
});
