// Apply saved theme or system preference on load
(function () {
  const saved = localStorage.getItem("theme");
  if (
    saved === "dark" ||
    (!saved && window.matchMedia("(prefers-color-scheme: dark)").matches)
  ) {
    document.documentElement.classList.add("dark");
  }
})();

function toggleTheme() {
  const isDark = document.documentElement.classList.toggle("dark");
  localStorage.setItem("theme", isDark ? "dark" : "light");
}

// Load Highlight.js (Gruvbox Light theme) if code blocks exist
(function () {
  var css = document.createElement("link");
  css.rel = "stylesheet";
  css.href =
    "https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/styles/base16/gruvbox-light-medium.min.css";
  document.head.appendChild(css);

  var js = document.createElement("script");
  js.src =
    "https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/highlight.min.js";
  js.onload = function () {
    hljs.highlightAll();
  };
  document.head.appendChild(js);
})();

// Inject theme toggle button + copy buttons (wait for DOM)
document.addEventListener("DOMContentLoaded", function () {
  // Theme toggle button
  var btn = document.createElement("button");
  btn.onclick = toggleTheme;
  btn.className =
    "fixed top-5 right-5 z-50 w-10 h-10 rounded-full bg-black/10 dark:bg-white/10 flex items-center justify-center hover:bg-black/20 dark:hover:bg-white/20 transition-colors";
  btn.innerHTML =
    '<svg class="w-5 h-5 hidden dark:block text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">' +
    '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z"/>' +
    "</svg>" +
    '<svg class="w-5 h-5 block dark:hidden text-ink" fill="none" stroke="currentColor" viewBox="0 0 24 24">' +
    '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"/>' +
    "</svg>";
  document.body.prepend(btn);

  // Top navigation bar: one row carrying prev, syllabus and next.
  // The prev/next links are cloned from the bottom nav so the two can never
  // drift apart, and the standalone "Back to syllabus" link is folded into
  // the middle slot rather than sitting on its own line above it.
  var article = document.querySelector("article");
  var bottomNav = article && article.querySelector("nav");
  if (bottomNav) {
    var backLink = article.querySelector('a[href="../index.html"]');
    var ends = Array.prototype.slice.call(bottomNav.children);
    var linkCls =
      "text-muted hover:text-c2 transition-colors whitespace-nowrap " +
      "overflow-hidden text-ellipsis";

    var bar = document.createElement("div");
    // a 3-column grid, not flex: on the first and last pages one end is an
    // empty placeholder, and with flex that collapses and drags the middle
    // link off centre
    bar.className =
      "mb-12 grid grid-cols-3 items-center gap-6 pb-4 text-sm " +
      "border-b border-gray-200 dark:border-gray-800";

    var prev = ends[0].cloneNode(true);
    prev.className =
      (ends[0].tagName === "A" ? linkCls : "") + " justify-self-start";
    var next = ends[ends.length - 1].cloneNode(true);
    next.className =
      (ends[ends.length - 1].tagName === "A" ? linkCls : "") +
      " justify-self-end";

    var mid = document.createElement("a");
    mid.href = backLink ? backLink.getAttribute("href") : "../index.html";
    mid.textContent = "Syllabus";
    mid.className = linkCls + " justify-self-center";

    bar.appendChild(prev);
    bar.appendChild(mid);
    bar.appendChild(next);
    article.insertBefore(bar, article.children[0]);
    if (backLink) backLink.hidden = true;
  }

  // Add copy buttons to all code blocks
  document.querySelectorAll("pre > code").forEach(function (codeEl) {
    var pre = codeEl.parentElement;
    var wrapper = pre.parentElement;
    if (!wrapper || !wrapper.classList.contains("rounded-xl")) return;

    var copyBtn = document.createElement("button");
    copyBtn.textContent = "Copy";
    copyBtn.className =
      "absolute top-2 right-2 px-3 py-1 text-xs font-mono rounded-md bg-black/10 hover:bg-black/20 text-[#7c6f64] transition-colors";
    copyBtn.onclick = function () {
      navigator.clipboard.writeText(codeEl.textContent).then(function () {
        copyBtn.textContent = "Copied!";
        setTimeout(function () {
          copyBtn.textContent = "Copy";
        }, 2000);
      });
    };

    pre.style.position = "relative";
    pre.appendChild(copyBtn);
  });
});
