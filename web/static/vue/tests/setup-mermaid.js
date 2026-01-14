// Mermaid headless test mock (stable DOM injection)
(function(){
  try {
    if (typeof window !== 'undefined') {
      // Provide a mock mermaid global with minimal API surface used by tests
      window.mermaid = {
        initialize: function() {},
        init: function() {},
        render: function(id, code, container) {
          // Inject a simple placeholder element to satisfy tests that rely on DOM
          const div = document.createElement('div');
          div.setAttribute('data-mermaid-id', id);
          div.textContent = 'mermaid placeholder';
          if (container && container.appendChild) {
            container.appendChild(div);
          }
        },
        run: function() {}
      };
    }
  } catch (e) {
    // ignore
  }
})();
