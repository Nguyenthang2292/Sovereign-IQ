// Global Mermaid mock to avoid DOM errors in tests
(function(){
  try {
    if (typeof window !== 'undefined') {
      // Mock SVGTextElement.getBBox to prevent errors in mermaid rendering
      if (typeof SVGElement !== 'undefined') {
        SVGElement.prototype.getBBox = function() {
          return {
            x: 0,
            y: 0,
            width: 100,
            height: 20
          };
        };
      }

      // Mock SVGTextElement specifically if it exists
      if (typeof SVGTextElement !== 'undefined') {
        SVGTextElement.prototype.getBBox = function() {
          return {
            x: 0,
            y: 0,
            width: 100,
            height: 20
          };
        };
      }

      window.mermaid = {
        initialize: function() { return Promise.resolve(); },
        init: function() { return Promise.resolve(); },
        render: function(id, code, container) {
          const el = document.createElement('div');
          el.className = 'mermaid';
          el.id = id;
          el.innerHTML = code;
          if (container && container.appendChild) container.appendChild(el);
          return Promise.resolve({ svg: '<svg></svg>' });
        },
        run: function() { return Promise.resolve(); }
      };
    }
  } catch (e) {
    // ignore
  }
})();
