// Lightweight getBBox mock for SVG in jsdom
(function(){
  try {
    const fakeBBox = () => ({ x:0, y:0, width:0, height:0 });
    if (typeof window !== 'undefined') {
      const targets = [SVGTextElement?.prototype, SVGElement?.prototype, Element?.prototype];
      targets.forEach(proto => {
        if (proto && typeof proto.getBBox !== 'function') {
          proto.getBBox = fakeBBox;
        }
      });
    }
  } catch (e) {
    // ignore
  }
})();
// Mermaid override to provide safe DOM in tests
if (typeof window !== 'undefined' && window.mermaid) {
  window.mermaid.render = function(id, code, container) {
    const div = document.createElement('div');
    div.className = 'mermaid-container bg-gray-900/50 rounded-lg';
    if (container && container.appendChild) container.appendChild(div);
  };
  window.mermaid.initialize = function() {};
  window.mermaid.init = function() {};
  window.mermaid.run = function() {};
}

