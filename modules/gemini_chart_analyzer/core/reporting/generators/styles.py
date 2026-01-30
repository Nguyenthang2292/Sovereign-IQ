"""
CSS styles for HTML reports.

This module contains all CSS styling used in the generated HTML reports.
"""


def get_single_report_styles() -> str:
    """Get CSS styles for single timeframe reports."""
    return """
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%); color: #e0e0e0; line-height: 1.6; padding: 20px; min-height: 100vh; }
        .container { max-width: 1200px; margin: 0 auto; background: #2a2a2a; border-radius: 12px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5); overflow: hidden; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; text-align: center; color: white; }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); }
        .header .subtitle { font-size: 1.1em; opacity: 0.9; }
        .header .datetime { margin-top: 15px; font-size: 0.95em; opacity: 0.85; font-style: italic; }
        .content { padding: 30px; }
        .info-section { background: #333; padding: 20px; border-radius: 8px; margin-bottom: 30px; border-left: 4px solid #667eea; }
        .info-section h2 { color: #667eea; margin-bottom: 15px; font-size: 1.5em; }
        .info-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px; }
        .info-item { background: #3a3a3a; padding: 12px; border-radius: 6px; }
        .info-item label { color: #aaa; font-size: 0.9em; display: block; margin-bottom: 5px; }
        .info-item span { color: #fff; font-size: 1.1em; font-weight: bold; }
        .chart-section { background: #333; padding: 20px; border-radius: 8px; margin-bottom: 30px; border-left: 4px solid #48bb78; }
        .chart-section h2 { color: #48bb78; margin-bottom: 15px; font-size: 1.5em; }
        .chart-container { text-align: center; background: #1a1a1a; padding: 15px; border-radius: 8px; margin-top: 15px; }
        .chart-container img { max-width: 100%; height: auto; border-radius: 6px; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4); }
        .analysis-section { background: #333; padding: 20px; border-radius: 8px; border-left: 4px solid #f6ad55; }
        .analysis-section h2 { color: #f6ad55; margin-bottom: 15px; font-size: 1.5em; }
        .analysis-content { background: #2a2a2a; padding: 20px; border-radius: 6px; margin-top: 15px; line-height: 1.8; }
        .analysis-content p { margin-bottom: 15px; }
        .analysis-content strong { color: #f6ad55; }
        .analysis-content em { color: #a0a0a0; font-style: italic; }
    """


def get_multi_tf_report_styles() -> str:
    """Get CSS styles for multi-timeframe reports."""
    return """
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #1a1a1a; color: #e0e0e0; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; background: #2a2a2a; border-radius: 12px; overflow: hidden; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; text-align: center; color: white; }
        .agg-signal { font-size: 2em; margin-top: 20px; padding: 10px; border-radius: 8px; background: rgba(0,0,0,0.2); }
        .accordion-item { border-bottom: 1px solid #444; }
        .accordion-checkbox { display: none; }
        .accordion-header { display: flex; justify-content: space-between; padding: 20px; background: #333; cursor: pointer; }
        .accordion-content { max-height: 0; overflow: hidden; transition: max-height 0.3s; background: #222; }
        .accordion-checkbox:checked ~ .accordion-content { max-height: 2000px; padding: 20px; }
        .signal-badge { padding: 5px 12px; border-radius: 6px; font-weight: bold; color: white; }
        .chart-container img { max-width: 100%; border-radius: 6px; }
        .analysis-content { line-height: 1.8; margin-top: 10px; }
    """


def get_batch_report_styles() -> str:
    """Get CSS styles for batch scan reports."""
    return """
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #1e1e1e; color: #e0e0e0; line-height: 1.6; padding: 20px; }
        .container { max-width: 1600px; margin: 0 auto; background: #2a2a2a; border-radius: 12px; overflow: hidden; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; text-align: center; color: white; }
        .content { padding: 30px; }
        .summary-section { background: #333; padding: 25px; border-radius: 8px; margin-bottom: 30px; border-left: 4px solid #667eea; }
        .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-top: 20px; }
        .summary-card { background: #3a3a3a; padding: 20px; border-radius: 8px; text-align: center; }
        .summary-card.long .value { color: #48bb78; }
        .summary-card.short .value { color: #f56565; }
        .summary-card .value { font-size: 2em; font-weight: bold; }
        .accordion-item { background: #333; border-radius: 8px; margin-bottom: 15px; border: 1px solid #444; overflow: hidden; }
        .accordion-checkbox { display: none; }
        .accordion-header { display: flex; align-items: center; justify-content: space-between; padding: 20px 25px; background: #3a3a3a; cursor: pointer; }
        .accordion-content { max-height: 0; overflow: hidden; transition: max-height 0.3s; }
        .accordion-checkbox:checked ~ .accordion-content { max-height: 10000px; }
        .symbols-table { width: 100%; border-collapse: collapse; background: #333; }
        .symbols-table th { background: #3a3a3a; padding: 15px; text-align: left; color: #667eea; cursor: pointer; position: sticky; top: 0; }
        .symbols-table td { padding: 12px 15px; border-bottom: 1px solid #444; }
        .confidence-bar-container { display: flex; align-items: center; gap: 10px; }
        .confidence-bar { flex: 1; height: 15px; background: #1a1a1a; border-radius: 10px; overflow: hidden; }
        .confidence-fill { height: 100%; border-radius: 10px; }
        .detail-btn { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; }
        .tf-badge { display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 0.8em; margin: 2px; }
        .modal-overlay { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.7); z-index: 1000; justify-content: center; align-items: center; }
        .modal-overlay.show { display: flex; }
        .modal-content { background: #2a2a2a; border-radius: 12px; padding: 30px; width: 90%; max-width: 700px; }
        .command-container { background: #1a1a1a; padding: 15px; border-radius: 6px; margin: 15px 0; }
        .command-text { color: #48bb78; font-family: monospace; word-break: break-all; }
    """
