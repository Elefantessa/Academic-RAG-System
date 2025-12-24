"""
API Routes

RESTful API endpoints for the RAG system.
"""

from flask import Blueprint, request, jsonify, current_app, render_template_string

from utils.logging_config import get_logger

logger = get_logger(__name__)

# HTML template for web interface
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Academic RAG System</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; padding: 20px;
        }
        .container { max-width: 900px; margin: 0 auto; }
        h1 { color: white; text-align: center; margin-bottom: 20px; }
        .chat-box {
            background: white; border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }
        .messages { height: 400px; overflow-y: auto; padding: 20px; }
        .message { margin: 10px 0; padding: 12px 16px; border-radius: 12px; }
        .user { background: #667eea; color: white; margin-left: 20%; }
        .assistant { background: #f0f0f0; margin-right: 20%; }
        .input-area {
            display: flex; padding: 15px;
            background: #f8f9fa; border-top: 1px solid #eee;
        }
        input {
            flex: 1; padding: 12px 16px; border: 2px solid #667eea;
            border-radius: 25px; font-size: 16px; outline: none;
        }
        button {
            margin-left: 10px; padding: 12px 24px;
            background: #667eea; color: white; border: none;
            border-radius: 25px; cursor: pointer; font-size: 16px;
        }
        button:hover { background: #5a6fd6; }
        .meta { font-size: 12px; color: #888; margin-top: 8px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎓 Academic RAG System</h1>
        <div class="chat-box">
            <div class="messages" id="messages"></div>
            <div class="input-area">
                <input type="text" id="query" placeholder="Ask about courses...">
                <button onclick="sendQuery()">Send</button>
            </div>
        </div>
    </div>
    <script>
        const input = document.getElementById('query');
        const messages = document.getElementById('messages');

        input.addEventListener('keypress', e => { if(e.key === 'Enter') sendQuery(); });

        async function sendQuery() {
            const query = input.value.trim();
            if (!query) return;

            addMessage(query, 'user');
            input.value = '';

            try {
                const res = await fetch('/api/query', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({query})
                });
                const data = await res.json();

                let meta = `Mode: ${data.generation_mode} | Confidence: ${(data.confidence*100).toFixed(0)}% | Time: ${data.processing_time?.toFixed(2)}s`;
                addMessage(data.answer, 'assistant', meta);
            } catch (err) {
                addMessage('Error: ' + err.message, 'assistant');
            }
        }

        function addMessage(text, type, meta='') {
            const div = document.createElement('div');
            div.className = 'message ' + type;
            div.innerHTML = text + (meta ? '<div class="meta">' + meta + '</div>' : '');
            messages.appendChild(div);
            messages.scrollTop = messages.scrollHeight;
        }
    </script>
</body>
</html>
'''


def register_routes(app):
    """Register all API routes."""

    @app.route('/')
    def index():
        """Serve web interface."""
        return render_template_string(HTML_TEMPLATE)

    @app.route('/api/query', methods=['POST'])
    def query():
        """Process a query."""
        try:
            data = request.get_json()
            if not data or 'query' not in data:
                return jsonify({"error": "Query required"}), 400

            query_text = data['query'].strip()
            if not query_text:
                return jsonify({"error": "Empty query"}), 400

            if len(query_text) > 1000:
                return jsonify({"error": "Query too long"}), 400

            agent = current_app.config.get('AGENT')
            if not agent:
                return jsonify({"error": "Agent not initialized"}), 500

            response = agent.process_query(query_text)

            return jsonify({
                "answer": response.answer,
                "confidence": response.confidence,
                "sources": response.sources,
                "generation_mode": response.generation_mode,
                "processing_time": response.processing_time,
                "metadata": response.metadata
            })

        except Exception as e:
            logger.error(f"Query error: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/health', methods=['GET'])
    def health():
        """Health check endpoint."""
        return jsonify({
            "status": "healthy",
            "service": "Academic RAG System",
            "version": "2.0-refactored"
        })

    @app.route('/api/stats', methods=['GET'])
    def stats():
        """Get system statistics."""
        agent = current_app.config.get('AGENT')
        if not agent:
            return jsonify({"error": "Agent not initialized"}), 500

        return jsonify(agent.get_stats())

    @app.route('/api/catalog', methods=['GET'])
    def catalog():
        """Get catalog information."""
        agent = current_app.config.get('AGENT')
        if not agent:
            return jsonify({"error": "Agent not initialized"}), 500

        cat = agent.catalog
        return jsonify({
            "stats": cat.get_catalog_stats(),
            "sample_codes": cat.get_all_codes()[:20],
            "sample_titles": cat.get_all_titles()[:10]
        })

    logger.info("API routes registered")
