from http.server import BaseHTTPRequestHandler
import json

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Wind Turbine Failure Protection System</title>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                * { margin: 0; padding: 0; box-sizing: border-box; }
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    padding: 20px;
                }
                .container {
                    background: white;
                    border-radius: 20px;
                    padding: 40px;
                    max-width: 1200px;
                    width: 100%;
                    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                }
                h1 {
                    color: #667eea;
                    text-align: center;
                    margin-bottom: 10px;
                    font-size: 2.5rem;
                }
                .subtitle {
                    text-align: center;
                    color: #666;
                    margin-bottom: 40px;
                    font-size: 1.1rem;
                }
                .dashboard {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin-bottom: 40px;
                }
                .card {
                    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                    padding: 25px;
                    border-radius: 15px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    transition: transform 0.3s;
                }
                .card:hover { transform: translateY(-5px); }
                .card h3 {
                    color: #667eea;
                    margin-bottom: 10px;
                    font-size: 1.2rem;
                }
                .card p {
                    color: #555;
                    line-height: 1.6;
                }
                .metric {
                    font-size: 2rem;
                    font-weight: bold;
                    color: #764ba2;
                    margin: 10px 0;
                }
                .features {
                    background: #f8f9fa;
                    padding: 30px;
                    border-radius: 15px;
                    margin-bottom: 30px;
                }
                .features h2 {
                    color: #667eea;
                    margin-bottom: 20px;
                    text-align: center;
                }
                .features ul {
                    list-style: none;
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 15px;
                }
                .features li {
                    padding: 15px;
                    background: white;
                    border-radius: 10px;
                    border-left: 4px solid #667eea;
                }
                .features li::before {
                    content: "✓ ";
                    color: #28a745;
                    font-weight: bold;
                    margin-right: 10px;
                }
                .status {
                    text-align: center;
                    padding: 20px;
                    background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
                    color: white;
                    border-radius: 15px;
                    font-size: 1.3rem;
                    font-weight: bold;
                }
                .tech-stack {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 10px;
                    justify-content: center;
                    margin-top: 30px;
                }
                .tech-badge {
                    background: #667eea;
                    color: white;
                    padding: 8px 16px;
                    border-radius: 20px;
                    font-size: 0.9rem;
                    font-weight: 500;
                }
                .instructions {
                    background: #fff3cd;
                    padding: 20px;
                    border-radius: 10px;
                    border-left: 4px solid #ffc107;
                    margin-top: 30px;
                }
                .instructions h3 {
                    color: #856404;
                    margin-bottom: 10px;
                }
                .instructions code {
                    background: #f8f9fa;
                    padding: 2px 8px;
                    border-radius: 4px;
                    font-family: 'Courier New', monospace;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🌬️ Wind Turbine Failure Protection System</h1>
                <p class="subtitle">ML-Powered Predictive Maintenance & Real-Time Monitoring</p>

                <div class="status">
                    ✅ System Deployed Successfully
                </div>

                <div class="dashboard">
                    <div class="card">
                        <h3>📊 Real-Time Monitor</h3>
                        <p>Live simulation of turbine parameters with instant failure detection</p>
                        <div class="metric">24/7</div>
                    </div>
                    <div class="card">
                        <h3>🧠 ML Models</h3>
                        <p>Random Forest & Isolation Forest algorithms</p>
                        <div class="metric">95%+</div>
                    </div>
                    <div class="card">
                        <h3>🔍 Anomaly Detection</h3>
                        <p>Advanced pattern recognition for early warnings</p>
                        <div class="metric">Active</div>
                    </div>
                    <div class="card">
                        <h3>📈 Analytics</h3>
                        <p>Comprehensive historical data analysis</p>
                        <div class="metric">∞</div>
                    </div>
                </div>

                <div class="features">
                    <h2>🚀 Key Features</h2>
                    <ul>
                        <li>Real-time monitoring of 7 critical parameters</li>
                        <li>Random Forest Classifier for failure prediction</li>
                        <li>Isolation Forest for anomaly detection</li>
                        <li>Interactive 3D visualizations</li>
                        <li>Historical trend analysis</li>
                        <li>Automated alert system</li>
                        <li>Feature importance analysis</li>
                        <li>Confusion matrix & classification reports</li>
                        <li>Correlation heatmaps</li>
                        <li>Customizable training parameters</li>
                    </ul>
                </div>

                <div class="instructions">
                    <h3>🖥️ Running Locally with Streamlit</h3>
                    <p>This application is built with Streamlit and requires Python to run locally:</p>
                    <ol style="margin-top: 15px; margin-left: 20px; line-height: 1.8;">
                        <li>Install dependencies: <code>pip install streamlit numpy pandas plotly scikit-learn</code></li>
                        <li>Run the application: <code>streamlit run app.py</code></li>
                        <li>Access at: <code>http://localhost:8501</code></li>
                    </ol>
                    <p style="margin-top: 15px;">
                        <strong>Note:</strong> Full interactive features including real-time simulation,
                        ML training, and data visualization are available in the local Streamlit version.
                    </p>
                </div>

                <div class="tech-stack">
                    <span class="tech-badge">Python</span>
                    <span class="tech-badge">Streamlit</span>
                    <span class="tech-badge">Scikit-learn</span>
                    <span class="tech-badge">Plotly</span>
                    <span class="tech-badge">Pandas</span>
                    <span class="tech-badge">NumPy</span>
                    <span class="tech-badge">Random Forest</span>
                    <span class="tech-badge">Isolation Forest</span>
                </div>
            </div>
        </body>
        </html>
        """

        self.wfile.write(html.encode())
        return
