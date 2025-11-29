#!/usr/bin/env python3
"""
TaskFlow AI Reward History Dashboard
Provides visualization for reward system performance and training history.
"""

from flask import Flask, render_template_string, jsonify
import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from rewards import RewardSystem
from mini_lm import MiniLM

# Initialize Flask app for dashboard
dashboard_app = Flask(__name__)

# Global instances
_reward_system = None
_mini_lm = None

def get_reward_system():
    """Get or create reward system instance."""
    global _reward_system
    if _reward_system is None:
        _mini_lm = MiniLM()
        _reward_system = RewardSystem(_mini_lm)
    return _reward_system

# HTML template for the dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TaskFlow AI - Reward Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        .header p {
            margin: 10px 0 0 0;
            opacity: 0.9;
            font-size: 1.1em;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            padding: 30px;
        }
        .stat-card {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            border-left: 4px solid #4facfe;
        }
        .stat-card h3 {
            margin: 0 0 10px 0;
            color: #333;
            font-size: 2em;
            font-weight: bold;
        }
        .stat-card p {
            margin: 0;
            color: #666;
            font-size: 0.9em;
        }
        .charts-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 30px;
            padding: 30px;
        }
        .chart-card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .chart-card h3 {
            margin-top: 0;
            color: #333;
            text-align: center;
        }
        .reward-history {
            padding: 30px;
            background: #f8f9fa;
        }
        .reward-history h2 {
            margin-top: 0;
            color: #333;
            text-align: center;
        }
        .reward-item {
            background: white;
            margin: 10px 0;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #28a745;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .reward-positive {
            border-left-color: #28a745;
        }
        .reward-negative {
            border-left-color: #dc3545;
        }
        .reward-score {
            font-weight: bold;
            font-size: 1.2em;
        }
        .reward-details {
            flex: 1;
            margin-left: 15px;
        }
        .footer {
            text-align: center;
            padding: 20px;
            background: #343a40;
            color: white;
        }
        .refresh-btn {
            background: #4facfe;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1em;
            margin: 20px 0;
        }
        .refresh-btn:hover {
            background: #3a9dfe;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 TaskFlow AI Dashboard</h1>
            <p>Real-time Reward System Performance & Training History</p>
        </div>

        <div class="stats-grid" id="stats-grid">
            <!-- Stats will be loaded here -->
        </div>

        <div class="charts-container">
            <div class="chart-card">
                <h3>Reward Distribution</h3>
                <canvas id="rewardChart"></canvas>
            </div>
            <div class="chart-card">
                <h3>Performance Over Time</h3>
                <canvas id="performanceChart"></canvas>
            </div>
        </div>

        <div class="reward-history">
            <h2>Recent Reward History</h2>
            <button class="refresh-btn" onclick="loadData()">🔄 Refresh Dashboard</button>
            <div id="reward-list">
                <!-- Reward history will be loaded here -->
            </div>
        </div>

        <div class="footer">
            <p>TaskFlow AI v2.0 - Advanced Reinforcement Learning System</p>
        </div>
    </div>

    <script>
        let rewardChart, performanceChart;

        async function loadData() {
            try {
                const response = await fetch('/api/rewards/stats');
                const data = await response.json();

                updateStats(data);
                updateCharts(data);
                updateRewardHistory(data);
            } catch (error) {
                console.error('Failed to load data:', error);
            }
        }

        function updateStats(data) {
            const statsGrid = document.getElementById('stats-grid');
            const feedback = data.feedback_summary;
            const rl = data.rl_stats;

            statsGrid.innerHTML = `
                <div class="stat-card">
                    <h3>${feedback.total_evaluations}</h3>
                    <p>Total Evaluations</p>
                </div>
                <div class="stat-card">
                    <h3>${feedback.positive_rewards}</h3>
                    <p>Positive Rewards</p>
                </div>
                <div class="stat-card">
                    <h3>${(feedback.accuracy * 100).toFixed(1)}%</h3>
                    <p>Success Rate</p>
                </div>
                <div class="stat-card">
                    <h3>${rl.total_steps}</h3>
                    <p>RL Training Steps</p>
                </div>
                <div class="stat-card">
                    <h3>${rl.average_reward.toFixed(2)}</h3>
                    <p>Average Reward</p>
                </div>
                <div class="stat-card">
                    <h3>${rl.learning_rate.toFixed(4)}</h3>
                    <p>Learning Rate</p>
                </div>
            `;
        }

        function updateCharts(data) {
            const feedback = data.feedback_summary;
            const rl = data.rl_stats;

            // Reward Distribution Chart
            const rewardCtx = document.getElementById('rewardChart').getContext('2d');
            if (rewardChart) rewardChart.destroy();

            rewardChart = new Chart(rewardCtx, {
                type: 'doughnut',
                data: {
                    labels: ['Positive Rewards', 'Negative Rewards'],
                    datasets: [{
                        data: [feedback.positive_rewards, feedback.negative_rewards],
                        backgroundColor: ['#28a745', '#dc3545'],
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            position: 'bottom'
                        }
                    }
                }
            });

            // Performance Over Time Chart
            const perfCtx = document.getElementById('performanceChart').getContext('2d');
            if (performanceChart) performanceChart.destroy();

            const recentPerf = rl.recent_performance || [];
            performanceChart = new Chart(perfCtx, {
                type: 'line',
                data: {
                    labels: recentPerf.map((_, i) => `Step ${i + 1}`),
                    datasets: [{
                        label: 'Reward Score',
                        data: recentPerf,
                        borderColor: '#4facfe',
                        backgroundColor: 'rgba(79, 172, 254, 0.1)',
                        borderWidth: 2,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 2,
                            min: -2
                        }
                    }
                }
            });
        }

        function updateRewardHistory(data) {
            const rewardList = document.getElementById('reward-list');
            const recentPerf = data.rl_stats.recent_performance || [];

            if (recentPerf.length === 0) {
                rewardList.innerHTML = '<p>No recent reward history available.</p>';
                return;
            }

            rewardList.innerHTML = recentPerf.slice(-10).reverse().map((reward, index) => `
                <div class="reward-item ${reward >= 0 ? 'reward-positive' : 'reward-negative'}">
                    <div class="reward-score">${reward >= 0 ? '+' : ''}${reward.toFixed(2)}</div>
                    <div class="reward-details">
                        <strong>Training Step ${recentPerf.length - index}</strong><br>
                        <small>${reward >= 0 ? 'Positive reinforcement' : 'Negative reinforcement'}</small>
                    </div>
                </div>
            `).join('');
        }

        // Load data on page load
        document.addEventListener('DOMContentLoaded', loadData);

        // Auto-refresh every 30 seconds
        setInterval(loadData, 30000);
    </script>
</body>
</html>
"""

@dashboard_app.route('/')
def dashboard():
    """Serve the main dashboard page."""
    return render_template_string(DASHBOARD_HTML)

@dashboard_app.route('/api/rewards/stats')
def get_stats():
    """API endpoint for reward statistics."""
    try:
        reward_system = get_reward_system()
        feedback_summary = reward_system.get_feedback_summary()
        rl_stats = reward_system.get_rl_stats()

        return jsonify({
            'feedback_summary': feedback_summary,
            'rl_stats': rl_stats,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def run_dashboard(host='0.0.0.0', port=5001):
    """Run the dashboard server."""
    print(f"🚀 Starting TaskFlow AI Dashboard on http://{host}:{port}")
    print("📊 Available endpoints:")
    print("  GET  / - Main dashboard")
    print("  GET  /api/rewards/stats - Reward statistics API")
    dashboard_app.run(host=host, port=port, debug=False)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='TaskFlow AI Reward Dashboard')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5001, help='Port to bind to')

    args = parser.parse_args()
    run_dashboard(host=args.host, port=args.port)