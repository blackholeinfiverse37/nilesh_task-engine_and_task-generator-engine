#!/usr/bin/env python3
"""
TaskFlow AI REST API Server
Provides REST endpoints for repository review and task generation.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import logging
from typing import Dict, Any, Optional

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from reviewer import TaskReviewer
from generator import TaskGenerator
from mini_lm import MiniLM
from rewards import RewardSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global instances (lazy-loaded)
_mini_lm = None
_reward_system = None
_reviewer = None
_generator = None

def get_mini_lm():
    """Lazy load MiniLM instance."""
    global _mini_lm
    if _mini_lm is None:
        logger.info("Initializing MiniLM...")
        _mini_lm = MiniLM()
    return _mini_lm

def get_reward_system():
    """Lazy load RewardSystem instance."""
    global _reward_system
    if _reward_system is None:
        logger.info("Initializing RewardSystem...")
        mini_lm = get_mini_lm()
        _reward_system = RewardSystem(mini_lm)
    return _reward_system

def get_reviewer():
    """Lazy load TaskReviewer instance."""
    global _reviewer
    if _reviewer is None:
        logger.info("Initializing TaskReviewer...")
        mini_lm = get_mini_lm()
        _reviewer = TaskReviewer(mini_lm)
    return _reviewer

def get_generator():
    """Lazy load TaskGenerator instance."""
    global _generator
    if _generator is None:
        logger.info("Initializing TaskGenerator...")
        mini_lm = get_mini_lm()
        _generator = TaskGenerator(mini_lm)
    return _generator

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "TaskFlow AI API",
        "version": "2.0"
    })

@app.route('/api/review', methods=['POST'])
def review_repository():
    """
    Review a GitHub repository.

    Expected JSON payload:
    {
        "repo_url": "https://github.com/user/repo",
        "metadata": {"optional": "data"}
    }
    """
    try:
        data = request.get_json()
        if not data or 'repo_url' not in data:
            return jsonify({
                "error": "Missing required field: repo_url",
                "example": {
                    "repo_url": "https://github.com/user/repo",
                    "metadata": {"optional": "data"}
                }
            }), 400

        repo_url = data['repo_url']
        metadata = data.get('metadata')

        logger.info(f"Reviewing repository: {repo_url}")

        # Get reviewer instance
        reviewer = get_reviewer()

        # Perform review
        result = reviewer.review(repo_url, metadata)

        logger.info(f"Review completed for {repo_url}")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Review failed: {e}")
        return jsonify({
            "error": str(e),
            "status": "failed"
        }), 500

@app.route('/api/generate', methods=['POST'])
def generate_task():
    """
    Generate next development task.

    Expected JSON payload:
    {
        "developer_id": "dev123",
        "skill_level": "intermediate",
        "last_task": "Implemented user authentication",
        "review": {
            "score": 8,
            "good_aspects": ["Good code structure"],
            "missing_aspects": ["Unit tests"]
        }
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "error": "Missing JSON payload",
                "example": {
                    "developer_id": "dev123",
                    "skill_level": "intermediate",
                    "last_task": "Implemented user authentication",
                    "review": {
                        "score": 8,
                        "good_aspects": ["Good code structure"],
                        "missing_aspects": ["Unit tests"]
                    }
                }
            }), 400

        # Extract required fields
        developer_id = data.get('developer_id', 'dev001')
        skill_level = data.get('skill_level', 'intermediate')
        last_task = data.get('last_task', 'Initial development')
        review = data.get('review', {"score": 7, "good_aspects": [], "missing_aspects": []})

        logger.info(f"Generating task for developer: {developer_id}")

        # Get generator instance
        generator = get_generator()

        # Generate task
        task = generator.generate_next_task(developer_id, skill_level, last_task, review)

        # Get reward system for evaluation
        reward_system = get_reward_system()

        # Evaluate the generation (optional)
        try:
            reward = reward_system.evaluate_output(
                f"Generate task for {developer_id} with skill {skill_level}",
                str(task),
                {}
            )
        except Exception as e:
            logger.warning(f"Reward evaluation failed: {e}")
            reward = 0

        result = {
            "task": task,
            "reward_score": reward,
            "generated_for": {
                "developer_id": developer_id,
                "skill_level": skill_level,
                "last_task": last_task
            }
        }

        logger.info(f"Task generated successfully for {developer_id}")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Task generation failed: {e}")
        return jsonify({
            "error": str(e),
            "status": "failed"
        }), 500

@app.route('/api/rewards/stats', methods=['GET'])
def get_reward_stats():
    """Get reward system statistics."""
    try:
        reward_system = get_reward_system()
        stats = reward_system.get_feedback_summary()
        rl_stats = reward_system.get_rl_stats()

        return jsonify({
            "feedback_summary": stats,
            "rl_stats": rl_stats
        })

    except Exception as e:
        logger.error(f"Failed to get reward stats: {e}")
        return jsonify({
            "error": str(e),
            "status": "failed"
        }), 500

@app.route('/api/pipeline', methods=['POST'])
def run_full_pipeline():
    """
    Run the complete TaskFlow AI pipeline.

    Expected JSON payload:
    {
        "repo_url": "https://github.com/user/repo",
        "developer_id": "dev123",
        "skill_level": "intermediate",
        "last_task": "Implemented basic features"
    }
    """
    try:
        data = request.get_json()
        if not data or 'repo_url' not in data:
            return jsonify({
                "error": "Missing required field: repo_url",
                "example": {
                    "repo_url": "https://github.com/user/repo",
                    "developer_id": "dev123",
                    "skill_level": "intermediate",
                    "last_task": "Implemented basic features"
                }
            }), 400

        repo_url = data['repo_url']
        developer_id = data.get('developer_id', 'dev001')
        skill_level = data.get('skill_level', 'intermediate')
        last_task = data.get('last_task', 'Initial development')

        logger.info(f"Running full pipeline for {repo_url}")

        # Step 1: Review repository
        reviewer = get_reviewer()
        review_result = reviewer.review(repo_url)

        # Step 2: Generate next task
        generator = get_generator()
        task = generator.generate_next_task(developer_id, skill_level, last_task, review_result)

        # Step 3: Get reward stats
        reward_system = get_reward_system()
        reward_stats = reward_system.get_feedback_summary()

        result = {
            "pipeline_status": "completed",
            "review": review_result,
            "next_task": task,
            "reward_stats": reward_stats,
            "metadata": {
                "developer_id": developer_id,
                "skill_level": skill_level,
                "repo_url": repo_url
            }
        }

        logger.info(f"Pipeline completed successfully for {repo_url}")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return jsonify({
            "error": str(e),
            "status": "failed"
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": [
            "/health",
            "/api/review",
            "/api/generate",
            "/api/rewards/stats",
            "/api/pipeline"
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error",
        "status": "failed"
    }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')

    logger.info(f"Starting TaskFlow AI API server on {host}:{port}")
    logger.info("Available endpoints:")
    logger.info("  GET  /health")
    logger.info("  POST /api/review")
    logger.info("  POST /api/generate")
    logger.info("  GET  /api/rewards/stats")
    logger.info("  POST /api/pipeline")

    app.run(host=host, port=port, debug=False)