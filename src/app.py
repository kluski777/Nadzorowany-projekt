"""
Flask application for latent space inpainting inference.

Provides a web interface to upload masked images and get reconstructed results.
"""

import io

from flask import Flask, render_template, request, send_file, jsonify
from PIL import Image

from pipeline import InferencePipeline

app = Flask(__name__)

print("Initializing inference pipeline...")
pipeline = InferencePipeline()
print("Pipeline ready!")


@app.route("/", methods=["GET"])
def index():
    """Render the main upload page."""
    return render_template("index.html")


@app.route("/inpaint", methods=["POST"])
def inpaint():
    """
    Process an uploaded image and return the inpainted result.
    
    Expects:
        - POST request with 'image' file field
        
    Returns:
        - PNG image of the reconstructed result
    """
    # Check if image was uploaded
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files["image"]
    
    if file.filename == "":
        return jsonify({"error": "No image file selected"}), 400
    
    try:
        # Load image from upload
        image = Image.open(file.stream)
        print(f"Received image: {file.filename} ({image.size})")
        
        # Run inpainting pipeline
        print("Running inpainting pipeline...")
        result_image, cluster_id = pipeline.inpaint(image)
        print(f"Inpainting complete! Used cluster {cluster_id}")
        
        # Convert result to bytes
        img_bytes = io.BytesIO()
        result_image.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        
        return send_file(
            img_bytes,
            mimetype="image/png",
            as_attachment=False,
            download_name="inpainted.png",
        )
        
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "device": str(pipeline.device)})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8081)
