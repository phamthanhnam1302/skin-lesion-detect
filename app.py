import os
import subprocess
from flask import Flask, request, render_template, jsonify, url_for
from pyngrok import ngrok
import openai

app = Flask(__name__)
ngrok.set_auth_token("2fwBjfJtS5JtDaIVYLivlpZL7FW_2yYJK3xLhFkKbXYjGLqRo")
public_url =  ngrok.connect(5000).public_url

# openai.api_key = 'sk-proj-zSxdjJAcwEwHtjeyG4E0T3BlbkFJFX5n5S9twVb60GCXiku6'

@app.route('/')
def index():
	return render_template('index.html')

@app.route("/detect", methods=["POST"]) 
def detect():
	file_names = os.listdir('./static')
	for f in file_names:
		os.remove(f'./static/{f}')
	if 'img' not in request.files:
		return render_template('index.html')
    
	file = request.files['img']

	if file:
		file.save(os.path.join('./static', file.filename))
		img_path = os.path.join('./static', file.filename)
	
	result = subprocess.run(
    [
        "venv\Scripts\python.exe",
        "./inference.py",
        "--ckpt_path",
        "./saved_model/resnet_epoch=03_val_loss=0.14.ckpt",
        "--img_path",
        img_path,
    ],
    capture_output=True,
    text=True,
    )
	
	preds = result.stdout.replace("\n", "")
	# question = "Please provide me with information and treatment for skin lesions with the scientific name - " + preds
	# response = openai.ChatCompletion.create(
	# 	model = "gpt-3.5-turbo",
	# 	messages = [{
	# 		"role": "user",
	# 		"content": question
	# 	}],
	# )
	# bot_response = response.choices[0].message.content.strip()
	bot_response = "To protect your skin, follow these tips:\n\n1. Cleanse gently with a mild, fragrance-free cleanser.\n\n2. Moisturize daily according to your skin type.\n\n3. Apply broad-spectrum sunscreen with at least SPF 30 every day, reapplying every two hours if you're outside, and wearing protective clothing like long sleeves and hats.\n\n4. Stay hydrated by drinking plenty of water.\n\n5. Maintain a balanced diet rich in fruits, vegetables, and healthy fats to nourish your skin from within.\n\n6. Avoid smoking and limit alcohol consumption to prevent skin damage.\n\n7. Tailor your skincare routine to your skin type; for instance, use non-comedogenic products if you have oily skin or hypoallergenic products if you have sensitive skin.\n\n8. Incorporate gentle exfoliation into your routine once or twice a week to remove dead skin cells without over-exfoliating.\n\n9. Use lukewarm water instead of hot for bathing to avoid stripping your skin of natural oils.\n\n10. Get enough sleep to support skin repair.\n\n11. Consider using a humidifier in dry climates or during winter to maintain skin moisture.\n\n12. Regularly check your skin for any changes and schedule visits with a dermatologist for professional advice and monitoring.\n\n"
	# if (preds != "Not infected") :
	# 	bot_response = ""
	return jsonify({
		'static' : url_for('static', filename=file.filename),
		'preds' : preds,
		'bot_response' : bot_response
	})
	
if __name__ == "__main__":
	print(public_url)
	app.run(port=5000)
	# app.run(host="0.0.0.0", port=5000, ssl_context=('cert.pem', 'key.pem'))