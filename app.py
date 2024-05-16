import os
import subprocess
from flask import Flask, request, redirect, render_template, jsonify, url_for
from pyngrok import ngrok
import openai

app = Flask(__name__)
ngrok.set_auth_token("2fwBjfJtS5JtDaIVYLivlpZL7FW_2yYJK3xLhFkKbXYjGLqRo")
public_url =  ngrok.connect(5000).public_url

openai.api_key = 'sk-proj-zSxdjJAcwEwHtjeyG4E0T3BlbkFJFX5n5S9twVb60GCXiku6'

@app.get('/')
def home():
	return render_template('index.html')

@app.post('/detect')
def get_file():
	file_names = os.listdir('./static')
	for f in file_names:
		os.remove(f'./static/{f}')

	if 'img' not in request.files:
		return redirect('/')
    
	file = request.files['img']

	if file:
		file.save(os.path.join('./static', file.filename))
		img_path = os.path.join('./static', file.filename)
	
	result = subprocess.run(
    [
        "venv\Scripts\python.exe",
        "./inference.py",
        "--ckpt_path",
        "saved_model\maxvit_epoch=08_val_loss=0.13.ckpt",
        "--img_path",
        img_path,
    ],
    capture_output=True,
    text=True,
    )
	
	preds = result.stdout.replace("\n", "")
	question = "Please provide me with information and treatment for skin lesions with the scientific name - " + preds
	response = openai.ChatCompletion.create(
		model = "gpt-3.5-turbo",
		messages = [{
			"role": "user",
			"content": question
		}],
	)
	bot_response = response.choices[0].message.content.strip()
	if (preds != "Not infected") :
		bot_response = ""
	return jsonify({
		'static' : url_for('static', filename=file.filename),
		'preds' : preds,
		'bot_response' : bot_response
	})
	
if __name__ == "__main__":
	print(public_url)
	app.run(port=5000)