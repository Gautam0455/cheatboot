import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
#third changes hello gautam
from keras.models import load_model
model = load_model('model.h5')
import json
import random
intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)


if __name__ == "__main__":
    app.run()

# from flask import Flask, render_template, request, redirect, url_for, flash
# from flask_mysqldb import MySQL

# app = Flask(__name__)

# # MySQL Configuration
# app.config['MYSQL_HOST'] = 'localhost'
# app.config['MYSQL_USER'] = 'root'  # Replace with your MySQL username
# app.config['MYSQL_PASSWORD'] = ''  # Replace with your MySQL password
# app.config['MYSQL_DB'] = 'cheatboot'  # Replace with your MySQL database name
# app.config['MYSQL_USE_UNICODE'] = True
# app.config['MYSQL_CHARSET'] = 'utf8'
# app.config['MYSQL_CONNECT_TIMEOUT'] = 10

# # Uncomment these lines if you want to use SSL
# # app.config['MYSQL_SSL_CA'] = '/path/to/ca-cert.pem'  
# # app.config['MYSQL_SSL_CERT'] = '/path/to/client-cert.pem'
# # app.config['MYSQL_SSL_KEY'] = '/path/to/client-key.pem'

# mysql = MySQL(app)

# @app.route("/add_login", methods=['GET', 'POST'])
# def add_login():
#     if request.method == 'POST':
#         username = request.form['username']
#         password = request.form['password']
        
#         # Create a cursor
#         cur = mysql.connection.cursor()
        
#         try:
#             # Execute the insert query
#             cur.execute("INSERT INTO login (username, password) VALUES (%s, %s)", (username, password))
#             # Commit the changes
#             mysql.connection.commit()
#             flash('Login added successfully!')
#         except Exception as e:
#             mysql.connection.rollback()  # Rollback in case of error
#             flash(f'An error occurred: {str(e)}', 'danger')  # Flash the error message
#         finally:
#             # Close the cursor
#             cur.close()
        
#         return redirect(url_for('home'))

#     return render_template("add_login.html")


# @app.route("/")
# def home():
#     return render_template("index.html")

# if __name__ == "__main__":
#     app.run(debug=True)
