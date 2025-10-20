from flask import Flask

app = Flask(__name__)  # this will be WSGI app that will be interacting with webserver

# decorator helps to declare the URLs
@app.route(rule = '/') # parameter rule takes url of the webpage we will open
def welcome(): # this function gets triggered as soon as we open the above mentioned URL
    return " Welcome to My first web page app"

# we add another decorator here
@app.route(rule = '/secondPage')
def secondPage():
    return "This is our second page"

if __name__ == '__main__':
    app.run() # here if we set debug = True, whenever we make changes to the code it will automatically restart the server.