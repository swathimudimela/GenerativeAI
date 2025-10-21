# Building URL dynamically

# Variable Rules and URL Building

# redirect helps us to redirect to different page

from flask import Flask,redirect,url_for

app = Flask(__name__)

@app.route(rule = '/')
def welcome():
    return "This is home page"

@app.route('/success/<int:score>') # the score will be passed in the url as an integer value
def success(score):
    return " The person passed with scored "+str(score)


@app.route('/Fail/<int:score>')
def fail(score):
    return " The person failed the test with score"+str(score)

@app.route("/Results/<int:marks>")
def Result(marks):
    result = ""
    if marks<50:
        result = "fail"
    else:
        result = "success"
    return redirect(url_for(result,score=marks))

if __name__ == '__main__':
    app.run(debug = True)
